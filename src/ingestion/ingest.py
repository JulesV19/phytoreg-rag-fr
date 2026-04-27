"""
Script principal d'ingestion.
Parse tous les documents, génère les embeddings via Ollama (mxbai-embed-large),
et indexe dans Qdrant.

Usage :
    python -m src.ingestion.ingest
    python -m src.ingestion.ingest --reset      # repart de zéro
    python -m src.ingestion.ingest --only-bio   # re-indexe uniquement la note biocontrôle

Pour les sources non-XML (PDFs, XLSX), ingest lit en priorité les fichiers
parsing_cache/<source>.json produits par parse_preview.py. Si le cache est absent,
il parse à la volée. Workflow recommandé :

    python -m src.ingestion.parse_preview     # inspecter visuellement les chunks
    python -m src.ingestion.ingest            # indexer depuis le cache
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode

from src.embeddings import MxbaiEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, SparseIndexParams, SparseVectorParams, VectorParams
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from .parse_xml import parse_amm_xml
from .parse_pdfs import (
    parse_arrete_2017, parse_decision_amm, parse_note_biocontrole,
    parse_arvalis_produits, parse_arvalis_varietes,
    parse_arvalis_fertilisants, parse_arvalis_couverts,
)
from .parse_xlsx import parse_substances_actives

console = Console()

# ─── Configuration ────────────────────────────────────────────────────────────

DATA_DIR    = Path(__file__).parent.parent.parent / "Data"
CACHE_DIR   = Path(__file__).parent.parent.parent / "parsing_cache"
ARVALIS_DIR = DATA_DIR / "arvalis_pdfs"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "phyto_docs"
EMBEDDING_MODEL = "mxbai-embed-large"
EMBEDDING_DIM = 1024  # dimension de mxbai-embed-large
BATCH_SIZE = 64           # nombre de documents embeddés par lot
MAX_CHUNK_CHARS = 1400    # mxbai-embed-large : fenêtre 512 tokens ≈ ~1500 chars
SPARSE_VECTOR_NAME = "sparse"  # nom du champ sparse dans la collection Qdrant

# Splitter pour re-découper les chunks trop longs
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=MAX_CHUNK_CHARS,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""],
)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _to_langchain_docs(chunk) -> list[Document]:
    """
    Convertit un chunk en un ou plusieurs Documents LangChain.
    Si le texte dépasse MAX_CHUNK_CHARS, il est re-découpé en gardant les métadonnées.
    """
    if len(chunk.text) <= MAX_CHUNK_CHARS:
        return [Document(page_content=chunk.text, metadata=chunk.metadata)]

    sub_texts = _splitter.split_text(chunk.text)
    return [
        Document(
            page_content=sub,
            metadata={**chunk.metadata, "chunk_index": i},
        )
        for i, sub in enumerate(sub_texts)
    ]


@dataclass
class _CachedChunk:
    text: str
    metadata: dict = field(default_factory=dict)


def _load_chunks(source_name: str, fallback_parser, fallback_path: Path):
    """
    Charge les chunks depuis parsing_cache/<source_name>.json si disponible,
    sinon exécute fallback_parser(fallback_path) et avertit.
    """
    cache_file = CACHE_DIR / f"{source_name}.json"
    if cache_file.exists():
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        chunks = [_CachedChunk(text=c["text"], metadata=c["metadata"]) for c in data["chunks"]]
        console.print(f"    [dim]Cache : {cache_file.name} ({len(chunks)} chunks)[/dim]")
        return chunks
    console.print(f"    [yellow]Cache absent → parsing à la volée (lance parse_preview.py pour inspecter)[/yellow]")
    return fallback_parser(str(fallback_path))


def _init_collection(client: QdrantClient, reset: bool = False):
    """Crée ou recrée la collection Qdrant avec vecteurs dense (cosine) + sparse (BM25)."""
    exists = any(c.name == COLLECTION_NAME for c in client.get_collections().collections)
    if exists and reset:
        client.delete_collection(COLLECTION_NAME)
        console.print(f"[yellow]Collection '{COLLECTION_NAME}' supprimée.[/yellow]")
        exists = False
    if not exists:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: SparseVectorParams(index=SparseIndexParams()),
            },
        )
        console.print(f"[green]Collection '{COLLECTION_NAME}' créée (dense + sparse BM25).[/green]")
    else:
        console.print(f"[cyan]Collection '{COLLECTION_NAME}' existante conservée.[/cyan]")


def _ingest_batch(docs: list[Document], vectorstore: QdrantVectorStore, progress, task):
    """Indexe les documents par lots."""
    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i : i + BATCH_SIZE]
        vectorstore.add_documents(batch)
        progress.advance(task, advance=len(batch))


# ─── Pipeline d'ingestion ─────────────────────────────────────────────────────

def run_ingestion(reset: bool = False, only_bio: bool = False, only_arvalis: bool = False):
    console.rule("[bold blue]Pipeline d'ingestion — LLM Phyto[/bold blue]")

    # Connexion Qdrant
    try:
        client = QdrantClient(url=QDRANT_URL)
        client.get_collections()
    except Exception as e:
        console.print(f"[red]Impossible de se connecter à Qdrant ({QDRANT_URL}).[/red]")
        console.print("[yellow]Démarrez Qdrant avec : docker compose up -d[/yellow]")
        sys.exit(1)

    if only_bio:
        console.print("[yellow]Mode --only-bio : suppression des points note_biocontrole...[/yellow]")
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value="note_biocontrole"))]
            ),
        )
        console.print("[green]Points supprimés.[/green]")
    elif only_arvalis:
        console.print("[yellow]Mode --only-arvalis : suppression des points arvalis_*...[/yellow]")
        for src in ("arvalis_produits", "arvalis_varietes", "arvalis_fertilisants", "arvalis_couverts"):
            client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=Filter(
                    must=[FieldCondition(key="metadata.source", match=MatchValue(value=src))]
                ),
            )
        console.print("[green]Points supprimés.[/green]")
    else:
        _init_collection(client, reset=reset)

    # Embeddings : dense (Ollama) + sparse BM25 (FastEmbed local, CPU)
    embeddings = MxbaiEmbeddings(model=EMBEDDING_MODEL)
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        sparse_vector_name=SPARSE_VECTOR_NAME,
    )

    if not only_bio and not only_arvalis:
        # ── 1. XML AMM (source principale, volumineuse) ──────────────────────────
        xml_path = DATA_DIR / "decisionamm-intrant-format-xml-20260407" / \
                   "decision_intrant_opendata_20260407_1775584949185.xml"

        if xml_path.exists():
            console.print("\n[bold]1/4 Parsing XML AMM...[/bold]")
            console.print(f"    Fichier : {xml_path.name} ({xml_path.stat().st_size / 1e6:.0f} MB)")
            console.print("    [dim]Parsing en streaming + embedding par lots — peut prendre plusieurs minutes.[/dim]")

            buffer: list[Document] = []
            total_xml = 0

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Indexation usages AMM...", total=None)

                for chunk in parse_amm_xml(str(xml_path)):
                    buffer.extend(_to_langchain_docs(chunk))
                    if len(buffer) >= BATCH_SIZE:
                        vectorstore.add_documents(buffer)
                        total_xml += len(buffer)
                        progress.advance(task, advance=len(buffer))
                        progress.update(task, description=f"Indexation AMM... {total_xml:,} docs")
                        buffer.clear()

                if buffer:
                    vectorstore.add_documents(buffer)
                    total_xml += len(buffer)

            console.print(f"    [green]✓ {total_xml:,} chunks XML indexés.[/green]")
        else:
            console.print("[yellow]XML AMM non trouvé, ignoré.[/yellow]")

        # ── 2. Arrêté du 4 mai 2017 ──────────────────────────────────────────────
        arrete_path = DATA_DIR / "20170504_AM_Utilisation_pdt_Phyto.pdf"
        if arrete_path.exists():
            console.print("\n[bold]2/4 Arrêté 4 mai 2017...[/bold]")
            chunks = _load_chunks("arrete_2017", parse_arrete_2017, arrete_path)
            docs = [doc for c in chunks for doc in _to_langchain_docs(c)]
            vectorstore.add_documents(docs)
            console.print(f"    [green]✓ {len(docs)} chunks indexés.[/green]")

        # ── 3. Décisions AMM individuelles ───────────────────────────────────────
        console.print("\n[bold]3/4 Parsing décisions AMM individuelles...[/bold]")
        decision_pdfs = list(DATA_DIR.glob("*.pdf"))
        decision_pdfs = [p for p in decision_pdfs if "AMM" in p.name or "DECISION" in p.name.upper()]
        # Exclure l'arrêté 2017 et la note biocontrôle
        decision_pdfs = [p for p in decision_pdfs if "20170504" not in p.name and "2026-168" not in p.name]

        total_decisions = 0
        for pdf_path in decision_pdfs:
            chunks = parse_decision_amm(str(pdf_path))
            docs = [doc for c in chunks for doc in _to_langchain_docs(c)]
            vectorstore.add_documents(docs)
            total_decisions += len(docs)
            console.print(f"    • {pdf_path.name} → {len(docs)} chunks")
        console.print(f"    [green]✓ {total_decisions} chunks de décisions AMM indexés.[/green]")

    # ── 4. Note biocontrôle + XLSX substances actives ─────────────────────────
    if not only_arvalis:
        step_label = "1/1" if only_bio else "4/4"
        console.print(f"\n[bold]{step_label} Parsing note biocontrôle + substances actives...[/bold]")

        bio_path = DATA_DIR / "2026-168_final (1).pdf"
        if bio_path.exists():
            chunks = _load_chunks("note_biocontrole", parse_note_biocontrole, bio_path)
            docs = [doc for c in chunks for doc in _to_langchain_docs(c)]
            vectorstore.add_documents(docs)
            console.print(f"    • Note biocontrôle → {len(docs)} chunks")

        xlsx_path = DATA_DIR / "ActiveSubstanceExport_11-04-2026.xlsx"
        if xlsx_path.exists():
            chunks = _load_chunks("substances_actives", parse_substances_actives, xlsx_path)
            docs = [doc for c in chunks for doc in _to_langchain_docs(c)]
            vectorstore.add_documents(docs)
            console.print(f"    • Substances actives XLSX → {len(docs)} chunks")

    # ── 5. Fiches ARVALIS ────────────────────────────────────────────────────
    if not only_bio:
        step_label = "1/1" if only_arvalis else "5/5"
        console.print(f"\n[bold]{step_label} Fiches ARVALIS (produits, variétés, fertilisants, couverts)...[/bold]")

        arvalis_sources = [
            ("arvalis_produits",     "Fiches produits",     parse_arvalis_produits,     ARVALIS_DIR / "produits"),
            ("arvalis_varietes",     "Fiches variétés",     parse_arvalis_varietes,     ARVALIS_DIR / "varietes"),
            ("arvalis_fertilisants", "Fiches fertilisants", parse_arvalis_fertilisants, ARVALIS_DIR / "fertilisants"),
            ("arvalis_couverts",     "Fiches couverts",     parse_arvalis_couverts,     ARVALIS_DIR / "couverts"),
        ]

        for source_name, label, parser_fn, source_path in arvalis_sources:
            if not source_path.exists():
                console.print(f"    [dim]{label} : répertoire introuvable, ignoré.[/dim]")
                continue
            chunks = _load_chunks(source_name, parser_fn, source_path)
            docs = [doc for c in chunks for doc in _to_langchain_docs(c)]
            vectorstore.add_documents(docs)
            console.print(f"    • {label} → {len(docs)} chunks indexés.")

    # ── Résumé final ──────────────────────────────────────────────────────────
    console.rule()
    count = client.count(COLLECTION_NAME).count
    console.print(f"[bold green]Ingestion terminée. {count:,} documents dans Qdrant.[/bold green]")


# ─── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingestion des données phytosanitaires")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Supprime et recrée la collection avant l'ingestion",
    )
    parser.add_argument(
        "--only-bio",
        action="store_true",
        help="Re-indexe uniquement la note biocontrôle (supprime les anciens points, ~2 min)",
    )
    parser.add_argument(
        "--only-arvalis",
        action="store_true",
        help="Re-indexe uniquement les fiches ARVALIS (produits, variétés, fertilisants, couverts)",
    )
    args = parser.parse_args()
    run_ingestion(reset=args.reset, only_bio=args.only_bio, only_arvalis=args.only_arvalis)
