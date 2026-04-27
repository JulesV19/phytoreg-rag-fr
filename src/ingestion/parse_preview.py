"""
Parse tous les documents non-XML et sérialise les chunks en JSON.

Usage :
    python -m src.ingestion.parse_preview          # tout parser
    python -m src.ingestion.parse_preview --source arrete_2017
    python -m src.ingestion.parse_preview --source note_biocontrole
    python -m src.ingestion.parse_preview --source substances_actives

Sorties (dans parsing_cache/) :
    arrete_2017.json
    note_biocontrole.json
    substances_actives.json

Format de chaque fichier :
    {"source": "...", "total": N, "chunks": [{"text": "...", "metadata": {...}}, ...]}

ingest.py lit ces fichiers s'ils existent, reparse sinon.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ingestion.parse_pdfs import (
    parse_arrete_2017, parse_note_biocontrole,
    parse_arvalis_produits, parse_arvalis_varietes,
    parse_arvalis_fertilisants, parse_arvalis_couverts,
)
from src.ingestion.parse_xlsx import parse_substances_actives

DATA_DIR      = Path(__file__).parent.parent.parent / "Data"
CACHE_DIR     = Path(__file__).parent.parent.parent / "parsing_cache"
ARVALIS_DIR   = DATA_DIR / "arvalis_pdfs"

SOURCES = {
    "arrete_2017": {
        "path": DATA_DIR / "20170504_AM_Utilisation_pdt_Phyto.pdf",
        "parser": lambda p: parse_arrete_2017(str(p)),
    },
    "note_biocontrole": {
        "path": DATA_DIR / "2026-168_final (1).pdf",
        "parser": lambda p: parse_note_biocontrole(str(p)),
    },
    "substances_actives": {
        "path": DATA_DIR / "ActiveSubstanceExport_11-04-2026.xlsx",
        "parser": lambda p: parse_substances_actives(str(p)),
    },
    "arvalis_produits": {
        "path": ARVALIS_DIR / "produits",
        "parser": lambda p: parse_arvalis_produits(str(p)),
    },
    "arvalis_varietes": {
        "path": ARVALIS_DIR / "varietes",
        "parser": lambda p: parse_arvalis_varietes(str(p)),
    },
    "arvalis_fertilisants": {
        "path": ARVALIS_DIR / "fertilisants",
        "parser": lambda p: parse_arvalis_fertilisants(str(p)),
    },
    "arvalis_couverts": {
        "path": ARVALIS_DIR / "couverts",
        "parser": lambda p: parse_arvalis_couverts(str(p)),
    },
}


def preview_source(name: str) -> Path:
    spec = SOURCES[name]
    if not spec["path"].exists():
        print(f"  [SKIP] Fichier introuvable : {spec['path']}")
        return None

    print(f"  Parsing {spec['path'].name}…", end=" ", flush=True)
    chunks = spec["parser"](spec["path"])
    print(f"{len(chunks)} chunks")

    out = CACHE_DIR / f"{name}.json"
    payload = {
        "source": name,
        "total": len(chunks),
        "chunks": [{"text": c.text, "metadata": c.metadata} for c in chunks],
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  → {out}")
    return out


def main():
    parser = argparse.ArgumentParser(description="Parse et sérialise les chunks non-XML")
    parser.add_argument(
        "--source",
        choices=list(SOURCES),
        default=None,
        help="Source à parser (défaut : toutes)",
    )
    args = parser.parse_args()

    CACHE_DIR.mkdir(exist_ok=True)

    targets = [args.source] if args.source else list(SOURCES)
    for name in targets:
        print(f"\n[{name}]")
        preview_source(name)

    print("\nDone. Inspecte parsing_cache/*.json avant de lancer l'ingestion.")


if __name__ == "__main__":
    main()
