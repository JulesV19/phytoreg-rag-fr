"""
Interface CLI du LLM phytosanitaire.
Lance une session interactive dans le terminal.

Usage :
    python -m src.app.main
    python -m src.app.main --no-stream   # désactive le streaming
"""

import argparse
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich import box

from ..rag.chain import PhytoRAG

console = Console()

BANNER = """
╔══════════════════════════════════════════════════════╗
║      LLM Phytosanitaire — Assistant Réglementaire    ║
║      Basé sur la base AMM nationale + textes de loi  ║
╚══════════════════════════════════════════════════════╝
"""

HELP_TEXT = """
**Exemples de questions :**
- Puis-je utiliser l'ALIETTE FLASH sur des noyers ?
- Quel est le délai de rentrée après application d'un produit H319 ?
- Quelle est la ZNT minimale près des points d'eau ?
- Quels produits de biocontrôle sont autorisés contre les pucerons ?
- Le Fosétyl-Al est-il une substance active approuvée en Europe ?
- Quelles sont les règles pour vider les fonds de cuve ?

Tapez **quitter** ou **exit** pour terminer.
"""


def print_sources(sources: list[dict]):
    """Affiche un tableau récapitulatif des sources utilisées."""
    if not sources:
        return

    table = Table(
        title="Sources consultées",
        box=box.SIMPLE,
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("N°", style="dim", width=3)
    table.add_column("Source", width=22)
    table.add_column("Produit / Section", width=25)
    table.add_column("Extrait", width=50)

    source_labels = {
        "amm_xml": "Base AMM nationale",
        "arrete_2017": "Arrêté 4 mai 2017",
        "decision_amm_individuelle": "Décision AMM",
        "note_biocontrole": "Note biocontrôle DGAL",
        "substances_actives_xlsx": "Substances actives CE",
    }

    for i, src in enumerate(sources, 1):
        label = source_labels.get(src.get("source", ""), src.get("source", ""))
        produit_section = (
            src.get("nom_produit")
            or src.get("section")
            or src.get("article", "")
        )
        if src.get("numero_amm"):
            produit_section += f" (AMM {src['numero_amm']})"

        table.add_row(
            str(i),
            label,
            produit_section[:25],
            src.get("extrait", "")[:50],
        )

    console.print(table)


def run_cli(stream: bool = True, backend: str = "local"):
    console.print(BANNER, style="bold green")
    console.print(Markdown(HELP_TEXT))

    # Initialisation du RAG
    with console.status("[bold cyan]Connexion à Qdrant et chargement du modèle...[/bold cyan]"):
        try:
            rag = PhytoRAG(backend=backend)
        except Exception as e:
            console.print(f"[red]Erreur d'initialisation : {e}[/red]")
            console.print("[yellow]Vérifiez que Qdrant tourne (docker compose up -d) "
                          "et que l'ingestion a été effectuée (python -m src.ingestion.ingest).[/yellow]")
            sys.exit(1)

    console.print("[green]Prêt. Posez vos questions.[/green]\n")

    while True:
        try:
            question = console.input("[bold yellow]Vous >[/bold yellow] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Session terminée.[/dim]")
            break

        if not question:
            continue
        if question.lower() in ("quitter", "exit", "quit", "q"):
            console.print("[dim]Au revoir.[/dim]")
            break

        console.print()

        if stream:
            # Affichage en streaming
            console.print("[bold blue]Assistant >[/bold blue] ", end="")
            full_answer = ""
            try:
                for chunk in rag.stream(question):
                    console.print(chunk, end="", highlight=False)
                    full_answer += chunk
                console.print("\n")
            except Exception as e:
                console.print(f"\n[red]Erreur lors de la génération : {e}[/red]")
        else:
            # Affichage complet avec sources
            with console.status("[dim]Recherche et génération en cours...[/dim]"):
                try:
                    result = rag.ask(question)
                except Exception as e:
                    console.print(f"[red]Erreur : {e}[/red]\n")
                    continue

            console.print(Panel(
                Markdown(result["answer"]),
                title="[bold blue]Réponse[/bold blue]",
                border_style="blue",
            ))
            print_sources(result["sources"])

        console.print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Phytosanitaire — CLI")
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Désactive le streaming (affiche réponse complète + sources)",
    )
    parser.add_argument(
        "--llm",
        choices=["local", "api"],
        default="local",
        help="Backend LLM pour la génération : local (Ollama 7B) ou api (Mistral API)",
    )
    args = parser.parse_args()
    run_cli(stream=not args.no_stream, backend=args.llm)
