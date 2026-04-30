"""
Benchmark automatique du RAG phytosanitaire.

Usage :
    python -m tests.benchmark
    python -m tests.benchmark --cases tests/cases.json --output tests/report.json

Le rapport JSON contient :
- Un résumé global et par catégorie en tête de fichier
- Le détail de chaque cas : pass/fail, checks, réponse LLM, sources retournées
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Ajout du répertoire racine au path pour permettre les imports src.*
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.chain import PhytoRAG


# ─── Assertions ──────────────────────────────────────────────────────────────

def check_retrieval(result: dict, spec: dict) -> dict:
    sources = result["sources"]
    n = len(sources)
    checks = {}

    min_s = spec.get("min_sources")
    max_s = spec.get("max_sources")

    if min_s is not None and max_s is not None and max_s == 0:
        # Cas hors-domaine ou négatif : on attend 0 source
        passed = n == 0
        checks["source_count_zero"] = {
            "expected": 0,
            "actual": n,
            "passed": passed,
        }
    elif min_s is not None:
        passed = n >= min_s
        checks["min_sources"] = {
            "expected": min_s,
            "actual": n,
            "passed": passed,
        }

    expected_types = spec.get("source_types", [])
    if expected_types:
        actual_types = {s.get("source", "") for s in sources}
        missing = [t for t in expected_types if t not in actual_types]
        checks["source_types"] = {
            "expected": expected_types,
            "actual": sorted(actual_types),
            "missing": missing,
            "passed": len(missing) == 0,
        }

    expected_amm = spec.get("amm")
    if expected_amm:
        found_amm = any(s.get("numero_amm") == expected_amm for s in sources)
        checks["amm_present"] = {
            "expected": expected_amm,
            "passed": found_amm,
        }

    return checks


def check_answer(answer: str, spec: dict) -> dict:
    lower = answer.lower()
    checks = {}

    must_contain = spec.get("must_contain", [])
    if must_contain:
        missing = [kw for kw in must_contain if kw.lower() not in lower]
        checks["must_contain"] = {
            "expected": must_contain,
            "missing": missing,
            "passed": len(missing) == 0,
        }

    must_not_contain = spec.get("must_not_contain", [])
    if must_not_contain:
        found = [kw for kw in must_not_contain if kw.lower() in lower]
        checks["must_not_contain"] = {
            "forbidden": must_not_contain,
            "found": found,
            "passed": len(found) == 0,
        }

    return checks


def run_case(rag: PhytoRAG, case: dict) -> dict:
    question = case["question"]
    try:
        result = rag.ask(question)
    except Exception as exc:
        return {
            "id": case["id"],
            "category": case["category"],
            "question": question,
            "passed": False,
            "error": str(exc),
            "checks": {},
            "answer": "",
            "sources": [],
        }

    retrieval_checks = check_retrieval(result, case["retrieval"])
    answer_checks = check_answer(result["answer"], case["answer"])
    all_checks = {**retrieval_checks, **answer_checks}

    answer_passed = all(c["passed"] for c in answer_checks.values()) if answer_checks else True
    retrieval_passed = all(c["passed"] for c in retrieval_checks.values()) if retrieval_checks else True

    return {
        "id": case["id"],
        "category": case["category"],
        "question": question,
        "passed": answer_passed,
        "retrieval_passed": retrieval_passed,
        "checks": all_checks,
        "answer": result["answer"],
        "sources": result["sources"],
    }


# ─── Statistiques ─────────────────────────────────────────────────────────────

def compute_stats(results: list[dict]) -> dict:
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    retrieval_ok = sum(1 for r in results if r.get("retrieval_passed", True))
    both_ok = sum(1 for r in results if r["passed"] and r.get("retrieval_passed", True))

    by_category: dict[str, dict] = {}
    for r in results:
        cat = r["category"]
        if cat not in by_category:
            by_category[cat] = {"total": 0, "passed": 0, "failed": 0, "retrieval_passed": 0}
        by_category[cat]["total"] += 1
        if r["passed"]:
            by_category[cat]["passed"] += 1
        else:
            by_category[cat]["failed"] += 1
        if r.get("retrieval_passed", True):
            by_category[cat]["retrieval_passed"] += 1

    failed_ids = [r["id"] for r in results if not r["passed"]]
    retrieval_failed_ids = [r["id"] for r in results if not r.get("retrieval_passed", True)]

    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(passed / total, 3) if total else 0,
        "retrieval_pass_rate": round(retrieval_ok / total, 3) if total else 0,
        "both_pass_rate": round(both_ok / total, 3) if total else 0,
        "by_category": by_category,
        "failed_ids": failed_ids,
        "retrieval_failed_ids": retrieval_failed_ids,
    }


# ─── Affichage terminal ───────────────────────────────────────────────────────

def print_progress(case_id: str, passed: bool, retrieval_passed: bool, n: int, total: int) -> None:
    if passed and retrieval_passed:
        symbol = "✓"
    elif passed:
        symbol = "~"  # réponse correcte, mauvaise source
    else:
        symbol = "✗"
    print(f"  [{n:02d}/{total}] {symbol} {case_id}", flush=True)


def print_summary(stats: dict) -> None:
    print()
    print("─" * 55)
    print(f"  Réponses  : {stats['passed']}/{stats['total']} ({stats['pass_rate']*100:.1f}%)")
    print(f"  Retrieval : {stats['total'] - len(stats['retrieval_failed_ids'])}/{stats['total']} ({stats['retrieval_pass_rate']*100:.1f}%)")
    print(f"  Les deux  : {round(stats['both_pass_rate'] * stats['total'])}/{stats['total']} ({stats['both_pass_rate']*100:.1f}%)")
    print()
    for cat, s in stats["by_category"].items():
        ret_ok = s.get("retrieval_passed", s["total"])
        bar = "✓" * s["passed"] + "✗" * s["failed"]
        print(f"  {cat:<24} rép {s['passed']}/{s['total']}  src {ret_ok}/{s['total']}  {bar}")
    if stats["failed_ids"]:
        print()
        print(f"  Réponses échouées  : {', '.join(stats['failed_ids'])}")
    if stats["retrieval_failed_ids"]:
        print(f"  Retrieval échoués  : {', '.join(stats['retrieval_failed_ids'])}")
    print("─" * 55)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark RAG phytosanitaire")
    parser.add_argument(
        "--cases",
        default="tests/cases.json",
        help="Fichier JSON des cas de test (défaut: tests/cases.json)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Fichier JSON de sortie (défaut: tests/report_YYYYMMDD_HHMMSS.json)",
    )
    parser.add_argument(
        "--filter",
        default=None,
        help="Ne lancer que les cas d'une catégorie (ex: regulation, amm_xml)",
    )
    parser.add_argument(
        "--llm",
        choices=["local", "api"],
        default="local",
        help="Backend LLM pour la génération : local (Ollama 7B) ou api (Mistral API)",
    )
    args = parser.parse_args()

    cases_path = Path(args.cases)
    if not cases_path.exists():
        print(f"Erreur : fichier de cas introuvable : {cases_path}", file=sys.stderr)
        sys.exit(1)

    with open(cases_path, encoding="utf-8") as f:
        cases = json.load(f)

    if args.filter:
        # Support exact match ("arvalis_produits") et prefix ("arvalis" → tous les arvalis_*)
        cases = [c for c in cases if c["category"] == args.filter or c["category"].startswith(args.filter + "_")]
        if not cases:
            print(f"Aucun cas pour le filtre '{args.filter}'", file=sys.stderr)
            sys.exit(1)

    print(f"\nInitialisation du RAG… (backend: {args.llm})")
    rag = PhytoRAG(backend=args.llm)
    print(f"Lancement de {len(cases)} cas de test\n")

    api_delay = 30 if args.llm == "api" else 0

    results = []
    for i, case in enumerate(cases, 1):
        if api_delay and i > 1:
            time.sleep(api_delay)
        result = run_case(rag, case)
        results.append(result)
        print_progress(case["id"], result["passed"], result.get("retrieval_passed", True), i, len(cases))

    stats = compute_stats(results)
    print_summary(stats)

    report = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "cases_file": str(cases_path),
        "summary": stats,
        "results": results,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output) if args.output else Path(f"tests/report_{timestamp}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n  Rapport écrit dans : {output_path}")
    sys.exit(0 if stats["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
