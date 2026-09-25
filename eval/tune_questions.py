"""Try alternative wordings of one of Jev's noul questions on a snapshot.

Jev-only (no Gemini), so each variant costs about $0.005 for 100 tickers and
takes a minute. A useful question spreads its probabilities across tickers;
one that returns ~the same value for everything can't drive escalation. When
the snapshot has hand labels, material_event variants are also scored against
the labeled material events (AUC, and recall/precision at 0.5).

Usage: python -m eval.tune_questions [--question material_event] [--snapshot eval/data/snapshot-....json]
"""

import argparse
import json
import statistics
from pathlib import Path

from dotenv import load_dotenv

from eval.compare import latest_snapshot, load_labels
from tools.jev_screen import QUESTIONS, screen_ticker

load_dotenv()

NEEDS_ANALYSIS_VARIANTS = {
    "v0_original": (
        "A professional equity analyst covering this company would want a full "
        "research brief on it today"
    ),
    "v1_new_info": (
        "The headlines contain new, company-specific information that could change this "
        "company's financial outlook - not routine coverage, general market commentary, "
        "or stock-price recaps"
    ),
    "v2_specific_development": (
        "At least one headline reports a specific new development at this company itself, "
        "rather than market commentary, stock-picking lists, price recaps, or articles that "
        "only mention the company in passing"
    ),
    "v3_update_view (adopted)": QUESTIONS["needs_analysis"]["instructions"],
}

MATERIAL_EVENT_VARIANTS = {
    "v0_original (adopted)": QUESTIONS["material_event"]["instructions"],
    "v1_broad_list": (
        "The headlines report a material event for this company: results or guidance, "
        "a merger, acquisition or partnership, a large order, contract or investment, "
        "a dividend or buyback, a new analyst rating or coverage, litigation, or a "
        "regulatory or policy decision that directly affects its business"
    ),
    "v2_concrete_development": (
        "At least one headline reports a concrete new development at this company itself - "
        "such as results, guidance, a deal, a large order or investment, a capital-return "
        "action, an analyst rating change, or a regulatory decision - not only market "
        "commentary, stock-picking lists, price recaps, or passing mentions"
    ),
    "v3_holder_would_want": (
        "The headlines contain company-specific news that someone holding this stock "
        "would want to know about today"
    ),
}

VARIANTS = {"needs_analysis": NEEDS_ANALYSIS_VARIANTS, "material_event": MATERIAL_EVENT_VARIANTS}


def auc(values: dict, positives: set) -> float | None:
    """Chance a labeled event scores above a labeled non-event (ties count half)."""
    pos = [v for t, v in values.items() if v is not None and t in positives]
    neg = [v for t, v in values.items() if v is not None and t not in positives]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--question", choices=list(VARIANTS), default="needs_analysis")
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()

    snapshot_path = args.snapshot or latest_snapshot()
    snapshot = json.loads(snapshot_path.read_text())
    labels = load_labels(snapshot_path.name)
    events = {t for t, lab in labels.items() if lab["material_event"]}
    items = [it for it in snapshot["items"] if it["quote"]["success"] and it["news"]["success"]]

    table = {}
    for name, instructions in VARIANTS[args.question].items():
        questions = {**QUESTIONS, args.question: {"type": "noul", "instructions": instructions}}
        values = {}
        for it in items:
            result = screen_ticker(it["ticker"], it["quote"], it["news"], questions=questions)
            values[it["ticker"]] = result[args.question] if result.get("success") else None
        table[name] = values
        vals = [v for v in values.values() if v is not None]
        line = (
            f"{name:26s} min={min(vals):.2f} median={statistics.median(vals):.2f} "
            f"max={max(vals):.2f} stdev={statistics.pstdev(vals):.3f} "
            f">=0.6: {sum(v >= 0.6 for v in vals)}/{len(vals)}"
        )
        if args.question == "material_event" and events:
            fired = {t for t, v in values.items() if v is not None and v >= 0.5}
            line += (
                f"  AUC={auc(values, events):.3f} recall@0.5={len(fired & events) / len(events):.2f} "
                f"precision@0.5={len(fired & events) / len(fired) if fired else 0:.2f}"
            )
        print(line)

    print("\nticker   " + "  ".join(f"{name[:8]:>8s}" for name in table))
    for it in items:
        t = it["ticker"]
        print(f"{t:8s} " + "  ".join(
            f"{table[name][t]:8.2f}" if table[name][t] is not None else f"{'ERR':>8s}" for name in table
        ))


if __name__ == "__main__":
    main()
