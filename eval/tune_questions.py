"""Try alternative wordings of Jev's needs_analysis question on a snapshot.

Jev-only (no Gemini), so each variant costs about $0.001 for 20 tickers and
takes seconds. A useful question spreads its probabilities across tickers;
one that returns ~the same value for everything can't drive escalation.

Usage: python -m eval.tune_questions [--snapshot eval/data/snapshot-....json]
"""

import argparse
import json
import statistics
from pathlib import Path

from dotenv import load_dotenv

from eval.compare import latest_snapshot
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()

    snapshot = json.loads((args.snapshot or latest_snapshot()).read_text())
    items = [it for it in snapshot["items"] if it["quote"]["success"] and it["news"]["success"]]

    table = {}
    for name, instructions in NEEDS_ANALYSIS_VARIANTS.items():
        questions = {**QUESTIONS, "needs_analysis": {"type": "noul", "instructions": instructions}}
        values = {}
        for it in items:
            result = screen_ticker(it["ticker"], it["quote"], it["news"], questions=questions)
            values[it["ticker"]] = result["needs_analysis"] if result.get("success") else None
        table[name] = values
        vals = [v for v in values.values() if v is not None]
        print(
            f"{name:26s} min={min(vals):.2f} median={statistics.median(vals):.2f} "
            f"max={max(vals):.2f} stdev={statistics.pstdev(vals):.3f} "
            f">=0.6: {sum(v >= 0.6 for v in vals)}/{len(vals)}"
        )

    print("\nticker   " + "  ".join(f"{name[:8]:>8s}" for name in table))
    for it in items:
        t = it["ticker"]
        print(f"{t:8s} " + "  ".join(
            f"{table[name][t]:8.2f}" if table[name][t] is not None else f"{'ERR':>8s}" for name in table
        ))


if __name__ == "__main__":
    main()
