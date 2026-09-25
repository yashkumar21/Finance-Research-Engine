"""Hand-label a snapshot's tickers: sentiment + material event.

Blind by design - shows only the price move and headlines, never Jev's or
Gemini's answers, so labels can't be anchored to either model. Uses the
exact definitions Jev is asked about. Saves after every label and resumes
where you left off; quit any time with q.

Usage: python -m eval.label [--snapshot eval/data/snapshot-....json] [--relabel AAPL]
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from eval.compare import LABELS_PATH, latest_snapshot
from tools.jev_screen import QUESTIONS, SENTIMENT_CRITERIA

ROOT = Path(__file__).resolve().parent.parent

SENTIMENT_KEYS = {"u": "bullish", "d": "bearish", "n": "neutral"}


def load_labels(snapshot_name: str) -> dict:
    if LABELS_PATH.exists():
        data = json.loads(LABELS_PATH.read_text())
        if data.get("snapshot") != snapshot_name:
            raise SystemExit(
                f"{LABELS_PATH.relative_to(ROOT)} holds labels for {data.get('snapshot')}, not "
                f"{snapshot_name}. Pass --snapshot {data.get('snapshot')} to keep labeling it, "
                "or move the file aside to start a new set."
            )
        return data
    return {"snapshot": snapshot_name, "labels": {}}


def save_labels(data: dict) -> None:
    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    LABELS_PATH.write_text(json.dumps(data, indent=2))


def ask(prompt: str, valid: set[str]) -> str:
    while True:
        answer = input(prompt).strip().lower()
        if answer in valid:
            return answer
        print(f"  Please enter one of: {', '.join(sorted(valid))}")


def show(item: dict, position: str) -> None:
    quote, news = item["quote"], item["news"]
    move = quote.get("pct_change")
    print("\n" + "=" * 78)
    print(f"{position}  {item['ticker']}   today's move: {move:+.2f}%" if move is not None else f"{position}  {item['ticker']}")
    print("=" * 78)
    headlines = news.get("headlines", [])
    if not headlines:
        print("  (no headlines in the last 7 days)")
    for i, h in enumerate(headlines, 1):
        print(f"  {i:2d}. {' '.join(h['headline'].split())}")  # collapse stray \r/\n in feed text
        print(f"      {h['source']} - {h['published_at'][:10]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--relabel", help="Re-label one ticker that was already labeled")
    args = parser.parse_args()

    snapshot_path = args.snapshot or latest_snapshot()
    snapshot = json.loads(snapshot_path.read_text())
    data = load_labels(snapshot_path.name)
    labels = data["labels"]

    items = [it for it in snapshot["items"] if it["quote"]["success"] and it["news"]["success"]]
    if args.relabel:
        items = [it for it in items if it["ticker"] == args.relabel.upper()]
        labels.pop(args.relabel.upper(), None)
    todo = [it for it in items if it["ticker"] not in labels]

    print(f"Snapshot {snapshot_path.name}: {len(labels)} labeled, {len(todo)} to go.")
    print("\nSentiment - the overall tone of the company's recent news coverage:")
    for key, label in SENTIMENT_KEYS.items():
        print(f"  {key} = {label:8s} {SENTIMENT_CRITERIA[label]}")
    print(f"\nMaterial event - {QUESTIONS['material_event']['instructions']}:")
    print("  y = yes, n = no")
    print("\nAt any prompt: s = skip this ticker, q = save and quit.")

    for i, item in enumerate(todo, 1):
        show(item, f"[{len(labels) + 1}/{len(items)}]")

        sentiment = ask("\nSentiment (u/d/n, s, q): ", set(SENTIMENT_KEYS) | {"s", "q"})
        if sentiment == "q":
            break
        if sentiment == "s":
            continue
        material = ask("Material event (y/n, s, q): ", {"y", "n", "s", "q"})
        if material == "q":
            break
        if material == "s":
            continue

        labels[item["ticker"]] = {
            "sentiment": SENTIMENT_KEYS[sentiment],
            "material_event": material == "y",
            "labeled_at": datetime.now(timezone.utc).isoformat(),
        }
        save_labels(data)

    save_labels(data)
    print(f"\nSaved {len(labels)} labels to {LABELS_PATH}.")


if __name__ == "__main__":
    main()
