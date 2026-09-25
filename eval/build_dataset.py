"""Build an eval snapshot: fetch quote + news once per ticker and save it.

Every model in the eval then judges identical inputs, and re-running the
comparison (e.g. after rewording a Jev question) costs no Finnhub quota.

With --from-scan, builds the snapshot from a run_scan report instead - no
Finnhub calls - optionally a random sample that skips tickers already
labeled in another universe, for a held-out test set.

Usage:
    python -m eval.build_dataset [--universe sp100] [--limit 20] [--tickers AAPL,MSFT]
    python -m eval.build_dataset --from-scan runs/scan-....json --sample 50 --exclude-universe sp100
"""

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from screener.scan import load_universe
from tools.news_tool import get_company_news
from tools.stock_data_tool import get_quote_snapshot

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_DIR = ROOT / "eval" / "data"

# One quote call + two news calls (profile2, company-news) per ticker, paced
# to stay under Finnhub's 60 calls/min free tier with some headroom.
FINNHUB_CALLS_PER_TICKER = 3
SECONDS_PER_TICKER = FINNHUB_CALLS_PER_TICKER * 60 / 55


def build_snapshot(tickers: list[str], universe: str) -> dict:
    items = []
    for i, ticker in enumerate(tickers, 1):
        started = time.monotonic()
        quote = get_quote_snapshot(ticker)
        news = get_company_news(ticker)
        items.append({"ticker": ticker, "quote": quote, "news": news})

        status = "ok" if quote["success"] and news["success"] else "FAILED"
        print(f"[{i}/{len(tickers)}] {ticker}: {status}, {len(news.get('headlines', []))} headlines")

        if i < len(tickers):
            time.sleep(max(0.0, SECONDS_PER_TICKER - (time.monotonic() - started)))

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "universe": universe,
        "items": items,
    }


def snapshot_from_scan(scan: dict, exclude: set[str], sample: int | None, seed: int) -> dict:
    """The tickers a scan screened successfully, in snapshot form.

    Sampling is seeded so the same command rebuilds the same set.
    """
    results = [r for r in scan["results"] if r.get("screen") and r["ticker"] not in exclude]
    if results and "headlines" not in results[0]:
        raise SystemExit("This scan predates headline saving - re-run run_scan.py to get one that has them.")
    if sample is not None and sample < len(results):
        results = random.Random(seed).sample(results, sample)
    items = [
        {
            "ticker": r["ticker"],
            "quote": {"success": True, "ticker": r["ticker"], "price": r.get("price"), "pct_change": r["pct_change"]},
            "news": {"success": True, "ticker": r["ticker"], "headlines": r["headlines"]},
        }
        for r in sorted(results, key=lambda r: r["ticker"])
    ]
    return {"created_at": scan["finished_at"], "universe": scan["universe"], "items": items}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--universe", default="sp100")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--tickers", help="Comma-separated tickers; overrides --universe/--limit")
    parser.add_argument("--from-scan", type=Path, help="Build from a runs/scan-*.json report instead of Finnhub")
    parser.add_argument("--sample", type=int, help="With --from-scan: random sample of N tickers")
    parser.add_argument("--exclude-universe", help="With --from-scan: skip tickers in this universe")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.from_scan:
        scan = json.loads(args.from_scan.read_text())
        exclude = set(load_universe(args.exclude_universe)) if args.exclude_universe else set()
        snapshot = snapshot_from_scan(scan, exclude, args.sample, args.seed)
        snapshot["source_scan"] = args.from_scan.name
    elif args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        snapshot = build_snapshot(tickers, "custom")
    else:
        snapshot = build_snapshot(load_universe(args.universe)[: args.limit], args.universe)

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    path = SNAPSHOT_DIR / f"snapshot-{stamp}.json"
    path.write_text(json.dumps(snapshot, indent=2))

    ok = sum(1 for it in snapshot["items"] if it["quote"]["success"] and it["news"]["success"])
    print(f"\nSaved {path.relative_to(ROOT)} ({ok}/{len(snapshot['items'])} tickers usable)")
    if ok == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
