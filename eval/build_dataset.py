"""Build an eval snapshot: fetch quote + news once per ticker and save it.

Every model in the eval then judges identical inputs, and re-running the
comparison (e.g. after rewording a Jev question) costs no Finnhub quota.

Usage: python -m eval.build_dataset [--universe sp100] [--limit 20] [--tickers AAPL,MSFT]
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from tools.news_tool import get_company_news
from tools.stock_data_tool import get_quote_snapshot

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
UNIVERSES_DIR = ROOT / "data" / "universes"
SNAPSHOT_DIR = ROOT / "eval" / "data"

# One quote call + two news calls (profile2, company-news) per ticker, paced
# to stay under Finnhub's 60 calls/min free tier with some headroom.
FINNHUB_CALLS_PER_TICKER = 3
SECONDS_PER_TICKER = FINNHUB_CALLS_PER_TICKER * 60 / 55


def load_universe(name: str) -> list[str]:
    path = UNIVERSES_DIR / f"{name}.txt"
    lines = path.read_text().splitlines()
    return [line.strip().upper() for line in lines if line.strip() and not line.startswith("#")]


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--universe", default="sp100")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--tickers", help="Comma-separated tickers; overrides --universe/--limit")
    args = parser.parse_args()

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        universe = "custom"
    else:
        tickers = load_universe(args.universe)[: args.limit]
        universe = args.universe

    snapshot = build_snapshot(tickers, universe)

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    path = SNAPSHOT_DIR / f"snapshot-{stamp}.json"
    path.write_text(json.dumps(snapshot, indent=2))

    ok = sum(1 for it in snapshot["items"] if it["quote"]["success"] and it["news"]["success"])
    print(f"\nSaved {path.relative_to(ROOT)} ({ok}/{len(tickers)} tickers usable)")
    if ok == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
