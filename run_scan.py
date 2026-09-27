"""Scan a universe of tickers: Jev screens each one, flagged ones get a brief.

Writes runs/scan-<timestamp>.json. Finnhub's 60 calls/min free tier sets the
pace: about 5-6 min for the S&P 100 and 25-30 min for the S&P 500.

Usage:
    python run_scan.py --universe sp500 --no-escalate      # Jev only, ~2.5 cents
    python run_scan.py --universe sp100 --max-briefs 5     # brief the top 5 escalations
    python run_scan.py --tickers AAPL,MSFT,NVDA
"""

import argparse
import asyncio
import sys

from dotenv import load_dotenv

from schemas import TickerScanResult
from screener.scan import ROOT, load_universe, run_scan, save_report

load_dotenv()


def _print_progress(done: int, total: int, result: TickerScanResult) -> None:
    if result.screen:
        s = result.screen
        detail = f"{s.sentiment} ({s.sentiment_confidence:.2f}) material={s.material_event:.2f} needs={s.needs_analysis:.2f}"
    else:
        detail = f"ERROR {result.screen_error}"
    flag = "ESCALATE" if result.decision.escalate else "skip"
    print(f"[{done}/{total}] {result.ticker}: {detail} -> {flag}", flush=True)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--universe", default="sp100", help="A data/universes/<name>.txt list")
    parser.add_argument("--tickers", help="Comma-separated tickers; overrides --universe")
    parser.add_argument("--limit", type=int, help="Scan only the first N tickers")
    parser.add_argument("--no-escalate", action="store_true", help="Record decisions without running briefs")
    parser.add_argument("--max-briefs", type=int, help="Cap on Gemini briefs this run (strongest signals first)")
    args = parser.parse_args()

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        universe = "custom"
    else:
        tickers = load_universe(args.universe)
        universe = args.universe
    tickers = tickers[: args.limit]

    escalate = not args.no_escalate
    mode = "Jev only" if not escalate else f"briefs capped at {args.max_briefs}" if args.max_briefs is not None else "briefs uncapped"
    print(f"Scanning {len(tickers)} tickers from {universe} ({mode})", flush=True)

    report = await run_scan(
        tickers,
        universe,
        escalate=escalate,
        max_briefs=args.max_briefs,
        on_result=_print_progress,
    )

    path = save_report(report)

    failed = sum(1 for r in report.results if not r.screen)
    brief_errors = [r for r in report.results if r.brief_error]
    print(
        f"\nScanned {report.tickers_scanned} ({failed} screening failures), "
        f"escalated {report.tickers_escalated} ({report.escalation_rate:.0%}), "
        f"briefs {report.briefs_generated}"
        + (f" ({len(brief_errors)} failed)" if brief_errors else "")
    )
    print(
        f"Jev ${report.jev_cost_usd:.4f} {'estimated' if report.jev_cost_is_estimate else 'exact'}, Gemini ~${report.gemini_cost_usd_estimate:.4f} estimated, "
        f"Jev latency p50 {report.jev_latency_p50_ms} ms / p95 {report.jev_latency_p95_ms} ms"
    )
    print(f"Wrote {path.relative_to(ROOT)}")
    if report.tickers_scanned and failed == report.tickers_scanned:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
