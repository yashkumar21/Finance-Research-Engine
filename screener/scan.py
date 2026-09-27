"""Scanner: screen a universe of tickers with Jev, escalate the flagged ones.

For each ticker: fetch the quote and headlines (rate-limited Finnhub calls),
screen with Jev, and apply the escalation policy. Escalated tickers then get
a full Gemini research brief, a few at a time - unless escalation is off, in
which case decisions are recorded but no brief runs, so a scan costs only
Jev calls and can be re-scored against a different policy later.
"""

import asyncio
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from agents.orchestrator import run_research_brief
from schemas import Decision, ScanReport, ScreenResult, TickerScanResult
from screener.policy import DEFAULT_POLICY, PolicyConfig, decide
from tools.jev_screen import screen_ticker
from tools.news_tool import get_company_news
from tools.rate_limit import TokenBucket
from tools.stock_data_tool import get_quote_snapshot

ROOT = Path(__file__).resolve().parent.parent
UNIVERSES_DIR = ROOT / "data" / "universes"
RUNS_DIR = ROOT / "runs"

# /quote, plus the news tool's /stock/profile2 and /company-news.
FINNHUB_CALLS_PER_TICKER = 3
# A brief's own Finnhub calls: resolve_ticker's /quote, get_stock_data's
# /quote and /stock/metric, and the news tool's two calls.
FINNHUB_CALLS_PER_BRIEF = 5
# Finnhub's rate limit is the bottleneck, so a few workers are enough to
# keep the bucket busy while Jev calls are in flight.
SCREEN_CONCURRENCY = 4
BRIEF_CONCURRENCY = 2
# Gemini's free tier allows 15 requests/min per model and one brief makes
# several, so two concurrent briefs regularly hit 429s; wait them out.
BRIEF_MAX_RETRIES = 3
BRIEF_RETRY_SECONDS = 30


def load_universe(name: str) -> list[str]:
    """Tickers from data/universes/<name>.txt, skipping blanks and # comments."""
    lines = (UNIVERSES_DIR / f"{name}.txt").read_text().splitlines()
    return [line.strip().upper() for line in lines if line.strip() and not line.startswith("#")]


def save_report(report: ScanReport) -> Path:
    """Write report to runs/scan-<timestamp>.json and return the path."""
    RUNS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    path = RUNS_DIR / f"scan-{stamp}.json"
    path.write_text(report.model_dump_json(indent=2))
    return path


def list_reports() -> list[Path]:
    """Saved scan reports, newest first."""
    return sorted(RUNS_DIR.glob("scan-*.json"), reverse=True)


async def _screen_one(ticker: str, finnhub: TokenBucket, config: PolicyConfig) -> TickerScanResult:
    await finnhub.acquire(1)
    quote = await asyncio.to_thread(get_quote_snapshot, ticker)
    await finnhub.acquire(2)
    news = await asyncio.to_thread(get_company_news, ticker)

    if not quote["success"] or not news["success"]:
        errors = [d["error"] for d in (quote, news) if not d["success"]]
        screen = {"success": False, "ticker": ticker, "error": f"data fetch failed: {'; '.join(errors)}"}
    else:
        screen = await asyncio.to_thread(screen_ticker, ticker, quote, news)

    return TickerScanResult(
        ticker=ticker,
        price=quote.get("price"),
        pct_change=quote.get("pct_change"),
        headline_count=len(news.get("headlines", [])),
        headlines=news.get("headlines", []),
        screen=ScreenResult(**screen) if screen["success"] else None,
        screen_error=None if screen["success"] else screen["error"],
        decision=decide(screen, quote.get("pct_change"), config),
    )


def _brief_priority(result: TickerScanResult, config: PolicyConfig) -> tuple:
    """Sort key for spending a limited brief budget: big price moves first,
    then the likeliest material events; failed screens last."""
    moved = result.pct_change is not None and abs(result.pct_change) >= config.price_move_threshold_pct
    material = result.screen.material_event if result.screen else -1.0
    return (not moved, -material)


async def _brief_one(result: TickerScanResult, finnhub: TokenBucket, slots: asyncio.Semaphore) -> None:
    async with slots:
        for attempt in range(BRIEF_MAX_RETRIES + 1):
            await finnhub.acquire(FINNHUB_CALLS_PER_BRIEF)
            try:
                result.brief = await run_research_brief(result.ticker, user_id="scanner")
                result.brief_error = None
                return
            except Exception as exc:  # one failed brief must not sink the scan
                result.brief_error = f"{type(exc).__name__}: {exc}"
                rate_limited = "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc)
                if not rate_limited or attempt == BRIEF_MAX_RETRIES:
                    return
                await asyncio.sleep(BRIEF_RETRY_SECONDS)


def _percentile(values: list[float], pct: int) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return round(values[0], 1)
    return round(statistics.quantiles(values, n=100, method="inclusive")[pct - 1], 1)


async def run_scan(
    tickers: list[str],
    universe: str,
    config: PolicyConfig = DEFAULT_POLICY,
    escalate: bool = True,
    max_briefs: Optional[int] = None,
    on_result: Optional[Callable[[int, int, TickerScanResult], None]] = None,
) -> ScanReport:
    """Screen every ticker, then brief the escalated ones.

    Args:
        tickers: Ticker symbols to scan, in order.
        universe: Name recorded on the report, e.g. "sp500".
        config: Escalation thresholds.
        escalate: When False, record decisions but run no briefs.
        max_briefs: Cap on briefs per run (None = no cap). When more tickers
            are escalated, the strongest signals are briefed first.
        on_result: Called with (done, total, result) as each ticker finishes
            screening - for progress output.
    """
    started_at = datetime.now(timezone.utc).isoformat()
    finnhub = TokenBucket()
    queue: asyncio.Queue[tuple[int, str]] = asyncio.Queue()
    for item in enumerate(tickers):
        queue.put_nowait(item)
    results: list[Optional[TickerScanResult]] = [None] * len(tickers)

    async def worker() -> None:
        while not queue.empty():
            i, ticker = queue.get_nowait()
            try:
                result = await _screen_one(ticker, finnhub, config)
            except Exception as exc:  # never silently drop a ticker
                error = f"{type(exc).__name__}: {exc}"
                result = TickerScanResult(
                    ticker=ticker,
                    screen_error=error,
                    decision=Decision(escalate=True, reasons=[f"scan failed ({error}) - escalating to be safe"]),
                )
            results[i] = result
            if on_result:
                on_result(sum(r is not None for r in results), len(tickers), result)

    await asyncio.gather(*(worker() for _ in range(SCREEN_CONCURRENCY)))

    escalated = [r for r in results if r.decision.escalate]
    if escalate:
        to_brief = sorted(escalated, key=lambda r: _brief_priority(r, config))[:max_briefs]
        slots = asyncio.Semaphore(BRIEF_CONCURRENCY)
        await asyncio.gather(*(_brief_one(r, finnhub, slots) for r in to_brief))

    screens = [r.screen for r in results if r.screen]
    brief_costs = [r.brief.usage.cost_usd for r in results if r.brief and r.brief.usage]
    return ScanReport(
        started_at=started_at,
        finished_at=datetime.now(timezone.utc).isoformat(),
        universe=universe,
        tickers_scanned=len(results),
        tickers_escalated=len(escalated),
        escalation_rate=round(len(escalated) / len(results), 3) if results else 0.0,
        escalation_enabled=escalate,
        max_briefs=max_briefs,
        briefs_generated=sum(1 for r in results if r.brief),
        jev_model=next((s.model for s in screens), None),
        jev_cost_usd=sum(s.usage.cost_usd for s in screens),
        jev_cost_is_estimate=any(s.usage.cost_is_estimate for s in screens),
        gemini_cost_usd_estimate=sum(brief_costs),
        baseline_all_briefs_cost_usd_estimate=(
            statistics.mean(brief_costs) * len(results) if brief_costs else None
        ),
        jev_latency_p50_ms=_percentile([s.latency_ms for s in screens], 50),
        jev_latency_p95_ms=_percentile([s.latency_ms for s in screens], 95),
        results=results,
    )
