"""Compare Jev against the Gemini Sentiment Agent on a cached snapshot.

For each usable ticker in the snapshot, both models judge the same cached
headlines: the real Sentiment Agent (Gemini, via ADK) with a news tool that
returns the cached payload, and Jev via screen_ticker. Reports sentiment
agreement, a confusion matrix, per-ticker cost and latency for each model,
the distribution of Jev's material_event / needs_analysis probabilities,
and how many tickers each escalation-policy variant would escalate.

Agreement with Gemini is not accuracy - Gemini is not ground truth. When
hand labels exist for the snapshot (eval/label.py), also reports each
model's sentiment accuracy, escalation recall/precision against labeled
material events, and a threshold sweep.

Usage: python -m eval.compare [--snapshot eval/data/snapshot-....json]
"""

import argparse
import asyncio
import itertools
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from google.adk.runners import InMemoryRunner
from google.genai import types

from agents.orchestrator import _parse_json
from agents.sentiment_agent import build_sentiment_agent
from pricing import gemini_cost
from screener.policy import DEFAULT_POLICY, PolicyConfig, decide
from tools.jev_screen import screen_ticker

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_DIR = ROOT / "eval" / "data"
RESULTS_DIR = ROOT / "eval" / "results"
# The first labeled set predates per-snapshot label files.
LEGACY_LABELS_PATH = SNAPSHOT_DIR / "labels.json"
APP_NAME = "finance_research_engine_eval"
LABELS = ["bullish", "bearish", "neutral"]
GEMINI_MAX_RETRIES = 3
GEMINI_RETRY_SECONDS = 30

POLICY_VARIANTS = {
    "default": DEFAULT_POLICY,
    "original (+ needs_analysis, low-confidence rules)": PolicyConfig(
        needs_analysis_threshold=0.6, min_sentiment_confidence=0.5
    ),
    "material_event only (no price rule)": PolicyConfig(price_move_threshold_pct=1000),
}

# Thresholds tried by the sweep; None turns a rule off. Price-move threshold
# stays at the default 4%.
SWEEP_GRID = {
    "material_event": [0.4, 0.5, 0.6, 0.7],
    "needs_analysis": [0.6, 0.65, 0.7, None],
    "min_sentiment_confidence": [None, 0.3, 0.4, 0.5],
}


def latest_snapshot() -> Path:
    snapshots = sorted(SNAPSHOT_DIR.glob("snapshot-*.json"))
    if not snapshots:
        raise SystemExit("No snapshot found - run: python -m eval.build_dataset")
    return snapshots[-1]


def load_cached_gemini(snapshot_name: str) -> dict:
    """Gemini results by ticker from the newest comparison of this snapshot."""
    for path in sorted(RESULTS_DIR.glob("compare-*.json"), reverse=True):
        previous = json.loads(path.read_text())
        if previous.get("snapshot") == snapshot_name:
            print(f"Reusing Gemini answers from {path.name}")
            return {r["ticker"]: r["gemini"] for r in previous["rows"] if r["gemini"].get("success")}
    print("No earlier comparison of this snapshot - running Gemini")
    return {}


async def run_gemini_sentiment(ticker: str, news: dict) -> dict:
    """Run the real Sentiment Agent on cached headlines; return label, cost, latency."""

    def get_company_news(ticker: str) -> dict:
        """Fetch recent news headlines for a stock ticker.

        Returns the raw headlines only (headline, source, url,
        published_at) and performs no sentiment analysis.

        Args:
            ticker: The stock ticker symbol to look up, e.g. "AAPL".
        """
        del ticker  # the signature mirrors the real tool, which ADK exposes to the model
        return news

    for attempt in range(GEMINI_MAX_RETRIES + 1):
        try:
            agent = build_sentiment_agent(news_tool=get_company_news)
            runner = InMemoryRunner(agent=agent, app_name=APP_NAME)
            session = await runner.session_service.create_session(app_name=APP_NAME, user_id="eval")
            message = types.Content(role="user", parts=[types.Part(text=f"Assess sentiment for {ticker}")])

            input_tokens = output_tokens = 0
            final_text = None
            started = time.perf_counter()
            async for event in runner.run_async(user_id="eval", session_id=session.id, new_message=message):
                usage = event.usage_metadata
                if usage:
                    input_tokens += usage.prompt_token_count or 0
                    output_tokens += (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if part.text:
                            final_text = part.text
            latency_ms = (time.perf_counter() - started) * 1000

            parsed = _parse_json(final_text or "")
            return {
                "success": bool(parsed.get("success")),
                "sentiment": parsed.get("sentiment"),
                "latency_ms": round(latency_ms, 1),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": gemini_cost(input_tokens, output_tokens),
            }
        except Exception as exc:
            rate_limited = "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc)
            if rate_limited and attempt < GEMINI_MAX_RETRIES:
                print(f"    Gemini rate-limited, retrying in {GEMINI_RETRY_SECONDS}s...")
                await asyncio.sleep(GEMINI_RETRY_SECONDS)
                continue
            return {"success": False, "error": f"{type(exc).__name__}: {exc}"}


def _summary(values: list[float]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "min": round(ordered[0], 3),
        "median": round(statistics.median(ordered), 3),
        "max": round(ordered[-1], 3),
        "mean": round(statistics.mean(ordered), 3),
    }


def labels_path(snapshot_name: str) -> Path:
    """Where this snapshot's hand labels live: labels-<snapshot stem>.json,
    or the legacy labels.json if that is the set it holds."""
    if LEGACY_LABELS_PATH.exists():
        if json.loads(LEGACY_LABELS_PATH.read_text()).get("snapshot") == snapshot_name:
            return LEGACY_LABELS_PATH
    return SNAPSHOT_DIR / f"labels-{Path(snapshot_name).stem}.json"


def load_labels(snapshot_name: str) -> dict:
    """Hand labels by ticker for this snapshot, or {} if there are none yet."""
    path = labels_path(snapshot_name)
    return json.loads(path.read_text())["labels"] if path.exists() else {}


def _escalation_vs_labels(rows: list[dict], labels: dict, config: PolicyConfig) -> dict:
    """Escalation rate, plus recall/precision against hand-labeled material events."""
    escalated = {
        r["ticker"] for r in rows if decide(r["jev"], r["quote"].get("pct_change"), config).escalate
    }
    material = {t for t, lab in labels.items() if lab["material_event"]}
    caught = escalated & material
    return {
        "rate": round(len(escalated) / len(rows), 3) if rows else 0,
        "recall": round(len(caught) / len(material), 3) if material else None,
        "precision": round(len(caught) / len(escalated), 3) if escalated else None,
        "missed": sorted(material - escalated),
    }


def threshold_sweep(rows: list[dict], labels: dict) -> list[dict]:
    """Every combination in SWEEP_GRID, scored on the hand-labeled tickers."""
    rows = [r for r in rows if r["ticker"] in labels]
    results = []
    for material, needs, confidence in itertools.product(*SWEEP_GRID.values()):
        config = PolicyConfig(
            material_event_threshold=material,
            needs_analysis_threshold=needs,
            min_sentiment_confidence=confidence,
        )
        results.append(
            {
                "material_event_threshold": material,
                "needs_analysis_threshold": needs,
                "min_sentiment_confidence": confidence,
                **_escalation_vs_labels(rows, labels, config),
            }
        )
    return results


def label_metrics(rows: list[dict], labels: dict) -> dict:
    labeled = [r for r in rows if r["ticker"] in labels]
    if not labeled:
        return {}

    def accuracy(model: str) -> float | None:
        judged = [r for r in labeled if r[model].get("success")]
        if not judged:
            return None
        right = sum(1 for r in judged if r[model]["sentiment"] == labels[r["ticker"]]["sentiment"])
        return round(right / len(judged), 3)

    return {
        "labeled": len(labeled),
        "material_events": sum(1 for r in labeled if labels[r["ticker"]]["material_event"]),
        "jev_accuracy": accuracy("jev"),
        "gemini_accuracy": accuracy("gemini"),
        "escalation_vs_labels": {
            name: _escalation_vs_labels(labeled, labels, config) for name, config in POLICY_VARIANTS.items()
        },
        "sweep": threshold_sweep(rows, labels),
    }


def compute_metrics(rows: list[dict]) -> dict:
    both = [r for r in rows if r["jev"].get("success") and r["gemini"].get("success")]
    jev_ok = [r for r in rows if r["jev"].get("success")]
    gem_ok = [r for r in rows if r["gemini"].get("success")]

    confusion = {g: {j: 0 for j in LABELS} for g in LABELS}
    for r in both:
        if r["gemini"]["sentiment"] in LABELS:
            confusion[r["gemini"]["sentiment"]][r["jev"]["sentiment"]] += 1
    agree = sum(1 for r in both if r["gemini"]["sentiment"] == r["jev"]["sentiment"])

    escalation = {}
    for name, config in POLICY_VARIANTS.items():
        escalated = [
            r["ticker"] for r in rows if decide(r["jev"], r["quote"].get("pct_change"), config).escalate
        ]
        escalation[name] = {
            "escalated": len(escalated),
            "rate": round(len(escalated) / len(rows), 3) if rows else 0,
            "tickers": escalated,
        }

    jev_cost = [r["jev"]["usage"]["cost_usd"] for r in jev_ok]
    gem_cost = [r["gemini"]["cost_usd"] for r in gem_ok]
    return {
        "tickers": len(rows),
        "compared": len(both),
        "jev_failures": len(rows) - len(jev_ok),
        "gemini_failures": len(rows) - len(gem_ok),
        "agreement": round(agree / len(both), 3) if both else None,
        "confusion_gemini_rows_jev_cols": confusion,
        "jev_sentiment_confidence": _summary([r["jev"]["sentiment_confidence"] for r in jev_ok]),
        "jev_material_event": _summary([r["jev"]["material_event"] for r in jev_ok]),
        "jev_needs_analysis": _summary([r["jev"]["needs_analysis"] for r in jev_ok]),
        "jev_cost_per_ticker_usd": statistics.mean(jev_cost) if jev_cost else None,
        "gemini_cost_per_ticker_usd": statistics.mean(gem_cost) if gem_cost else None,
        "jev_latency_ms": _summary([r["jev"]["latency_ms"] for r in jev_ok]),
        "gemini_latency_ms": _summary([r["gemini"]["latency_ms"] for r in gem_ok]),
        "escalation_by_policy": escalation,
        "jev_models": sorted({r["jev"]["model"] for r in jev_ok}),
    }


def _pct(value: float | None) -> str:
    return f"{value:.0%}" if value is not None else "n/a"


def render_label_section(lm: dict) -> list[str]:
    if not lm:
        return ["", "## Hand labels", "", "No labels for this snapshot yet - run `python -m eval.label`."]

    lines = [
        "",
        f"## Against hand labels ({lm['labeled']} tickers, {lm['material_events']} material events)",
        "",
        "| Model | Sentiment accuracy |",
        "|---|---|",
        f"| Jev | {_pct(lm['jev_accuracy'])} |",
        f"| Gemini | {_pct(lm['gemini_accuracy'])} |",
        "",
        "Recall = share of hand-labeled material events that got escalated. "
        "Precision = share of escalations that were labeled material events.",
        "",
        "| Policy | Escalation rate | Recall | Precision | Missed events |",
        "|---|---|---|---|---|",
    ]
    for name, e in lm["escalation_vs_labels"].items():
        lines.append(
            f"| {name} | {_pct(e['rate'])} | {_pct(e['recall'])} | {_pct(e['precision'])} | {', '.join(e['missed']) or '-'} |"
        )

    sweep = lm["sweep"]
    best_recall = max((s["recall"] or 0) for s in sweep)
    target = min(0.9, best_recall)
    qualifying = sorted(
        (s for s in sweep if (s["recall"] or 0) >= target), key=lambda s: (s["rate"], -(s["recall"] or 0))
    )[:10]
    lines += [
        "",
        f"## Threshold sweep - lowest escalation rate with recall >= {target:.0%}",
        "",
        f"{len(sweep)} combinations tried; price-move threshold fixed at 4%. \"off\" = rule not applied.",
        "",
        "| material_event | needs_analysis | min confidence | Escalation rate | Recall | Precision |",
        "|---|---|---|---|---|---|",
    ]
    for s in qualifying:
        lines.append(
            f"| {s['material_event_threshold']} | {s['needs_analysis_threshold'] or 'off'} | "
            f"{s['min_sentiment_confidence'] or 'off'} | "
            f"{_pct(s['rate'])} | {_pct(s['recall'])} | {_pct(s['precision'])} |"
        )
    return lines


def render_markdown(snapshot_path: Path, rows: list[dict], m: dict) -> str:
    if m.get("gemini_skipped"):
        return _render_jev_only(snapshot_path, m)
    lines = [
        f"# Jev vs. Gemini sentiment - {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC",
        "",
        f"Snapshot: `{snapshot_path.name}` - {m['tickers']} tickers, {m['compared']} compared by both models "
        f"(Jev failures: {m['jev_failures']}, Gemini failures: {m['gemini_failures']}). Jev model: {', '.join(m['jev_models'])}.",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Sentiment agreement (Jev vs. Gemini) | {m['agreement']:.0%} |" if m["agreement"] is not None else "| Sentiment agreement | n/a |",
    ]
    if m["jev_cost_per_ticker_usd"] and m["gemini_cost_per_ticker_usd"]:
        ratio = m["gemini_cost_per_ticker_usd"] / m["jev_cost_per_ticker_usd"]
        lines += [
            f"| Cost per ticker - Jev (exact list price) | ${m['jev_cost_per_ticker_usd']:.7f} |",
            f"| Cost per ticker - Gemini sentiment (estimated) | ${m['gemini_cost_per_ticker_usd']:.7f} |",
            f"| Gemini / Jev cost ratio (sentiment step only) | {ratio:.1f}x |",
        ]
    lines += [
        f"| Median latency - Jev | {m['jev_latency_ms'].get('median')} ms |",
        f"| Median latency - Gemini | {m['gemini_latency_ms'].get('median')} ms |",
        "",
        "## Confusion matrix (rows = Gemini, columns = Jev)",
        "",
        "| Gemini \\ Jev | " + " | ".join(LABELS) + " |",
        "|---|" + "---|" * len(LABELS),
    ]
    for g in LABELS:
        lines.append(f"| {g} | " + " | ".join(str(m["confusion_gemini_rows_jev_cols"][g][j]) for j in LABELS) + " |")

    lines += [
        "",
        "## Jev probability distributions",
        "",
        "| Answer | min | median | mean | max |",
        "|---|---|---|---|---|",
    ]
    for key in ("jev_sentiment_confidence", "jev_material_event", "jev_needs_analysis"):
        s = m[key]
        lines.append(f"| {key.removeprefix('jev_')} | {s.get('min')} | {s.get('median')} | {s.get('mean')} | {s.get('max')} |")

    lines += ["", "## Escalation rate by policy variant", "", "| Policy | Escalated | Rate |", "|---|---|---|"]
    for name, e in m["escalation_by_policy"].items():
        lines.append(f"| {name} | {e['escalated']}/{m['tickers']} | {e['rate']:.0%} |")

    lines += render_label_section(m.get("labels") or {})

    lines += [
        "",
        "## Per ticker",
        "",
        "| Ticker | Move % | Headlines | Gemini | Jev (conf) | material | needs | Escalate (default) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        jev, gem = r["jev"], r["gemini"]
        decision = decide(jev, r["quote"].get("pct_change"))
        move = r["quote"].get("pct_change")
        lines.append(
            f"| {r['ticker']} | {move:+.2f} | {len(r['news'].get('headlines', []))} | "
            f"{gem.get('sentiment', 'ERR')} | "
            + (
                f"{jev['sentiment']} ({jev['sentiment_confidence']:.2f}) | {jev['material_event']:.2f} | {jev['needs_analysis']:.2f} | "
                if jev.get("success")
                else "ERR | - | - | "
            )
            + ("yes: " + "; ".join(decision.reasons) if decision.escalate else "no")
            + " |"
        )
    return "\n".join(lines) + "\n"


def _render_jev_only(snapshot_path: Path, m: dict) -> str:
    lines = [
        f"# Jev screening vs. hand labels - {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC",
        "",
        f"Snapshot: `{snapshot_path.name}` - {m['tickers']} tickers (Jev failures: {m['jev_failures']}). "
        f"Jev model: {', '.join(m['jev_models'])}. Gemini not run (--jev-only).",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Cost per ticker - Jev (exact list price) | ${m['jev_cost_per_ticker_usd'] or 0:.7f} |",
        f"| Median latency - Jev | {m['jev_latency_ms'].get('median')} ms |",
    ]
    lines += render_label_section(m["labels"])
    return "\n".join(line for line in lines if not line.startswith("| Gemini |")) + "\n"


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--snapshot", type=Path, help="Defaults to the newest eval/data/snapshot-*.json")
    parser.add_argument(
        "--reuse-gemini",
        action="store_true",
        help="Reuse Gemini answers from the newest earlier comparison of the same snapshot "
        "(re-runs Jev only - for iterating on Jev questions without Gemini quota)",
    )
    parser.add_argument(
        "--jev-only",
        action="store_true",
        help="Skip Gemini entirely: Jev accuracy and escalation vs. hand labels, no model comparison",
    )
    args = parser.parse_args()

    snapshot_path = args.snapshot or latest_snapshot()
    snapshot = json.loads(snapshot_path.read_text())
    items = [it for it in snapshot["items"] if it["quote"]["success"] and it["news"]["success"]]
    print(f"Comparing {len(items)} tickers from {snapshot_path.name}")

    cached_gemini = load_cached_gemini(snapshot_path.name) if args.reuse_gemini else {}

    rows = []
    for i, it in enumerate(items, 1):
        ticker = it["ticker"]
        jev = screen_ticker(ticker, it["quote"], it["news"])
        if args.jev_only:
            gemini = {"success": False, "error": "skipped (--jev-only)"}
        else:
            gemini = cached_gemini.get(ticker) or await run_gemini_sentiment(ticker, it["news"])
        rows.append({"ticker": ticker, "quote": it["quote"], "news": it["news"], "jev": jev, "gemini": gemini})
        print(
            f"[{i}/{len(items)}] {ticker}: Gemini={gemini.get('sentiment', 'ERR')} "
            f"Jev={jev.get('sentiment', 'ERR')} needs={jev.get('needs_analysis', '-')}"
        )

    metrics = compute_metrics(rows)
    metrics["gemini_skipped"] = args.jev_only
    metrics["labels"] = label_metrics(rows, load_labels(snapshot_path.name))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    (RESULTS_DIR / f"compare-{stamp}.json").write_text(
        json.dumps({"snapshot": snapshot_path.name, "metrics": metrics, "rows": rows}, indent=2)
    )
    md_path = RESULTS_DIR / f"compare-{stamp}.md"
    md_path.write_text(render_markdown(snapshot_path, rows, metrics))
    print(f"\nWrote {md_path.relative_to(ROOT)}")


if __name__ == "__main__":
    asyncio.run(main())
