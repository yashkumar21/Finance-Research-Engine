"""Escalation policy: decides which screened tickers get a full research brief.

Pure and deterministic - no network, no model calls - so thresholds can be
unit-tested and swept by the eval harness. Errs toward escalating: a ticker
is only skipped when Jev is confident nothing needs a closer look.
"""

from dataclasses import dataclass
from typing import Optional

from schemas import Decision


@dataclass(frozen=True)
class PolicyConfig:
    material_event_threshold: float = 0.6
    needs_analysis_threshold: float = 0.6
    price_move_threshold_pct: float = 4.0
    min_sentiment_confidence: float = 0.5


DEFAULT_POLICY = PolicyConfig()


def decide(
    screen: Optional[dict],
    pct_change: Optional[float],
    config: PolicyConfig = DEFAULT_POLICY,
) -> Decision:
    """Decide whether to escalate a ticker to the full Gemini pipeline.

    Args:
        screen: A screen_ticker result (success or error dict), or None if
            screening never ran.
        pct_change: Today's price move in percent, or None if unknown.
        config: Thresholds.

    Returns:
        A Decision with every reason that triggered escalation (empty when
        not escalated).
    """
    reasons = []

    if pct_change is not None and abs(pct_change) >= config.price_move_threshold_pct:
        reasons.append(f"price moved {pct_change:+.2f}% today")

    if not screen or not screen.get("success"):
        error = (screen or {}).get("error", "screening did not run")
        reasons.append(f"screening failed ({error}) - escalating to be safe")
        return Decision(escalate=True, reasons=reasons)

    if screen["material_event"] >= config.material_event_threshold:
        reasons.append(f"material event likely (p={screen['material_event']:.2f})")
    if screen["needs_analysis"] >= config.needs_analysis_threshold:
        reasons.append(f"analyst attention likely needed (p={screen['needs_analysis']:.2f})")
    if screen["sentiment_confidence"] < config.min_sentiment_confidence:
        reasons.append(
            f"Jev unsure of sentiment (confidence={screen['sentiment_confidence']:.2f})"
        )

    return Decision(escalate=bool(reasons), reasons=reasons)
