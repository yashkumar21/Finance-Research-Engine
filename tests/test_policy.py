"""Tests for screener.policy.decide.

Pure unit tests - the policy makes no network or model calls, so every
threshold edge is exercised deterministically with hand-built screen
results.
"""

from screener.policy import PolicyConfig, decide


def _screen(material_event=0.1, needs_analysis=0.1, confidence=0.9, sentiment="neutral"):
    return {
        "success": True,
        "ticker": "TEST",
        "sentiment": sentiment,
        "sentiment_probabilities": {},
        "sentiment_confidence": confidence,
        "material_event": material_event,
        "needs_analysis": needs_analysis,
        "model": "typesafe-ai/jev",
        "latency_ms": 100.0,
        "usage": {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "cost_is_estimate": False},
    }


def test_quiet_ticker_is_not_escalated():
    decision = decide(_screen(), pct_change=0.5)
    assert decision.escalate is False
    assert decision.reasons == []


def test_material_event_at_threshold_escalates():
    decision = decide(_screen(material_event=0.6), pct_change=0.0)
    assert decision.escalate is True
    assert any("material event" in r for r in decision.reasons)


def test_material_event_just_below_threshold_does_not_escalate():
    assert decide(_screen(material_event=0.59), pct_change=0.0).escalate is False


def test_needs_analysis_escalates():
    decision = decide(_screen(needs_analysis=0.8), pct_change=0.0)
    assert decision.escalate is True
    assert any("analyst attention" in r for r in decision.reasons)


def test_large_price_move_escalates_in_either_direction():
    for move in (4.0, -4.0, 12.5, -9.1):
        decision = decide(_screen(), pct_change=move)
        assert decision.escalate is True, move
        assert any("price moved" in r for r in decision.reasons)


def test_small_price_move_does_not_escalate():
    assert decide(_screen(), pct_change=3.99).escalate is False
    assert decide(_screen(), pct_change=-3.99).escalate is False


def test_unknown_price_move_is_ignored():
    assert decide(_screen(), pct_change=None).escalate is False


def test_low_confidence_falls_back_to_llm():
    decision = decide(_screen(confidence=0.49), pct_change=0.0)
    assert decision.escalate is True
    assert any("unsure" in r for r in decision.reasons)


def test_screen_error_escalates_to_be_safe():
    error = {"success": False, "ticker": "TEST", "error": "Jev API error (500): boom"}
    decision = decide(error, pct_change=0.0)
    assert decision.escalate is True
    assert any("screening failed" in r and "boom" in r for r in decision.reasons)


def test_missing_screen_escalates_and_keeps_price_reason():
    decision = decide(None, pct_change=-6.0)
    assert decision.escalate is True
    assert len(decision.reasons) == 2


def test_multiple_triggers_are_all_reported():
    decision = decide(_screen(material_event=0.9, needs_analysis=0.9, confidence=0.3), pct_change=5.0)
    assert decision.escalate is True
    assert len(decision.reasons) == 4


def test_custom_config_changes_thresholds():
    strict = PolicyConfig(material_event_threshold=0.95)
    assert decide(_screen(material_event=0.9), pct_change=0.0, config=strict).escalate is False
