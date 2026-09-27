"""List prices used to estimate model cost from token counts.

Gemini's cost is always estimated: token counts from ADK's usage_metadata x
these list prices. Jev's is used as-is when the provider reports it per call
(see tools/jev_screen.py), otherwise estimated from its list price here. Paid-tier standard prices from
https://ai.google.dev/gemini-api/docs/pricing as of 2026-09-25; re-check
before quoting cost numbers, prices change.
"""

# gemini-3.5-flash-lite, USD per 1M tokens. Output includes thinking tokens.
GEMINI_FLASH_LITE_INPUT_PER_M = 0.30
GEMINI_FLASH_LITE_OUTPUT_PER_M = 2.50


def gemini_cost(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens * GEMINI_FLASH_LITE_INPUT_PER_M
        + output_tokens * GEMINI_FLASH_LITE_OUTPUT_PER_M
    ) / 1_000_000


# typesafe/jev-1.13.0 on Requesty, USD per 1M input tokens; output is free.
# https://www.requesty.ai/model/typesafe/jev as of 2026-09-27.
JEV_INPUT_PER_M = 0.04


def jev_cost(input_tokens: int) -> float:
    return input_tokens * JEV_INPUT_PER_M / 1_000_000
