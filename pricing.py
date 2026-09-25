"""List prices used to estimate Gemini cost.

Jev's cost is exact - Vercel AI Gateway reports it per call (see
tools/jev_screen.py) - so only Gemini needs estimating: token counts from
ADK's usage_metadata x these list prices. Paid-tier standard prices from
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
