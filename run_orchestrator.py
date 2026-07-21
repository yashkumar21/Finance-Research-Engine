"""Manual CLI harness: python run_orchestrator.py AAPL"""

import asyncio
import sys

from dotenv import load_dotenv

from agents.orchestrator import run_research_brief

load_dotenv()


async def main(ticker: str) -> None:
    brief = await run_research_brief(ticker)
    print(brief.brief_markdown)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_orchestrator.py <TICKER>")
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))
