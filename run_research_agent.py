"""Manual CLI harness: python run_research_agent.py AAPL"""

import asyncio
import sys

from dotenv import load_dotenv
from google.genai import types

from google.adk.runners import InMemoryRunner

from agents.research_agent import research_agent

load_dotenv()

APP_NAME = "finance_research_engine_cli"
USER_ID = "cli_user"


async def main(ticker: str) -> None:
    runner = InMemoryRunner(agent=research_agent, app_name=APP_NAME)
    session = await runner.session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
    message = types.Content(role="user", parts=[types.Part(text=f"Research {ticker}")])

    async for event in runner.run_async(user_id=USER_ID, session_id=session.id, new_message=message):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(part.text)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_research_agent.py <TICKER>")
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))
