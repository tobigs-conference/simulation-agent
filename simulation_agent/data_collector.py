import logging
import os
from typing import Optional

from dotenv import load_dotenv

from simulation_agent import _external_deps

from interfaces import BaseRelationalDB
from storage.sqlite_db import SQLiteDB
from functions.get_agent_context import get_agent_context

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_B_DB_PATH = os.environ.get(
    "B_DB_PATH", "../financial-research-agent/db/reports.db"
)

def _mock_get_user_context(user_id: str) -> dict:
    logger.warning(
        "get_user_context()가 아직 실제 구현과 연결되지 않았습니다. "
        "목업 데이터를 반환합니다. (user_id=%s)", user_id
    )
    return {
        "risk_profile": "moderate",
        "investment_goal": "mid_term",
        "investment_amount_range": "500_2000",
        "investment_experience": "intermediate",
        "interest_sectors": [],
        "onboarding_done": True,
    }


# 실제 get_user_context가 준비되면 이 줄만 교체:
# from <어딘가> import get_user_context
get_user_context = _mock_get_user_context


def collect_simulation_inputs(
    ticker: str,
    user_id: str,
    relational_db: Optional[BaseRelationalDB] = None,
    db_path: str = DEFAULT_B_DB_PATH,
) -> dict:

    if relational_db is None:
        relational_db = SQLiteDB(db_path=db_path)

    agent_context = get_agent_context(
        ticker=ticker,
        agent_type="simulation",
        relational_db=relational_db,
    )

    if "error" in agent_context:
        raise ValueError(f"get_agent_context 실패: {agent_context['error']}")

    user_context = get_user_context(user_id)

    return {
        "ticker": ticker,
        "price_data": agent_context.get("price_data"),
        "macro_data": agent_context.get("macro_data"),
        "target_prices": agent_context.get("target_prices"),
        "user_context": user_context,
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Simulation Agent 데이터 수집 레이어 테스트")
    parser.add_argument("--ticker", default="005930")
    parser.add_argument("--user-id", default="u1")
    parser.add_argument("--db-path", default=DEFAULT_B_DB_PATH)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    result = collect_simulation_inputs(
        ticker=args.ticker, user_id=args.user_id, db_path=args.db_path
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
