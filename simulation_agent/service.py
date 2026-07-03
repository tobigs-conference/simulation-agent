import logging
import os
from typing import Optional

from dotenv import load_dotenv

from simulation_agent.data_collector import collect_simulation_inputs, DEFAULT_B_DB_PATH
from simulation_agent.preprocessor import build_feature_table, InsufficientDataError
from simulation_agent.risk_classifier import classify_risk_factors
from simulation_agent.model import get_or_train_model, predict_distribution, apply_shock
from simulation_agent.monte_carlo import run_monte_carlo, N_PATHS, HORIZON
from simulation_agent.result_packager import package_result

load_dotenv()
logger = logging.getLogger(__name__)


async def run_simulation(
    ticker: str,
    user_id: str,
    agenda_2: dict,
    db_path: str = DEFAULT_B_DB_PATH,
    send_result_fn=None,
) -> None:

    logger.info("[G] run_simulation 시작 - ticker=%s, user_id=%s", ticker, user_id)

    try:
        logger.info("[G] 1단계: 데이터 수집")
        raw = collect_simulation_inputs(
            ticker=ticker, user_id=user_id, db_path=db_path
        )
        latest = raw["price_data"].get("latest")
        if latest is None or latest.get("current_price") is None:
            raise ValueError(f"현재가를 가져올 수 없습니다. (ticker={ticker}). 주가 데이터가 없거나 비어있습니다.")
        current_price = float(latest["current_price"])

        logger.info("[G] 2단계: 전처리")
        table = build_feature_table(raw["price_data"], raw["macro_data"])

        logger.info("[G] 3단계: 리스크 분류")
        risk_factors = classify_risk_factors(macro_agenda=agenda_2)
        logger.info("[G] 분류된 리스크 요인: %s", risk_factors)

        logger.info("[G] 4단계: LSTM 예측")
        model = get_or_train_model(
            ticker=ticker,
            feature_table=table,
            sequence_length=60,
            horizon=30,
        )
        base_pred = predict_distribution(model, table)

        shock_preds = []
        for rf in risk_factors:
            shocked = apply_shock(table, rf["variable"], rf["direction"])
            shock_pred = predict_distribution(model, shocked)
            shock_preds.append({
                "variable": rf["variable"],
                "direction": rf["direction"],
                "prediction": shock_pred,
            })

        logger.info("[G] 5단계: Monte Carlo 샘플링")
        mc_result = run_monte_carlo(
            base_prediction=base_pred,
            current_price=current_price,
            shock_predictions=shock_preds if shock_preds else None,
        )

        logger.info("[G] 6단계: 결과 패키징")
        final_result = package_result(
            ticker=ticker,
            mc_result=mc_result,
            user_context=raw["user_context"],
            risk_factors=risk_factors,
            current_price=current_price,
        )

        logger.info("[G] 7단계: 프론트 전송")
        if send_result_fn:
            await send_result_fn(final_result)
        else:
            import json
            logger.info("[G] 전송 결과:\n%s", json.dumps(
                {k: v for k, v in final_result.items() if k != "chart_data"},
                ensure_ascii=False, indent=2
            ))

        logger.info("[G] run_simulation 완료 - ticker=%s", ticker)

    except InsufficientDataError as e:
        logger.error("[G] 데이터 부족으로 시뮬레이션 불가: %s", e)
        if send_result_fn:
            await send_result_fn({
                "ticker": ticker,
                "error": "데이터 부족",
                "message": str(e),
            })

    except Exception as e:
        logger.error("[G] 시뮬레이션 실패: %s", e, exc_info=True)
        if send_result_fn:
            await send_result_fn({
                "ticker": ticker,
                "error": "시뮬레이션 실패",
                "message": str(e),
            })


if __name__ == "__main__":
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="Agent G 전체 파이프라인 테스트")
    parser.add_argument("--ticker", default="005930")
    parser.add_argument("--user-id", default="u1")
    parser.add_argument("--db-path", default=DEFAULT_B_DB_PATH)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    agenda_2 = {
        "bull_summary": "환율 상승이 수출에 긍정적",
        "bull_arguments": "원달러 환율 상승으로 수출 채산성 개선",
        "bear_summary": "금리 인상 우려",
        "bear_arguments": "기준금리 추가 인상 시 투자심리 위축 우려",
    }

    asyncio.run(run_simulation(
        ticker=args.ticker,
        user_id=args.user_id,
        agenda_2=agenda_2,
        db_path=args.db_path,
    ))
