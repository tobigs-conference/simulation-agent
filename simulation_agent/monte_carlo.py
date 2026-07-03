import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

N_PATHS = 1000 
HORIZON = 30 

OPTIMISTIC_PERCENTILE = 90
PESSIMISTIC_PERCENTILE = 10 


def sample_paths(
    mu: float,
    sigma: float,
    current_price: float,
    horizon: int = HORIZON,
    n_paths: int = N_PATHS,
    seed: Optional[int] = None,
) -> np.ndarray:

    rng = np.random.default_rng(seed)

    daily_mu = mu / horizon
    daily_sigma = sigma / np.sqrt(horizon)

    daily_log_returns = rng.normal(
        loc=daily_mu,
        scale=daily_sigma,
        size=(n_paths, horizon),
    )

    cumulative = np.cumsum(daily_log_returns, axis=1)
    price_paths = current_price * np.exp(
        np.concatenate([np.zeros((n_paths, 1)), cumulative], axis=1)
    )

    return price_paths


def summarize_paths(
    price_paths: np.ndarray,
    current_price: float,
) -> dict:

    final_prices = price_paths[:, -1]
    final_returns = (final_prices - current_price) / current_price

    expected_return_pct = float(np.mean(final_returns) * 100)
    upside_probability = float(np.mean(final_returns > 0))
    volatility = float(np.std(final_returns))

    opt_mask = final_returns >= np.percentile(final_returns, OPTIMISTIC_PERCENTILE)
    pes_mask = final_returns <= np.percentile(final_returns, PESSIMISTIC_PERCENTILE)
    neu_mask = ~opt_mask & ~pes_mask

    def scenario_summary(mask: np.ndarray) -> dict:
        paths = price_paths[mask]
        returns = final_returns[mask]
        return {
            "expected_return_pct": float(np.mean(returns) * 100),
            "avg_final_price": float(np.mean(paths[:, -1])),
            "path_count": int(mask.sum()),
        }

    p10_idx = np.argsort(final_returns)[int(N_PATHS * 0.10)]
    p50_idx = np.argsort(final_returns)[int(N_PATHS * 0.50)]
    p90_idx = np.argsort(final_returns)[int(N_PATHS * 0.90)]

    return {
        "expected_return_pct": expected_return_pct,
        "upside_probability": upside_probability,
        "volatility": volatility,
        "scenarios": {
            "optimistic": scenario_summary(opt_mask),
            "neutral": scenario_summary(neu_mask),
            "pessimistic": scenario_summary(pes_mask),
        },
        "percentile_paths": {
            "p10": price_paths[p10_idx].tolist(),
            "p50": price_paths[p50_idx].tolist(),
            "p90": price_paths[p90_idx].tolist(),
        },
    }


def run_monte_carlo(
    base_prediction: dict,
    current_price: float,
    shock_predictions: Optional[list] = None,
    horizon: int = HORIZON,
    n_paths: int = N_PATHS,
) -> dict:

    base_paths = sample_paths(
        mu=base_prediction["mu"],
        sigma=base_prediction["sigma"],
        current_price=current_price,
        horizon=horizon,
        n_paths=n_paths,
        seed=42,
    )
    base_summary = summarize_paths(base_paths, current_price)

    result = {"base": base_summary, "what_if": []}

    if shock_predictions:
        for shock in shock_predictions:
            shock_paths = sample_paths(
                mu=shock["prediction"]["mu"],
                sigma=shock["prediction"]["sigma"],
                current_price=current_price,
                horizon=horizon,
                n_paths=n_paths,
                seed=42,
            )
            shock_summary = summarize_paths(shock_paths, current_price)
            impact = (
                shock_summary["expected_return_pct"]
                - base_summary["expected_return_pct"]
            )

            result["what_if"].append({
                "variable": shock["variable"],
                "direction": shock["direction"],
                "result": shock_summary,
                "impact_pct": round(impact, 2),
            })

            logger.info(
                "What-if [%s %s]: 기대수익률 변화 %+.2f%%",
                shock["variable"], shock["direction"], impact,
            )

    return result


if __name__ == "__main__":
    import argparse
    import json

    from simulation_agent.data_collector import collect_simulation_inputs, DEFAULT_B_DB_PATH
    from simulation_agent.preprocessor import build_feature_table
    from simulation_agent.model import get_or_train_model, predict_distribution, apply_shock
    from simulation_agent.risk_classifier import classify_risk_factors

    parser = argparse.ArgumentParser(description="Simulation Agent Monte Carlo 테스트")
    parser.add_argument("--ticker", default="005930")
    parser.add_argument("--user-id", default="u1")
    parser.add_argument("--db-path", default=DEFAULT_B_DB_PATH)
    parser.add_argument("--sequence-length", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    raw = collect_simulation_inputs(
        ticker=args.ticker, user_id=args.user_id, db_path=args.db_path
    )

    table = build_feature_table(raw["price_data"], raw["macro_data"])
    current_price = float(raw["price_data"]["latest"]["current_price"])

    macro_agenda = {
        "bull_summary": "환율 상승이 수출에 긍정적",
        "bull_arguments": "원달러 환율 상승으로 수출 채산성 개선",
        "bear_summary": "금리 인상 우려",
        "bear_arguments": "기준금리 추가 인상 시 투자심리 위축 우려",
    }
    risk_factors = classify_risk_factors(macro_agenda=macro_agenda)
    logger.info("분류된 리스크 요인: %s", risk_factors)

    model = get_or_train_model(
        ticker=args.ticker,
        feature_table=table,
        sequence_length=args.sequence_length,
        horizon=args.horizon,
        epochs=args.epochs,
        force_retrain=True,
    )
    base_pred = predict_distribution(model, table, sequence_length=args.sequence_length)
    logger.info("일반 예측: %s", base_pred)

    shock_preds = []
    for rf in risk_factors:
        shocked_table = apply_shock(table, rf["variable"], rf["direction"])
        shock_pred = predict_distribution(model, shocked_table, args.sequence_length)
        shock_preds.append({
            "variable": rf["variable"],
            "direction": rf["direction"],
            "prediction": shock_pred,
        })

    mc_result = run_monte_carlo(
        base_prediction=base_pred,
        current_price=current_price,
        shock_predictions=shock_preds if shock_preds else None,
        horizon=args.horizon,
        n_paths=N_PATHS,
    )

    output = {
        "base": {k: v for k, v in mc_result["base"].items() if k != "percentile_paths"},
        "what_if": [
            {k: v for k, v in w.items() if k != "result" or True}
            for w in mc_result["what_if"]
        ],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
