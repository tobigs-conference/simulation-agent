import logging
import argparse
import json
import numpy as np
import torch

logger = logging.getLogger(__name__)


def evaluate_model(
    model,
    feature_table,
    sequence_length: int = 60,
    horizon: int = 30,
    test_ratio: float = 0.2,
) -> dict:

    from simulation_agent.model import FEATURE_COLS, build_training_windows

    X, y = build_training_windows(feature_table, sequence_length, horizon)

    n_total = len(X)
    n_test = max(1, int(n_total * test_ratio))
    n_train = n_total - n_test

    if n_train <= 0:
        raise ValueError(f"데이터가 너무 적어 평가 불가. 전체 샘플: {n_total}")

    X_test = torch.from_numpy(X[n_train:])
    y_test = y[n_train:]

    model.eval()
    with torch.no_grad():
        mu, sigma = model(X_test)

    mu_np = mu.numpy()
    sigma_np = sigma.numpy()

    # MAE
    mae = float(np.mean(np.abs(mu_np - y_test)))

    # RMSE
    rmse = float(np.sqrt(np.mean((mu_np - y_test) ** 2)))

    # 방향 정확도 (예측 mu > 0 이면 상승, 실제 y > 0 이면 상승)
    pred_direction = mu_np > 0
    actual_direction = y_test > 0
    direction_accuracy = float(np.mean(pred_direction == actual_direction))

    # 커버리지 (실제값이 mu ± 1*sigma 구간 안에 드는 비율)
    lower = mu_np - sigma_np
    upper = mu_np + sigma_np
    coverage = float(np.mean((y_test >= lower) & (y_test <= upper)))

    result = {
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "direction_accuracy": round(direction_accuracy * 100, 2),
        "coverage": round(coverage * 100, 2),
        "n_test_samples": n_test,
        "n_train_samples": n_train,
    }

    logger.info("=== 모델 평가 결과 ===")
    logger.info("MAE:              %.6f", result["mae"])
    logger.info("RMSE:             %.6f", result["rmse"])
    logger.info("방향 정확도:      %.2f%%", result["direction_accuracy"])
    logger.info("커버리지(±1σ):   %.2f%%", result["coverage"])
    logger.info("학습 샘플 수:     %d", result["n_train_samples"])
    logger.info("검증 샘플 수:     %d", result["n_test_samples"])

    return result


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from simulation_agent import _external_deps
    from simulation_agent.data_collector import collect_simulation_inputs, DEFAULT_B_DB_PATH
    from simulation_agent.preprocessor import build_feature_table
    from simulation_agent.model import get_or_train_model

    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Agent G 모델 평가")
    parser.add_argument("--ticker", default="005930")
    parser.add_argument("--user-id", default="u1")
    parser.add_argument("--db-path", default=DEFAULT_B_DB_PATH)
    parser.add_argument("--sequence-length", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    args = parser.parse_args()

    raw = collect_simulation_inputs(
        ticker=args.ticker, user_id=args.user_id, db_path=args.db_path
    )
    table = build_feature_table(raw["price_data"], raw["macro_data"])

    model = get_or_train_model(
        ticker=args.ticker,
        feature_table=table,
        sequence_length=args.sequence_length,
        horizon=args.horizon,
    )

    result = evaluate_model(
        model=model,
        feature_table=table,
        sequence_length=args.sequence_length,
        horizon=args.horizon,
        test_ratio=args.test_ratio,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))
