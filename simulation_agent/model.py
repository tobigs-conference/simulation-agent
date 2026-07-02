import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time

from simulation_agent.preprocessor import MACRO_INDICATORS, to_lstm_input

logger = logging.getLogger(__name__)

FEATURE_COLS = ["log_return", "volume", "volatility_30d"] + MACRO_INDICATORS
N_FEATURES = len(FEATURE_COLS)

MODEL_DIR = Path(__file__).parent / "_model_cache"
MODEL_DIR.mkdir(exist_ok=True)


class ReturnDistributionLSTM(nn.Module):

    def __init__(self, n_features: int = N_FEATURES, hidden_size: int = 32, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 2)  # [mu, log_sigma]

    def forward(self, x: torch.Tensor) -> tuple:
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]  # (batch, hidden_size)
        out = self.head(last_hidden)  # (batch, 2)
        mu = out[:, 0]
        log_sigma = out[:, 1]
        sigma = torch.nn.functional.softplus(log_sigma) + 1e-4  # 항상 양수 보장
        return mu, sigma


def _gaussian_nll_loss(mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (
        torch.log(sigma)
        + 0.5 * ((target - mu) / sigma) ** 2
    ).mean()


def build_training_windows(
    feature_table: pd.DataFrame,
    sequence_length: int = 60,
    horizon: int = 30,
) -> tuple:

    values = feature_table[FEATURE_COLS].to_numpy(dtype=np.float32)
    log_returns = feature_table["log_return"].to_numpy(dtype=np.float32)

    n_total = len(feature_table)
    n_samples = n_total - sequence_length - horizon
    if n_samples <= 0:
        raise ValueError(
            f"학습 윈도우를 만들기엔 데이터가 부족합니다. "
            f"필요: sequence_length({sequence_length}) + horizon({horizon}) + 1행 이상, "
            f"보유: {n_total}행"
        )

    X = np.zeros((n_samples, sequence_length, len(FEATURE_COLS)), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.float32)

    for i in range(n_samples):
        window = values[i: i + sequence_length]
        mean = window.mean(axis=0, keepdims=True)
        std = window.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        X[i] = (window - mean) / std
        y[i] = log_returns[i + sequence_length: i + sequence_length + horizon].sum()

    return X, y


def train_model(
    feature_table: pd.DataFrame,
    sequence_length: int = 60,
    horizon: int = 30,
    epochs: int = 50,
    lr: float = 1e-3,
) -> ReturnDistributionLSTM:
    X, y = build_training_windows(feature_table, sequence_length, horizon)

    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)

    model = ReturnDistributionLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        mu, sigma = model(X_t)
        loss = _gaussian_nll_loss(mu, sigma, y_t)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info("epoch %d/%d - loss: %.4f", epoch + 1, epochs, loss.item())

    return model


def get_or_train_model(
    ticker: str,
    feature_table: pd.DataFrame,
    sequence_length: int = 60,
    horizon: int = 30,
    epochs: int = 50,
    force_retrain: bool = False,
) -> ReturnDistributionLSTM:
    cache_path = MODEL_DIR / f"{ticker}.pt"

    cache_age = (time.time() - cache_path.stat().st_mtime) if cache_path.exists() else float("inf")

    if cache_path.exists() and cache_age < 86400 and not force_retrain:
        logger.info("캐시된 모델 로드 (%.1f시간 경과): %s", cache_age / 3600, cache_path)
        model = ReturnDistributionLSTM()
        model.load_state_dict(torch.load(cache_path, weights_only=True))
        model.eval()
        return model

    logger.info("ticker=%s 모델 학습 시작 (캐시 없음 또는 강제 재학습)", ticker)
    model = train_model(feature_table, sequence_length, horizon, epochs)
    torch.save(model.state_dict(), cache_path)
    model.eval()
    return model


def predict_distribution(
    model: ReturnDistributionLSTM,
    feature_table: pd.DataFrame,
    sequence_length: int = 60,
) -> dict:

    x = to_lstm_input(feature_table, sequence_length=sequence_length, feature_cols=FEATURE_COLS)
    x_t = torch.from_numpy(x).unsqueeze(0)  # (1, seq_len, n_features)

    model.eval()
    with torch.no_grad():
        mu, sigma = model(x_t)

    return {"mu": float(mu.item()), "sigma": float(sigma.item())}


def apply_shock(
    feature_table: pd.DataFrame,
    variable: str,
    direction: str,
    magnitude_pct: float = 1.0,
) -> pd.DataFrame:

    if variable not in MACRO_INDICATORS:
        raise ValueError(f"지원하지 않는 변수입니다: {variable} (허용: {MACRO_INDICATORS})")

    shocked = feature_table.copy()
    delta = magnitude_pct if direction == "up" else -magnitude_pct
    shocked[variable] = shocked[variable] + delta
    return shocked


if __name__ == "__main__":
    import argparse
    import json

    from simulation_agent.data_collector import collect_simulation_inputs, DEFAULT_B_DB_PATH
    from simulation_agent.preprocessor import build_feature_table

    parser = argparse.ArgumentParser(description="Simulation Agent LSTM 모델 레이어 테스트")
    parser.add_argument("--ticker", default="005930")
    parser.add_argument("--user-id", default="u1")
    parser.add_argument("--db-path", default=DEFAULT_B_DB_PATH)
    parser.add_argument("--sequence-length", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    raw = collect_simulation_inputs(ticker=args.ticker, user_id=args.user_id, db_path=args.db_path)
    table = build_feature_table(raw["price_data"], raw["macro_data"])

    model = get_or_train_model(
        ticker=args.ticker,
        feature_table=table,
        sequence_length=args.sequence_length,
        horizon=args.horizon,
        epochs=args.epochs,
        force_retrain=True,
    )

    base_pred = predict_distribution(model, table, sequence_length=args.sequence_length)
    print("일반 예측:", json.dumps(base_pred, indent=2))

    shocked_table = apply_shock(table, variable="BASE_RATE_KR", direction="up", magnitude_pct=1.0)
    shock_pred = predict_distribution(model, shocked_table, sequence_length=args.sequence_length)
    print("금리 +1%p 충격 후 예측:", json.dumps(shock_pred, indent=2))
