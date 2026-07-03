import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MACRO_INDICATORS = ["BASE_RATE_KR", "CPI_KR", "KTB_3Y_KR", "KTB_10Y_KR", "USD_KRW"]


class InsufficientDataError(Exception):
    pass


def _price_data_to_df(price_data: dict) -> pd.DataFrame:
    prices = price_data.get("prices") or []
    if not prices:
        raise InsufficientDataError("price_data가 비어 있습니다 (해당 ticker의 가격 데이터 없음).")

    df = pd.DataFrame(prices)
    df["price_date"] = pd.to_datetime(df["price_date"])
    df = df.sort_values("price_date").reset_index(drop=True)

    for col in ["close", "volume", "volatility_30d"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    if "volatility_30d" not in df.columns or df["volatility_30d"].isna().all():
        df["volatility_30d"] = df["log_return"].rolling(window=30, min_periods=5).std()

    return df[["price_date", "close", "volume", "volatility_30d", "log_return"]]


def _macro_data_to_df(macro_data: dict) -> pd.DataFrame:
    indicators = macro_data.get("indicators") or []

    if not indicators:
        logger.warning("macro_data가 비어 있습니다. 매크로 피처는 전부 NaN으로 채워집니다.")
        return pd.DataFrame(columns=["date"] + MACRO_INDICATORS)

    frames = []
    for ind in indicators:
        ind_id = ind.get("indicator_id")
        if ind_id not in MACRO_INDICATORS:
            continue
        records = ind.get("records") or []
        if not records:
            continue
        s = pd.DataFrame(records)[["date", "value"]].rename(columns={"value": ind_id})
        s["date"] = pd.to_datetime(s["date"])
        frames.append(s.set_index("date"))

    if not frames:
        return pd.DataFrame(columns=["date"] + MACRO_INDICATORS)

    wide = pd.concat(frames, axis=1).sort_index()
    wide = wide.reset_index().rename(columns={"index": "date"})
    return wide


def build_feature_table(
    price_data: dict,
    macro_data: dict,
    min_rows: int = 20,
) -> pd.DataFrame:

    price_df = _price_data_to_df(price_data)
    macro_df = _macro_data_to_df(macro_data)

    merged = price_df.rename(columns={"price_date": "date"})

    if not macro_df.empty:
        merged = pd.merge_asof(
            merged.sort_values("date"),
            macro_df.sort_values("date"),
            on="date",
            direction="backward",
        )
    else:
        for col in MACRO_INDICATORS:
            merged[col] = np.nan

    merged[MACRO_INDICATORS] = merged[MACRO_INDICATORS].ffill().fillna(0)

    merged = merged.dropna(subset=["log_return"]).reset_index(drop=True)

    if len(merged) < min_rows:
        raise InsufficientDataError(
            f"전처리 후 데이터가 {len(merged)}행밖에 없습니다 (최소 {min_rows}행 필요). "
            f"해당 ticker의 가격 데이터가 더 필요합니다."
        )

    return merged


def to_lstm_input(
    feature_table: pd.DataFrame,
    sequence_length: int = 60,
    feature_cols: Optional[list] = None,
) -> np.ndarray:

    if feature_cols is None:
        feature_cols = ["log_return", "volume", "volatility_30d"] + MACRO_INDICATORS

    if len(feature_table) < sequence_length:
        raise InsufficientDataError(
            f"시퀀스 길이({sequence_length}) 확보를 위한 데이터가 부족합니다 "
            f"(현재 {len(feature_table)}행)."
        )

    window = feature_table.tail(sequence_length)
    arr = window[feature_cols].to_numpy(dtype=np.float32)

    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    arr = (arr - mean) / std

    return arr


if __name__ == "__main__":
    import argparse
    import logging as _logging

    from simulation_agent.data_collector import collect_simulation_inputs, DEFAULT_B_DB_PATH

    parser = argparse.ArgumentParser(description="Simulation Agent 전처리 레이어 테스트")
    parser.add_argument("--ticker", default="005930")
    parser.add_argument("--user-id", default="u1")
    parser.add_argument("--db-path", default=DEFAULT_B_DB_PATH)
    parser.add_argument("--sequence-length", type=int, default=60)
    args = parser.parse_args()

    _logging.basicConfig(level=_logging.INFO)

    raw = collect_simulation_inputs(
        ticker=args.ticker, user_id=args.user_id, db_path=args.db_path
    )
    table = build_feature_table(raw["price_data"], raw["macro_data"])
    print("피처 테이블 shape:", table.shape)
    print(table.tail())

    lstm_input = to_lstm_input(table, sequence_length=args.sequence_length)
    print("LSTM 입력 shape:", lstm_input.shape)