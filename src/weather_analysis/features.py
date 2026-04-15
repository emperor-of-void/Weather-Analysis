from __future__ import annotations

import numpy as np
import pandas as pd


TARGET_COL = "meantemp"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = df.copy()

    feat["year"] = feat.index.year
    feat["month"] = feat.index.month
    feat["day"] = feat.index.day
    feat["dayofweek"] = feat.index.dayofweek
    feat["dayofyear"] = feat.index.dayofyear
    feat["is_weekend"] = (feat["dayofweek"] >= 5).astype(int)

    feat["sin_doy"] = np.sin(2 * np.pi * feat["dayofyear"] / 365.25)
    feat["cos_doy"] = np.cos(2 * np.pi * feat["dayofyear"] / 365.25)

    for lag in [1, 2, 3, 7, 14, 30]:
        feat[f"lag_{lag}"] = feat[TARGET_COL].shift(lag)

    feat["roll_mean_7"] = feat[TARGET_COL].rolling(window=7).mean()
    feat["roll_std_7"] = feat[TARGET_COL].rolling(window=7).std()
    feat["roll_mean_30"] = feat[TARGET_COL].rolling(window=30).mean()

    return feat.dropna().copy()


def train_test_split_time(df: pd.DataFrame, test_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(df) * (1 - test_ratio))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def get_feature_columns(df: pd.DataFrame, target_col: str = TARGET_COL) -> list[str]:
    return [c for c in df.columns if c != target_col]
