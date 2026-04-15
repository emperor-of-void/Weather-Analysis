from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"date", "meantemp"}


def load_and_clean_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if not REQUIRED_COLUMNS.issubset(df.columns):
        raise ValueError("CSV must contain at least 'date' and 'meantemp' columns")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df = df.sort_values("date").drop_duplicates(subset=["date"]).set_index("date")

    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method="time").ffill().bfill()
    return df
