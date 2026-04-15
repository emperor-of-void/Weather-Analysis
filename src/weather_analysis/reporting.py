from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
# Use a headless backend for stable CLI/CI execution on environments without GUI mainloop.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save_plots(
    raw_df: pd.DataFrame,
    test_df: pd.DataFrame,
    predictions,
    model,
    feature_cols: list[str],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(14, 5))
    plt.plot(raw_df.index, raw_df["meantemp"], color="#d04a35", linewidth=1.2)
    plt.title("Xu huong nhiet do trung binh tai Delhi")
    plt.xlabel("Ngay")
    plt.ylabel("Nhiet do (C)")
    plt.tight_layout()
    plt.savefig(output_dir / "01_temperature_trend.png", dpi=150)
    plt.close()

    corr_cols = [c for c in ["meantemp", "humidity", "wind_speed", "meanpressure"] if c in raw_df.columns]
    if len(corr_cols) >= 2:
        plt.figure(figsize=(6, 5))
        sns.heatmap(raw_df[corr_cols].corr(), annot=True, cmap="YlGnBu", fmt=".2f")
        plt.title("Tuong quan giua cac bien thoi tiet")
        plt.tight_layout()
        plt.savefig(output_dir / "02_correlation_heatmap.png", dpi=150)
        plt.close()

    plt.figure(figsize=(14, 5))
    plt.plot(test_df.index, test_df["meantemp"], label="Thuc te", color="#205072")
    plt.plot(test_df.index, predictions, label="Du bao", color="#f18f01", linestyle="--")
    plt.title("So sanh du bao va thuc te")
    plt.xlabel("Ngay")
    plt.ylabel("Nhiet do (C)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "03_forecast_vs_actual.png", dpi=150)
    plt.close()

    importances = getattr(model, "feature_importances_", None)
    if importances is not None:
        imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False).head(12)
        plt.figure(figsize=(9, 5))
        sns.barplot(x=imp.values, y=imp.index, hue=imp.index, palette="crest", legend=False)
        plt.title("Top 12 dac trung quan trong nhat")
        plt.xlabel("Muc do quan trong")
        plt.ylabel("Dac trung")
        plt.tight_layout()
        plt.savefig(output_dir / "04_feature_importance.png", dpi=150)
        plt.close()
    else:
        logging.info("Selected model does not expose feature_importances_")


def save_report(raw_df: pd.DataFrame, metrics: dict[str, float], model_name: str, output_dir: Path) -> Path:
    hottest_day = raw_df["meantemp"].idxmax()
    coldest_day = raw_df["meantemp"].idxmin()

    report_lines = [
        "# Bao Cao Weather-Analysis",
        "",
        "## Du Lieu",
        f"- So dong: {len(raw_df)}",
        f"- Khoang thoi gian: {raw_df.index.min().date()} -> {raw_df.index.max().date()}",
        "",
        "## Thong Ke Quan Trong",
        f"- Nhiet do trung binh: {raw_df['meantemp'].mean():.2f} C",
        f"- Nhiet do cao nhat: {raw_df['meantemp'].max():.2f} C vao ngay {hottest_day.date()}",
        f"- Nhiet do thap nhat: {raw_df['meantemp'].min():.2f} C vao ngay {coldest_day.date()}",
        "",
        f"## Chi So Mo Hinh ({model_name})",
        f"- MAE: {metrics['MAE']:.4f}",
        f"- RMSE: {metrics['RMSE']:.4f}",
        f"- R2: {metrics['R2']:.4f}",
    ]

    out_file = output_dir / "report.md"
    out_file.write_text("\n".join(report_lines), encoding="utf-8")
    return out_file
