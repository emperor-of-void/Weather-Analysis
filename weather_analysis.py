from __future__ import annotations

import argparse
from pathlib import Path

from src.weather_analysis.config import PipelineConfig
from src.weather_analysis.pipeline import run_pipeline


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(description="Weather-Analysis CLI")
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON config file")
    parser.add_argument("--data", type=Path, default=Path("DailyDelhiClimateTrain.csv"), help="Path to CSV data")
    parser.add_argument("--out", type=Path, default=Path("outputs"), help="Output directory")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--model", type=str, default="auto", choices=["auto", "rf", "gbr", "xgb"], help="Model choice")
    parser.add_argument("--n-estimators", type=int, default=400, help="Tree count for random forest")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    if args.config:
        cfg = PipelineConfig.from_json(args.config)
    else:
        cfg = PipelineConfig(
            data_path=args.data,
            output_dir=args.out,
            test_ratio=args.test_ratio,
            random_state=args.random_state,
            n_estimators=args.n_estimators,
            model=args.model,
            save_plots=not args.no_plots,
        )

    cfg.validate()
    return cfg


def main() -> None:
    cfg = parse_args()
    result = run_pipeline(cfg)

    metrics = result["metrics"]
    print("Da hoan thanh Weather-Analysis")
    print(f"Mo hinh tot nhat: {result['model_name']}")
    print(f"MAE={metrics['MAE']:.4f} | RMSE={metrics['RMSE']:.4f} | R2={metrics['R2']:.4f}")
    print(f"Bao cao: {result['report_path']}")


if __name__ == "__main__":
    main()
