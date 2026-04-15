from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .config import PipelineConfig
from .data import load_and_clean_data
from .features import build_features, get_feature_columns, train_test_split_time
from .modeling import ModelResult, choose_and_train
from .reporting import save_plots, save_report


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def run_pipeline(config: PipelineConfig) -> dict[str, object]:
    setup_logging()
    config.validate()

    logging.info("Loading data from %s", config.data_path)
    raw_df = load_and_clean_data(config.data_path)

    logging.info("Building time-series features")
    feat_df = build_features(raw_df)
    train_df, test_df = train_test_split_time(feat_df, config.test_ratio)
    feature_cols = get_feature_columns(feat_df)

    X_train = train_df[feature_cols]
    y_train = train_df["meantemp"]
    X_test = test_df[feature_cols]
    y_test = test_df["meantemp"]

    logging.info("Training model with choice=%s", config.model)
    result: ModelResult = choose_and_train(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model_choice=config.model,
        n_estimators=config.n_estimators,
        random_state=config.random_state,
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.save_plots:
        save_plots(
            raw_df=raw_df,
            test_df=test_df,
            predictions=result.predictions,
            model=result.model,
            feature_cols=feature_cols,
            output_dir=output_dir,
        )

    report_path = save_report(raw_df, result.metrics, result.model_name, output_dir)

    logging.info("Best model: %s", result.model_name)
    logging.info("MAE=%.4f RMSE=%.4f R2=%.4f", result.metrics["MAE"], result.metrics["RMSE"], result.metrics["R2"])

    pred_df = pd.DataFrame(
        {
            "date": test_df.index,
            "actual": y_test.values,
            "predicted": result.predictions,
        }
    )

    return {
        "model_name": result.model_name,
        "metrics": result.metrics,
        "report_path": report_path,
        "predictions": pred_df,
    }
