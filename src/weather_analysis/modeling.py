from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass
class ModelResult:
    model_name: str
    model: object
    metrics: dict[str, float]
    predictions: np.ndarray


def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def _train_rf(X_train: pd.DataFrame, y_train: pd.Series, n_estimators: int, random_state: int) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        max_depth=14,
        min_samples_leaf=2,
    )
    model.fit(X_train, y_train)
    return model


def _train_gbr(X_train: pd.DataFrame, y_train: pd.Series, random_state: int) -> GradientBoostingRegressor:
    model = GradientBoostingRegressor(
        random_state=random_state,
        n_estimators=500,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.9,
    )
    model.fit(X_train, y_train)
    return model


def _train_xgb(X_train: pd.DataFrame, y_train: pd.Series, random_state: int) -> object:
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("xgboost is not installed") from exc

    model = XGBRegressor(
        n_estimators=700,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def choose_and_train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_choice: str,
    n_estimators: int,
    random_state: int,
) -> ModelResult:
    candidates: list[tuple[str, object]] = []

    if model_choice in {"rf", "auto"}:
        candidates.append(("RandomForest", _train_rf(X_train, y_train, n_estimators, random_state)))

    if model_choice in {"gbr", "auto"}:
        candidates.append(("GradientBoosting", _train_gbr(X_train, y_train, random_state)))

    if model_choice in {"xgb", "auto"}:
        try:
            candidates.append(("XGBoost", _train_xgb(X_train, y_train, random_state)))
        except RuntimeError:
            if model_choice == "xgb":
                raise

    if not candidates:
        raise RuntimeError("No model was trained")

    best_result: ModelResult | None = None
    for name, model in candidates:
        preds = model.predict(X_test)
        metrics = evaluate(y_test, preds)
        result = ModelResult(model_name=name, model=model, metrics=metrics, predictions=preds)

        if best_result is None or result.metrics["RMSE"] < best_result.metrics["RMSE"]:
            best_result = result

    if best_result is None:
        raise RuntimeError("Could not compute model results")

    return best_result
