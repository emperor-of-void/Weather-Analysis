from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PipelineConfig:
    data_path: Path = Path("DailyDelhiClimateTrain.csv")
    output_dir: Path = Path("outputs")
    test_ratio: float = 0.2
    random_state: int = 42
    n_estimators: int = 400
    model: str = "auto"
    save_plots: bool = True

    @staticmethod
    def from_json(path: Path) -> "PipelineConfig":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return PipelineConfig(
            data_path=Path(payload.get("data_path", "DailyDelhiClimateTrain.csv")),
            output_dir=Path(payload.get("output_dir", "outputs")),
            test_ratio=float(payload.get("test_ratio", 0.2)),
            random_state=int(payload.get("random_state", 42)),
            n_estimators=int(payload.get("n_estimators", 400)),
            model=str(payload.get("model", "auto")),
            save_plots=bool(payload.get("save_plots", True)),
        )

    def validate(self) -> None:
        if not (0 < self.test_ratio < 1):
            raise ValueError("test_ratio must be between 0 and 1")
        if self.model not in {"auto", "rf", "gbr", "xgb"}:
            raise ValueError("model must be one of: auto, rf, gbr, xgb")
