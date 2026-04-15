from pathlib import Path

from src.weather_analysis.config import PipelineConfig
from src.weather_analysis.pipeline import run_pipeline


def test_pipeline_smoke(tmp_path: Path) -> None:
    cfg = PipelineConfig(
        data_path=Path("DailyDelhiClimateTrain.csv"),
        output_dir=tmp_path / "outputs",
        test_ratio=0.2,
        random_state=42,
        n_estimators=100,
        model="rf",
        save_plots=False,
    )

    result = run_pipeline(cfg)

    assert result["model_name"] == "RandomForest"
    assert "RMSE" in result["metrics"]
    assert (tmp_path / "outputs" / "report.md").exists()
