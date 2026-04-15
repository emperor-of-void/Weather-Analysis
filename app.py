from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.weather_analysis.config import PipelineConfig
from src.weather_analysis.pipeline import run_pipeline


st.set_page_config(page_title="Weather-Analysis", page_icon="🌤", layout="wide")

st.markdown(
    """
    <style>
    .hero {
        background: linear-gradient(120deg, #0f4c5c 0%, #1a759f 50%, #76c893 100%);
        padding: 1.2rem 1.6rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 1rem;
    }
    .kpi {
        border: 1px solid #d9e2ec;
        border-radius: 14px;
        padding: 0.9rem;
        background-color: #f8fbff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='hero'><h2>Weather-Analysis</h2><p>Phan tich va du bao nhiet do tu du lieu Kaggle Daily Delhi Climate.</p></div>", unsafe_allow_html=True)

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("Cau hinh")
    data_path = st.text_input("CSV path", value="DailyDelhiClimateTrain.csv")
    output_dir = st.text_input("Output folder", value="outputs")
    model = st.selectbox("Model", ["auto", "rf", "gbr", "xgb"], index=0)
    test_ratio = st.slider("Test ratio", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
    n_estimators = st.slider("n_estimators (RF)", min_value=100, max_value=1000, value=400, step=50)
    run_clicked = st.button("Run analysis", type="primary", use_container_width=True)

with col_right:
    st.subheader("Ket qua")
    if run_clicked:
        with st.spinner("Dang huan luyen va tao bao cao..."):
            cfg = PipelineConfig(
                data_path=Path(data_path),
                output_dir=Path(output_dir),
                test_ratio=float(test_ratio),
                random_state=42,
                n_estimators=int(n_estimators),
                model=model,
                save_plots=True,
            )
            result = run_pipeline(cfg)

        metrics = result["metrics"]
        k1, k2, k3 = st.columns(3)
        k1.markdown(f"<div class='kpi'><b>Model</b><br>{result['model_name']}</div>", unsafe_allow_html=True)
        k2.markdown(f"<div class='kpi'><b>MAE</b><br>{metrics['MAE']:.4f}</div>", unsafe_allow_html=True)
        k3.markdown(f"<div class='kpi'><b>RMSE</b><br>{metrics['RMSE']:.4f}</div>", unsafe_allow_html=True)

        pred_df: pd.DataFrame = result["predictions"]
        line = (
            alt.Chart(pred_df)
            .transform_fold(["actual", "predicted"], as_=["series", "temperature"])
            .mark_line(strokeWidth=2)
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("temperature:Q", title="Temperature (C)"),
                color=alt.Color("series:N", title="Series"),
            )
            .properties(height=360)
        )
        st.altair_chart(line, use_container_width=True)

        report_path = Path(result["report_path"])
        st.success(f"Da hoan thanh. Bao cao: {report_path}")

        for name in [
            "01_temperature_trend.png",
            "02_correlation_heatmap.png",
            "03_forecast_vs_actual.png",
            "04_feature_importance.png",
        ]:
            img_path = Path(output_dir) / name
            if img_path.exists():
                st.image(str(img_path), caption=name, use_container_width=True)
    else:
        st.info("Nhap cau hinh ben trai va nhan Run analysis.")
