# Weather-Analysis (Kaggle Delhi Climate)

Du an Weather-Analysis day du gom 3 phan:

- App giao dien bang Streamlit
- Pipeline du bao nhiet do nang cap (auto chon mo hinh tot nhat)
- Cau truc production voi src/, config/, tests/ va CI

## 1) Cai dat

```bash
pip install -r requirements.txt
```

Hoac cai dat theo package mode:

```bash
pip install -e .
```

## 2) Chay pipeline CLI

Theo config mac dinh:

```bash
python weather_analysis.py --config config/default.json
```

Tuy chinh truc tiep tham so:

```bash
python weather_analysis.py --data DailyDelhiClimateTrain.csv --out outputs --model auto --test-ratio 0.2
```

Lua chon model:

- auto: thu RF + GradientBoosting + XGBoost (neu co) va chon model RMSE tot nhat
- rf: Random Forest
- gbr: Gradient Boosting Regressor
- xgb: XGBoost

## 3) Chay app Streamlit

```bash
streamlit run app.py
```

Trong app ban co the:

- Chon model
- Doi test ratio va n_estimators
- Chay phan tich va xem chart Actual vs Predicted
- Xem cac hinh anh report duoc tao tu dong

## 4) Kiem thu

```bash
pytest -q
```

## 5) CI

Workflow GitHub Actions nam tai .github/workflows/ci.yml

- Cai dat package
- Chay test
- Chay CLI smoke test

## 6) Cau truc thu muc

```text
.
|- src/weather_analysis/
|  |- config.py
|  |- data.py
|  |- features.py
|  |- modeling.py
|  |- reporting.py
|  |- pipeline.py
|- config/default.json
|- tests/test_pipeline_smoke.py
|- app.py
|- weather_analysis.py
|- .github/workflows/ci.yml
```

## 7) Dau ra

Trong thu muc outputs/:

- 01_temperature_trend.png
- 02_correlation_heatmap.png
- 03_forecast_vs_actual.png
- 04_feature_importance.png
- report.md
