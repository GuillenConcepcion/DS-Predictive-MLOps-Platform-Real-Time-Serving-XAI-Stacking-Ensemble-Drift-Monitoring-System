import json
import pickle
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from src.data.make_dataset import load_raw_data
from src.monitoring.drift_detector import DataDriftDetector
from src.serving.schemas import (
    BatchPredictionInput,
    BatchPredictionOutput,
    InferenceMetricsResponse,
    ModelMetadataResponse,
    PassengerInput,
    PredictionDriftResponse,
    PredictionOutput,
)
from src.utils.logger import logger

# Estado global del pipeline unificado y monitor de drift en memoria
production_pipeline = None
drift_detector = None
reference_raw_df = None
top_shap_features_cache = []
optimal_threshold = 0.340

# Buffers de observabilidad y prediction drift en tiempo real
prediction_history_probs: deque[float] = deque(maxlen=2000)
prediction_history_preds: deque[int] = deque(maxlen=2000)
total_inference_count: int = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    global production_pipeline, drift_detector, reference_raw_df, top_shap_features_cache, optimal_threshold
    logger.info("Inicializando artefactos de inferencia para producción...")

    # 1. Dataset de referencia para detección de Drift
    df_train, _ = load_raw_data()
    reference_raw_df = df_train.copy()
    drift_detector = DataDriftDetector(reference_data=df_train, drift_share_threshold=0.33)
    logger.success("Detector de Data Drift inicializado con dataset de referencia.")

    # 2. Cargar Pipeline Atómico Unificado de Producción
    pipeline_path = Path("models/titanic_production_pipeline.pkl")
    if not pipeline_path.exists():
        pipeline_path = Path("models/titanic_ensemble_model.pkl")

    if not pipeline_path.exists():
        logger.warning(f"No se encontró pipeline en {pipeline_path}. Entrenando fallback...")
        from src.models.train_stacking import train_and_evaluate_stacking

        train_and_evaluate_stacking()
        pipeline_path = Path("models/titanic_production_pipeline.pkl")

    with open(pipeline_path, "rb") as f:
        production_pipeline = pickle.load(f)
    logger.success(f"Pipeline Atómico de Producción cargado exitosamente desde {pipeline_path}")

    # 3. Cargar umbral óptimo y metadata
    meta_path = Path("models/stacking_metadata.json")
    if meta_path.exists():
        try:
            with open(meta_path, encoding="utf-8") as f_meta:
                meta = json.load(f_meta)
                optimal_threshold = float(meta.get("optimal_threshold", 0.340))
        except Exception:
            optimal_threshold = 0.340

    # 4. Establecer baseline de predicciones en el detector de drift
    try:
        ref_probs = production_pipeline.predict_proba(reference_raw_df)[:, 1]
        ref_preds = (ref_probs >= optimal_threshold).astype(int)
        drift_detector.set_reference_predictions(ref_probs, ref_preds)
        logger.success("Baseline de Prediction Drift establecido exitosamente.")
    except Exception as e:
        logger.warning(f"No se pudo establecer baseline de prediction drift: {e}")

    # 5. Cargar top SHAP features si existen
    shap_path = Path("reports/shap_feature_importance.csv")
    if shap_path.exists():
        shap_df = pd.read_csv(shap_path)
        top_shap_features_cache = shap_df["feature"].head(5).tolist()
    else:
        top_shap_features_cache = ["Title_Mr", "Pclass_3", "Sex_male", "Fare", "Age"]

    yield
    logger.info("Apagando servicio de inferencia de Titanic MLOps.")


app = FastAPI(
    title="Titanic Survival Inference API - Odysseus AI",
    description="Microservicio de inferencia en tiempo real para predicción de supervivencia en el Titanic con validación Pydantic, explicabilidad y soporte batch.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "service": "titanic-survival-serving-api",
        "pipeline_loaded": production_pipeline is not None,
        "model_loaded": production_pipeline is not None,
        "drift_detector_ready": drift_detector is not None,
        "optimal_threshold": optimal_threshold,
    }


@app.get("/model/metadata", response_model=ModelMetadataResponse, tags=["Model Info"])
async def get_model_metadata():
    return ModelMetadataResponse(
        model_name="Titanic_Survival_Production_Pipeline",
        framework="Scikit-Learn (Atomic Pipeline)",
        algorithm="Unified Atomic Pipeline (Bayesian TE + RFECV + Calibrated Stacking L2)",
        cv_accuracy=0.8406,
        cv_roc_auc=0.8896,
        features_count=14,
        top_shap_features=top_shap_features_cache,
    )


def _predict_dataframe(df: pd.DataFrame) -> list[PredictionOutput]:
    global total_inference_count
    # Inferencia atómica directa sobre el DataFrame crudo
    probabilities = production_pipeline.predict_proba(df)[:, 1]
    predictions = (probabilities >= optimal_threshold).astype(int)

    # Registrar en el buffer de observabilidad
    for p_val, y_val in zip(probabilities, predictions, strict=False):
        prediction_history_probs.append(float(p_val))
        prediction_history_preds.append(int(y_val))
        total_inference_count += 1

    results = []
    for idx, (prob, pred) in enumerate(zip(probabilities, predictions, strict=False)):
        pid = (
            int(df.iloc[idx]["PassengerId"])
            if "PassengerId" in df.columns and pd.notnull(df.iloc[idx]["PassengerId"])
            else None
        )

        if prob >= 0.70:
            risk = "High"
        elif prob >= 0.40:
            risk = "Moderate"
        else:
            risk = "Low"

        results.append(
            PredictionOutput(
                passenger_id=pid,
                prediction=int(pred),
                survival_probability=round(float(prob), 4),
                status="Survived" if pred == 1 else "Did Not Survive",
                risk_level=risk,
            )
        )
    return results


@app.post("/predict", response_model=PredictionOutput, tags=["Inference"])
async def predict_single(passenger: PassengerInput):
    """
    Inferencia unitaria en tiempo real para un pasajero.
    """
    if production_pipeline is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible en este momento.")

    df = pd.DataFrame([passenger.model_dump()])
    output = _predict_dataframe(df)[0]
    return output


@app.post("/predict/batch", response_model=BatchPredictionOutput, tags=["Inference"])
async def predict_batch(batch: BatchPredictionInput):
    """
    Inferencia por lotes de alta concurrencia.
    """
    if production_pipeline is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible en este momento.")

    if not batch.passengers:
        raise HTTPException(status_code=400, detail="La lista de pasajeros no puede estar vacía.")

    df = pd.DataFrame([p.model_dump() for p in batch.passengers])
    predictions = _predict_dataframe(df)

    survived_count = sum(p.prediction for p in predictions)
    rate = round(survived_count / len(predictions), 4)

    return BatchPredictionOutput(total_samples=len(predictions), survival_rate=rate, predictions=predictions)


@app.post("/monitoring/drift", tags=["Monitoring"])
async def evaluate_data_drift(batch: BatchPredictionInput):
    """
    Evalúa si la distribución de datos recibidos presenta Data Drift estadístico
    respecto al conjunto de referencia de entrenamiento (KS-Test, PSI, Chi2).
    """
    if drift_detector is None:
        raise HTTPException(status_code=503, detail="Detector de Drift no inicializado.")

    if not batch.passengers:
        raise HTTPException(status_code=400, detail="La lista de pasajeros no puede estar vacía.")

    df_incoming = pd.DataFrame([p.model_dump() for p in batch.passengers])
    drift_results = drift_detector.detect_drift(df_incoming)

    # Generar y persistir reporte HTML
    report_file = "reports/live_drift_report.html"
    drift_detector.generate_html_report(df_incoming, output_path=report_file)

    return {
        "status": "success",
        "dataset_drift": drift_results["dataset_drift"],
        "drift_share": drift_results["drift_share"],
        "number_of_drifted_features": drift_results["number_of_drifted_features"],
        "number_of_features": drift_results["number_of_features"],
        "reference_rows": drift_results["reference_rows"],
        "current_rows": drift_results["current_rows"],
        "drift_by_column": drift_results["drift_by_column"],
        "html_dashboard_url": "/monitoring/drift/dashboard",
    }


@app.get("/monitoring/prediction-drift", response_model=PredictionDriftResponse, tags=["Monitoring"])
async def get_prediction_drift(alpha: float = 0.05):
    """
    Evalúa si la distribución de probabilidades predichas y decisiones en vivo
    presenta drift estadístico frente al baseline de entrenamiento (PSI, KS-Test, TVD).
    """
    if drift_detector is None:
        raise HTTPException(status_code=503, detail="Detector de Drift no disponible.")

    probs_arr = np.array(prediction_history_probs)
    preds_arr = np.array(prediction_history_preds)

    drift_report = drift_detector.calculate_prediction_drift(probs_arr, preds_arr, alpha=alpha)
    return drift_report


@app.get("/monitoring/inference-metrics", response_model=InferenceMetricsResponse, tags=["Monitoring"])
async def get_inference_metrics():
    """
    Retorna métricas agregadas de observabilidad sobre la cola de inferencias recientes.
    """
    buf_size = len(prediction_history_probs)
    if buf_size > 0:
        avg_prob = round(float(np.mean(prediction_history_probs)), 4)
        pos_rate = round(float(np.mean(prediction_history_preds)), 4)
        status = "HEALTHY_OPERATIONAL"
    else:
        avg_prob = 0.0
        pos_rate = 0.0
        status = "AWAITING_INFERENCES"

    return InferenceMetricsResponse(
        total_inferences=total_inference_count,
        current_buffer_size=buf_size,
        average_survival_probability=avg_prob,
        positive_prediction_rate=pos_rate,
        optimal_threshold=optimal_threshold,
        buffer_status=status,
    )


@app.get("/monitoring/drift/dashboard", response_class=HTMLResponse, tags=["Monitoring"])
async def get_drift_dashboard():
    """
    Renderiza el dashboard interactivo HTML del último análisis de Data Drift.
    """
    report_file = Path("reports/live_drift_report.html")
    if not report_file.exists():
        # Generar un reporte inicial usando test.csv como muestra representativa
        if drift_detector is not None:
            _, df_test = load_raw_data()
            drift_detector.generate_html_report(df_test, output_path=str(report_file))
        else:
            raise HTTPException(status_code=404, detail="No se ha generado ningún reporte de drift todavía.")

    with open(report_file, encoding="utf-8") as f:
        html_body = f.read()

    return HTMLResponse(content=html_body, status_code=200)


@app.get("/monitoring/eda/dashboard", response_class=HTMLResponse, tags=["Analytics & BI"])
async def get_eda_bi_dashboard():
    """
    Renderiza el Dashboard Ejecutivo Interactivo de EDA, Demografía y Business Intelligence (BI).
    """
    dashboard_file = Path("reports/eda_bi_dashboard.html")
    if not dashboard_file.exists():
        from src.data.eda_bi_generator import generate_eda_bi_artifacts

        generate_eda_bi_artifacts(output_dir="reports")

    with open(dashboard_file, encoding="utf-8") as f:
        html_body = f.read()

    return HTMLResponse(content=html_body, status_code=200)
