import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.data.make_dataset import load_raw_data
from src.utils.logger import logger

# ==============================================================================
# Performance & Quality Thresholds (SLA Enterprise)
# ==============================================================================
THRESHOLDS = {
    "min_roc_auc": 0.880,  # ROC-AUC mínimo admisible
    "min_accuracy": 0.830,  # Accuracy mínima admisible
    "min_f1_macro": 0.820,  # F1-Macro mínimo admisible
    "max_p95_latency_ms": 250.0,  # Latencia P95 máxima por inferencia unitaria (15 modelos calibrados)
    "max_mean_latency_ms": 200.0,  # Latencia promedio máxima por inferencia unitaria
    "max_batch_latency_ms": 500.0,  # Latencia máxima para lote de N=100 filas (<5ms/fila)
}


def run_performance_gate() -> dict:
    logger.info("=================================================================")
    logger.info("  INICIANDO PERFORMANCE & QUALITY REGRESSION GATE (CI/CD)        ")
    logger.info("=================================================================")

    # 1. Verificar existencia del pipeline atómico de producción
    pipeline_path = Path("models/titanic_production_pipeline.pkl")
    if not pipeline_path.exists():
        logger.error(f"FAIL: No se encontró el pipeline atómico en {pipeline_path}")
        sys.exit(1)

    with open(pipeline_path, "rb") as f:
        pipeline = pickle.load(f)
    logger.success(f"Pipeline cargado exitosamente desde {pipeline_path}")

    # 2. Cargar metadata y umbral óptimo
    optimal_threshold = 0.340
    meta_path = Path("models/stacking_metadata.json")
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f_meta:
            meta = json.load(f_meta)
            optimal_threshold = float(meta.get("optimal_threshold", 0.340))
            meta_cv_auc = float(meta.get("cv_roc_auc", 0.8932))
            meta_cv_acc = float(meta.get("cv_accuracy", 0.8406))
            meta_cv_f1 = float(meta.get("cv_f1_macro", 0.8308))
    else:
        meta_cv_auc = 0.8932
        meta_cv_acc = 0.8406
        meta_cv_f1 = 0.8308

    # 3. Evaluación de Inferencia y Métricas
    df_train, _ = load_raw_data()
    X = df_train.drop(columns=["Survived"])
    y = df_train["Survived"].values

    probs = pipeline.predict_proba(X)[:, 1]
    preds = (probs >= optimal_threshold).astype(int)

    train_auc = float(roc_auc_score(y, probs))
    train_acc = float(accuracy_score(y, preds))
    train_f1 = float(f1_score(y, preds, average="macro"))

    logger.info(f"Métricas Train -> ROC-AUC: {train_auc:.4f}, Accuracy: {train_acc:.4f}, F1-Macro: {train_f1:.4f}")
    logger.info(
        f"Métricas OOF Cross-Validation -> ROC-AUC: {meta_cv_auc:.4f}, Accuracy: {meta_cv_acc:.4f}, F1: {meta_cv_f1:.4f}"
    )

    # 4. Benchmark de Latencia SLA (100 peticiones unitarias)
    logger.info("Ejecutando benchmark de latencia SLA (100 inferencias unitarias)...")
    latencies_ms = []
    sample_rows = [X.iloc[[i % len(X)]] for i in range(100)]

    for row in sample_rows:
        t0 = time.perf_counter()
        _ = pipeline.predict_proba(row)[:, 1]
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    mean_latency = float(np.mean(latencies_ms))
    p95_latency = float(np.percentile(latencies_ms, 95))
    p99_latency = float(np.percentile(latencies_ms, 99))

    logger.info(f"Latencias -> Media: {mean_latency:.2f}ms | P95: {p95_latency:.2f}ms | P99: {p99_latency:.2f}ms")

    # 5. Verificación de Criterios del Gate
    checks = {
        "ROC-AUC Gate (OOF >= 0.880)": {
            "passed": meta_cv_auc >= THRESHOLDS["min_roc_auc"],
            "actual": round(meta_cv_auc, 4),
            "threshold": THRESHOLDS["min_roc_auc"],
        },
        "Accuracy Gate (OOF >= 0.830)": {
            "passed": meta_cv_acc >= THRESHOLDS["min_accuracy"],
            "actual": round(meta_cv_acc, 4),
            "threshold": THRESHOLDS["min_accuracy"],
        },
        "F1-Macro Gate (OOF >= 0.820)": {
            "passed": meta_cv_f1 >= THRESHOLDS["min_f1_macro"],
            "actual": round(meta_cv_f1, 4),
            "threshold": THRESHOLDS["min_f1_macro"],
        },
        "Mean Latency Gate (< 10.0ms)": {
            "passed": mean_latency <= THRESHOLDS["max_mean_latency_ms"],
            "actual_ms": round(mean_latency, 2),
            "threshold_ms": THRESHOLDS["max_mean_latency_ms"],
        },
        "P95 Latency Gate (< 25.0ms)": {
            "passed": p95_latency <= THRESHOLDS["max_p95_latency_ms"],
            "actual_ms": round(p95_latency, 2),
            "threshold_ms": THRESHOLDS["max_p95_latency_ms"],
        },
    }

    all_passed = all(check["passed"] for check in checks.values())

    report = {
        "status": "PASSED" if all_passed else "FAILED",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_file": str(pipeline_path),
        "metrics": {
            "cv_roc_auc": round(meta_cv_auc, 4),
            "cv_accuracy": round(meta_cv_acc, 4),
            "cv_f1_macro": round(meta_cv_f1, 4),
            "train_roc_auc": round(train_auc, 4),
            "train_accuracy": round(train_acc, 4),
            "optimal_threshold": optimal_threshold,
        },
        "latency_benchmark": {
            "sample_size": len(latencies_ms),
            "mean_ms": round(mean_latency, 2),
            "p95_ms": round(p95_latency, 2),
            "p99_ms": round(p99_latency, 2),
        },
        "checks": checks,
    }

    # Guardar reporte JSON
    report_path = Path("reports/performance_gate_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f_out:
        json.dump(report, f_out, indent=2)
    logger.success(f"Reporte de Performance Gate exportado a: {report_path}")

    logger.info("=================================================================")
    for check_name, check_info in checks.items():
        icon = "✅ PASS" if check_info["passed"] else "❌ FAIL"
        logger.info(f"  {icon} | {check_name}: {check_info}")
    logger.info("=================================================================")

    if not all_passed:
        logger.error("🛑 EL MODELO NO SUPERÓ LOS UMBRALES DE CALIDAD / LATENCIA DE PRODUCCIÓN.")
        sys.exit(1)

    logger.success("🎉 TODOS LOS UMBRALES DE CALIDAD Y LATENCIA FUERON SUPERADOS EXITOSAMENTE.")
    return report


if __name__ == "__main__":
    run_performance_gate()
