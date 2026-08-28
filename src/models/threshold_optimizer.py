"""
threshold_optimizer.py — Calibracion de probabilidades y optimizacion de umbral de decision.

Implementa:
  1. Calibracion isotonica de probabilidades (CalibratedClassifierCV)
  2. Optimizacion del umbral de clasificacion que maximiza F1-Score (clase minoritaria)
  3. Exportacion del threshold optimo como parametro auditado en MLflow
"""

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.utils.logger import logger


def calibrate_model(model, X_train, y_train, method: str = "isotonic", cv: int = 5):
    """
    Aplica calibracion de probabilidades sobre el modelo ajustado.

    Args:
        model: Clasificador scikit-learn ya entrenado.
        X_train: Features de entrenamiento.
        y_train: Etiquetas de entrenamiento.
        method: 'isotonic' o 'sigmoid' (Platt Scaling).
        cv: Numero de folds para la calibracion cruzada.

    Returns:
        CalibratedClassifierCV ajustado.
    """
    logger.info(f"Calibrando probabilidades con metodo '{method}' (cv={cv})...")
    calibrated = CalibratedClassifierCV(estimator=model, method=method, cv=cv)
    calibrated.fit(X_train, y_train)
    logger.success("Calibracion completada. Modelo calibrado listo.")
    return calibrated


def optimize_threshold(
    model,
    X_val,
    y_val,
    threshold_range: tuple[float, float] = (0.30, 0.70),
    step: float = 0.01,
    metric: str = "f1_macro",
) -> dict:
    """
    Encuentra el umbral de decision optimo barriendo [threshold_range] con pasos de 'step'.

    Args:
        model: Clasificador con predict_proba().
        X_val: Features de validacion.
        y_val: Etiquetas verdaderas de validacion.
        threshold_range: Rango de busqueda (min, max).
        step: Paso del barrido.
        metric: Metrica a maximizar: 'f1_macro', 'f1_binary', 'accuracy'.

    Returns:
        Diccionario con optimal_threshold, best_metric_value, full_sweep_results.
    """
    probs = model.predict_proba(X_val)[:, 1]
    thresholds = np.arange(threshold_range[0], threshold_range[1] + step, step)

    results = []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1_mac = f1_score(y_val, preds, average="macro", zero_division=0)
        f1_bin = f1_score(y_val, preds, average="binary", zero_division=0)
        prec = precision_score(y_val, preds, zero_division=0)
        rec = recall_score(y_val, preds, zero_division=0)
        acc = float((preds == y_val).mean())
        results.append(
            {
                "threshold": round(float(t), 3),
                "f1_macro": round(f1_mac, 5),
                "f1_binary": round(f1_bin, 5),
                "precision": round(prec, 5),
                "recall": round(rec, 5),
                "accuracy": round(acc, 5),
            }
        )

    metric_key = {"f1_macro": "f1_macro", "f1_binary": "f1_binary", "accuracy": "accuracy"}.get(metric, "f1_macro")
    best_result = max(results, key=lambda x: x[metric_key])

    logger.success(
        f"Threshold optimo: {best_result['threshold']:.3f} | "
        f"{metric_key}: {best_result[metric_key]:.4f} | "
        f"Precision: {best_result['precision']:.4f} | Recall: {best_result['recall']:.4f}"
    )

    return {
        "optimal_threshold": best_result["threshold"],
        "best_metric": metric_key,
        "best_metric_value": best_result[metric_key],
        "precision": best_result["precision"],
        "recall": best_result["recall"],
        "accuracy": best_result["accuracy"],
        "sweep_results": results,
    }


def cross_validated_threshold_optimization(
    model_class,
    model_params: dict,
    X_train,
    y_train,
    n_splits: int = 5,
    metric: str = "f1_macro",
) -> dict:
    """
    Optimizacion del threshold con validacion cruzada para evitar sobreajuste al umbral.

    Calcula el threshold optimo como la media de los thresholds optimos de cada fold.

    Args:
        model_class: Clase del clasificador (e.g. GradientBoostingClassifier).
        model_params: Hiperparametros del modelo.
        X_train: Features de entrenamiento.
        y_train: Etiquetas de entrenamiento.
        n_splits: Numero de folds CV.
        metric: Metrica a maximizar.

    Returns:
        Diccionario con mean_optimal_threshold, std_threshold, mean_cv_auc.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_thresholds = []
    fold_aucs = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        X_tr = X_train.iloc[train_idx]
        y_tr = y_train.iloc[train_idx]
        X_va = X_train.iloc[val_idx]
        y_va = y_train.iloc[val_idx]

        model = model_class(**model_params)
        model.fit(X_tr, y_tr)

        probs_va = model.predict_proba(X_va)[:, 1]
        fold_auc = roc_auc_score(y_va, probs_va)
        fold_aucs.append(fold_auc)

        threshold_result = optimize_threshold(model, X_va, y_va, metric=metric)
        fold_thresholds.append(threshold_result["optimal_threshold"])
        logger.info(
            f"Fold {fold + 1}: AUC={fold_auc:.4f} | Optimal Threshold={threshold_result['optimal_threshold']:.3f}"
        )

    mean_threshold = float(np.mean(fold_thresholds))
    std_threshold = float(np.std(fold_thresholds))
    mean_auc = float(np.mean(fold_aucs))

    logger.success(f"Threshold CV: mean={mean_threshold:.3f} ± std={std_threshold:.3f} | Mean CV AUC: {mean_auc:.4f}")

    return {
        "mean_optimal_threshold": round(mean_threshold, 3),
        "std_threshold": round(std_threshold, 4),
        "mean_cv_auc": round(mean_auc, 6),
        "fold_thresholds": fold_thresholds,
    }
