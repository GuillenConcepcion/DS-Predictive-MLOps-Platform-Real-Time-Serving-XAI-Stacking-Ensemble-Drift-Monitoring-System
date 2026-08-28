import json
import os
import pickle
from pathlib import Path

import lightgbm as lgb
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.data.make_dataset import load_raw_data
from src.features.build_features import TitanicFeaturePipeline
from src.models.threshold_optimizer import calibrate_model
from src.utils.logger import logger

MODELS_DIR = Path("models")
DATA_PROCESSED_DIR = Path("data/processed")


def _get_base_models():
    """Define los modelos base con parámetros robustos (o los últimos mejores)."""

    # 1. GBM (Mejores parámetros encontrados por Optuna)
    gbm_params = {
        "n_estimators": 450,
        "max_depth": 4,
        "learning_rate": 0.03836630493725778,
        "subsample": 0.6612627071218214,
        "min_samples_leaf": 17,
        "max_features": None,
        "random_state": 42,
    }

    # 2. LGBM (Parámetros robustos base)
    lgbm_params = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbose": -1,
    }

    # 3. XGBoost (Parámetros robustos base)
    xgb_params = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "logloss",
        "random_state": 42,
    }

    clf1 = GradientBoostingClassifier(**gbm_params)
    clf2 = lgb.LGBMClassifier(**lgbm_params)
    clf3 = xgb.XGBClassifier(**xgb_params)

    return [("gbm", clf1), ("lgbm", clf2), ("xgb", clf3)]


def train_and_evaluate_ensemble():
    """Entrena y evalúa un VotingClassifier con GBM, LGBM y XGBoost."""

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://172.17.212.149:5000")
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = "Titanic-Survival-MLOps"
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow conectado a: {tracking_uri} | Experimento: {experiment_name}")

    # 1. Cargar datos y features
    df_train, df_test = load_raw_data()
    pipeline = TitanicFeaturePipeline()
    X_train, y_train = pipeline.fit_transform(df_train)
    X_test = pipeline.transform(df_test)
    logger.info(f"Feature Engineering v2 completado. Features: {X_train.shape[1]}")

    # 1.5 Aplicar RFECV
    logger.info("Aplicando RFECV para selección de características...")
    rfecv_estimator = GradientBoostingClassifier(n_estimators=100, random_state=42)
    rfecv = RFECV(
        estimator=rfecv_estimator, step=1, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="roc_auc"
    )
    rfecv.fit(X_train, y_train)

    selected_features = X_train.columns[rfecv.support_]
    dropped_features = X_train.columns[~rfecv.support_]
    logger.success(f"RFECV completado. Features seleccionadas: {len(selected_features)} de {X_train.shape[1]}")
    logger.info(f"Features descartadas: {list(dropped_features)}")

    # Guardar selected_features para la API
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / "rfecv_selected_features.json", "w") as f:
        json.dump(selected_features.tolist(), f)

    X_train = X_train[selected_features]
    X_test = X_test[selected_features]

    with mlflow.start_run(run_name="Voting_Ensemble_calibrated") as run:
        logger.info(f"MLflow Run ID activo: {run.info.run_id}")

        mlflow.log_param("model_name", "Voting_Ensemble_RFECV")
        mlflow.log_param("ensemble_components", "GBM, LGBM, XGB")
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("rfecv_original_features", 38)
        mlflow.log_param("features_count", len(selected_features))
        mlflow.log_param("feature_engineering_version", "v2")
        mlflow.log_text(str(list(dropped_features)), "dropped_features.txt")
        mlflow.set_tag("author", "Guillen Concepcion")
        mlflow.set_tag("project", "Proyecto Odysseus")
        mlflow.set_tag("optimization_type", "ensemble_manual")

        # 3. Cross-Validated Threshold Optimization
        logger.info("Ejecutando optimización de threshold OOF 5-Fold para el ensemble...")

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof_probs = np.zeros(len(X_train))
        test_preds_folds = np.zeros(len(X_test))

        for _fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
            X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
            X_va = X_train.iloc[val_idx]

            model = VotingClassifier(estimators=_get_base_models(), voting="soft")
            model.fit(X_tr, y_tr)

            val_probs = model.predict_proba(X_va)[:, 1]
            oof_probs[val_idx] = val_probs

            test_preds_folds += model.predict_proba(X_test)[:, 1] / cv.n_splits

        # Optimización del threshold global basándonos en las probabilidades OOF
        thresholds = np.arange(0.1, 0.9, 0.01)
        best_threshold = 0.5
        best_f1 = 0.0
        for th in thresholds:
            preds = (oof_probs >= th).astype(int)
            f1 = f1_score(y_train, preds, average="macro")
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = th

        optimal_threshold = best_threshold
        mlflow.log_param("optimal_threshold", optimal_threshold)
        logger.success(f"Threshold óptimo OOF: {optimal_threshold:.3f}")

        # 4. Métricas globales OOF con threshold óptimo
        oof_classes = (oof_probs >= optimal_threshold).astype(int)
        acc = accuracy_score(y_train, oof_classes)
        auc = roc_auc_score(y_train, oof_probs)

        mlflow.log_metric("cv_accuracy", acc)
        mlflow.log_metric("cv_f1_macro", best_f1)
        mlflow.log_metric("cv_roc_auc", auc)

        logger.success("=======================================================")
        logger.success(f" RESULTADOS OOF ENSEMBLE (Threshold={optimal_threshold:.3f}):")
        logger.success(f" - Accuracy:  {acc:.4f} ({acc * 100:.2f}%)")
        logger.success(f" - F1-Macro:  {best_f1:.4f} ({best_f1 * 100:.2f}%)")
        logger.success(f" - ROC-AUC:   {auc:.4f} ({auc * 100:.2f}%)")
        logger.success("=======================================================")

        # 5. Entrenamiento del modelo final sobre TODOS los datos
        logger.info("Entrenando ensemble en todos los datos y calibrando (isotonic)...")
        final_model = VotingClassifier(estimators=_get_base_models(), voting="soft")
        final_model.fit(X_train, y_train)

        # 6. Calibración isotónica
        calibrated_model = calibrate_model(final_model, X_train, y_train, method="isotonic", cv=5)
        mlflow.log_param("calibration_method", "isotonic")

        # 7. Guardar modelo
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        local_model_path = MODELS_DIR / "titanic_ensemble_model.pkl"
        with open(local_model_path, "wb") as f_out:
            pickle.dump(calibrated_model, f_out)

        # Registrar en MLflow
        _trusted_types = [
            "sklearn.calibration._CalibratedClassifier",
            "sklearn.calibration.CalibratedClassifierCV",
            "sklearn.ensemble._gb.GradientBoostingClassifier",
            "sklearn.ensemble._voting.VotingClassifier",
            "sklearn.preprocessing._label.LabelEncoder",
            "numpy.ndarray",
            "builtins.NoneType",
        ]
        try:
            mlflow.sklearn.log_model(
                sk_model=calibrated_model,
                artifact_path="model",
                registered_model_name="Titanic_Ensemble_Optimized",
                skops_trusted_types=_trusted_types,
            )
            logger.success("Modelo ensemble registrado en MLflow Model Registry.")
        except Exception as e:
            logger.warning(f"No se pudo registrar en Model Registry: {e}")

        # 8. Predicciones Finales y Submission
        calibrated_probs_test = calibrated_model.predict_proba(X_test)[:, 1]
        test_binary_preds = (calibrated_probs_test >= optimal_threshold).astype(int)
        submission_df = pd.DataFrame({"PassengerId": df_test["PassengerId"], "Survived": test_binary_preds})

        DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        submission_file = DATA_PROCESSED_DIR / "titanic_ensemble_submission.csv"
        submission_df.to_csv(submission_file, index=False)
        mlflow.log_artifact(str(submission_file), artifact_path="predictions")
        logger.success(f"Submission generado: {submission_file} | {test_binary_preds.sum()} supervivientes predichos")

        logger.success("Corrida Ensemble completada exitosamente.")

    return {
        "accuracy": acc,
        "f1_macro": best_f1,
        "roc_auc": auc,
        "optimal_threshold": optimal_threshold,
        "mlflow_run_id": run.info.run_id,
    }


if __name__ == "__main__":
    train_and_evaluate_ensemble()
