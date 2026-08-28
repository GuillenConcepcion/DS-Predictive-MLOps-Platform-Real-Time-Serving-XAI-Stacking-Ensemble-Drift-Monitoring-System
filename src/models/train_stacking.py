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
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.data.make_dataset import load_raw_data
from src.features.build_features import ColumnSelector, TitanicFeaturePipeline
from src.models.registry_manager import ModelRegistryManager
from src.models.threshold_optimizer import calibrate_model
from src.utils.logger import logger

MODELS_DIR = Path("models")
DATA_PROCESSED_DIR = Path("data/processed")


def _get_base_models():
    """Define los modelos base de Nivel 1 con hiperparámetros optimizados."""

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


def _build_stacking_classifier():
    """Construye el StackingClassifier con Meta-Learner LogisticRegression regularizado."""
    meta_learner = LogisticRegression(
        C=0.1,
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
    )
    cv_internal = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return StackingClassifier(
        estimators=_get_base_models(),
        final_estimator=meta_learner,
        cv=cv_internal,
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=-1,
    )


def train_and_evaluate_stacking():
    """
    Entrena y evalúa un Stacking Classifier de 2 Niveles:
      - Nivel 1: GBM + LGBM + XGBoost
      - Nivel 2: Logistic Regression (L2, C=0.1)
      - Feature Selection: RFECV (14 variables de élite)
      - Calibración: Isotonic Calibration (cv=5)
      - Optimización de Threshold OOF para maximizar F1-Macro
    """
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

    # 2. Selección de Características con RFECV
    logger.info("Aplicando RFECV para selección de características...")
    rfecv_estimator = GradientBoostingClassifier(n_estimators=100, random_state=42)
    rfecv = RFECV(
        estimator=rfecv_estimator,
        step=1,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring="roc_auc",
        n_jobs=-1,
    )
    rfecv.fit(X_train, y_train)

    selected_features = X_train.columns[rfecv.support_]
    dropped_features = X_train.columns[~rfecv.support_]
    logger.success(f"RFECV completado. Features seleccionadas: {len(selected_features)} de {X_train.shape[1]}")
    logger.info(f"Features descartadas: {list(dropped_features)}")

    # Guardar selected_features para la API
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / "rfecv_selected_features.json", "w", encoding="utf-8") as f:
        json.dump(selected_features.tolist(), f, indent=2)

    X_train_sel = X_train[selected_features]
    X_test_sel = X_test[selected_features]

    with mlflow.start_run(run_name="Stacking_Ensemble_MetaLearner_L2") as run:
        logger.info(f"MLflow Run ID activo: {run.info.run_id}")

        mlflow.log_param("model_name", "StackingClassifier_L2_MetaLearner")
        mlflow.log_param("meta_learner", "LogisticRegression(C=0.1, penalty='l2')")
        mlflow.log_param("base_estimators", "GBM, LGBM, XGBoost")
        mlflow.log_param("stack_method", "predict_proba")
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("rfecv_original_features", X_train.shape[1])
        mlflow.log_param("rfecv_selected_features", len(selected_features))
        mlflow.log_param("feature_engineering_version", "v2")
        mlflow.log_text(str(list(dropped_features)), "dropped_features.txt")
        mlflow.set_tag("author", "Guillen Concepcion")
        mlflow.set_tag("project", "Proyecto Odysseus")
        mlflow.set_tag("architecture", "2-Level-Stacking-MLOps")

        # 3. Validación Cruzada OOF 5-Fold para estimar probabilidades y optimizar Threshold
        logger.info("Ejecutando validación OOF 5-Fold con StackingClassifier...")
        cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof_probs = np.zeros(len(X_train_sel))
        test_preds_folds = np.zeros(len(X_test_sel))

        for fold, (train_idx, val_idx) in enumerate(cv_outer.split(X_train_sel, y_train)):
            X_tr, y_tr = X_train_sel.iloc[train_idx], y_train.iloc[train_idx]
            X_va, y_va = X_train_sel.iloc[val_idx], y_train.iloc[val_idx]

            fold_model = _build_stacking_classifier()
            fold_model.fit(X_tr, y_tr)

            val_probs = fold_model.predict_proba(X_va)[:, 1]
            oof_probs[val_idx] = val_probs
            test_preds_folds += fold_model.predict_proba(X_test_sel)[:, 1] / cv_outer.n_splits

            fold_auc = roc_auc_score(y_va, val_probs)
            logger.info(f"Fold {fold + 1}/5 ROC-AUC: {fold_auc:.4f}")
            mlflow.log_metric(f"fold_{fold + 1}_roc_auc", fold_auc)

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

        optimal_threshold = round(float(best_threshold), 3)
        mlflow.log_param("optimal_threshold", optimal_threshold)
        logger.success(f"Threshold óptimo OOF: {optimal_threshold:.3f}")

        # 4. Métricas globales OOF con threshold óptimo
        oof_classes = (oof_probs >= optimal_threshold).astype(int)
        acc = accuracy_score(y_train, oof_classes)
        auc = roc_auc_score(y_train, oof_probs)

        mlflow.log_metric("cv_accuracy", acc)
        mlflow.log_metric("cv_f1_macro", best_f1)
        mlflow.log_metric("cv_roc_auc", auc)

        logger.success("=" * 65)
        logger.success(f" RESULTADOS OOF STACKING CLASSIFIER (Threshold={optimal_threshold:.3f}):")
        logger.success(f" - Accuracy:  {acc:.4f} ({acc * 100:.2f}%)")
        logger.success(f" - F1-Macro:  {best_f1:.4f} ({best_f1 * 100:.2f}%)")
        logger.success(f" - ROC-AUC:   {auc:.4f} ({auc * 100:.2f}%)")
        logger.success("=" * 65)

        # 5. Entrenamiento del modelo final sobre TODOS los datos y extracción de pesos
        logger.info("Entrenando Stacking Classifier final sobre todo el dataset...")
        final_stacking = _build_stacking_classifier()
        final_stacking.fit(X_train_sel, y_train)

        # Extraer coeficientes del Meta-Learner
        meta_coefs = final_stacking.final_estimator_.coef_[0]
        meta_intercept = float(final_stacking.final_estimator_.intercept_[0])
        model_names = [name for name, _ in final_stacking.estimators]

        # En predict_proba binario, scikit-learn puede pasar prob(clase 1) o prob(clase 0) y prob(clase 1)
        logger.info(f"Meta-Learner Intercept (b): {meta_intercept:.4f}")
        logger.info(f"Meta-Learner Coeficientes crudos: {meta_coefs}")

        for idx, name in enumerate(model_names):
            if idx < len(meta_coefs):
                weight = float(meta_coefs[idx])
                mlflow.log_metric(f"meta_weight_{name}", weight)
                logger.info(f" -> Peso asignado a {name.upper()}: {weight:.4f}")

        # 6. Calibración isotónica del Stacking Classifier final
        logger.info("Calibrando probabilidades con método 'isotonic' (cv=5)...")
        calibrated_stacking = calibrate_model(final_stacking, X_train_sel, y_train, method="isotonic", cv=5)
        mlflow.log_param("calibration_method", "isotonic")

        # 7. Construir y persistir Pipeline Atómico Unificado de Producción
        logger.info("Construyendo Pipeline Atómico de Producción (sklearn.pipeline.Pipeline)...")
        production_pipeline = Pipeline(
            [
                ("features", pipeline),
                ("selector", ColumnSelector(columns=selected_features.tolist())),
                ("model", calibrated_stacking),
            ]
        )

        pipeline_model_path = MODELS_DIR / "titanic_production_pipeline.pkl"
        with open(pipeline_model_path, "wb") as f_out:
            pickle.dump(production_pipeline, f_out)
        logger.success(f"Pipeline Atómico serializado en: {pipeline_model_path}")

        # Mantener retrocompatibilidad
        stacking_model_path = MODELS_DIR / "titanic_stacking_model.pkl"
        with open(stacking_model_path, "wb") as f_out:
            pickle.dump(calibrated_stacking, f_out)

        ensemble_model_path = MODELS_DIR / "titanic_ensemble_model.pkl"
        with open(ensemble_model_path, "wb") as f_out:
            pickle.dump(production_pipeline, f_out)
        logger.success(f"Modelo de producción actualizado en: {ensemble_model_path}")

        # Guardar artefacto de pipeline en MLflow
        mlflow.log_artifact(str(pipeline_model_path), artifact_path="pipeline")

        # Exportar metadata estructurada
        stacking_meta = {
            "model_name": "Titanic_Stacking_Classifier_L2",
            "meta_learner": "LogisticRegression(C=0.1)",
            "cv_accuracy": round(acc, 4),
            "cv_f1_macro": round(best_f1, 4),
            "cv_roc_auc": round(auc, 4),
            "optimal_threshold": optimal_threshold,
            "rfecv_features_count": len(selected_features),
            "selected_features": selected_features.tolist(),
            "meta_intercept": meta_intercept,
            "meta_coefs": meta_coefs.tolist(),
            "pipeline_path": str(pipeline_model_path),
        }
        with open(MODELS_DIR / "stacking_metadata.json", "w", encoding="utf-8") as f_meta:
            json.dump(stacking_meta, f_meta, indent=2)

        # 8. Generar Submission para Kaggle
        calibrated_probs_test = production_pipeline.predict_proba(df_test)[:, 1]
        test_binary_preds = (calibrated_probs_test >= optimal_threshold).astype(int)
        submission_df = pd.DataFrame({"PassengerId": df_test["PassengerId"], "Survived": test_binary_preds})

        DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        submission_file = DATA_PROCESSED_DIR / "titanic_stacking_submission.csv"
        submission_df.to_csv(submission_file, index=False)
        mlflow.log_artifact(str(submission_file), artifact_path="predictions")
        logger.success(
            f"Submission Stacking generado con Pipeline Atómico: {submission_file} | {test_binary_preds.sum()} supervivientes predichos"
        )

        # 9. Promoción automática en MLflow Model Registry (Champion / Challenger)
        try:
            registry_mgr = ModelRegistryManager(tracking_uri=tracking_uri)
            promo_res = registry_mgr.register_and_promote(
                run_id=run.info.run_id,
                artifact_path="pipeline",
                candidate_auc=auc,
                baseline_threshold=0.880,
            )
            logger.info(f"Resultado Model Registry: {promo_res}")
        except Exception as e:
            logger.warning(f"No se pudo completar el ciclo de Model Registry: {e}")

        logger.success("Entrenamiento y empaquetado de Pipeline Atómico completado exitosamente.")

    return stacking_meta


if __name__ == "__main__":
    train_and_evaluate_stacking()
