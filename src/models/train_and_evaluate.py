import json
import os
from pathlib import Path

import lightgbm as lgb
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.data.make_dataset import load_raw_data
from src.features.build_features import TitanicFeaturePipeline
from src.models.threshold_optimizer import calibrate_model, cross_validated_threshold_optimization
from src.utils.logger import logger

MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
DATA_PROCESSED_DIR = Path("data/processed")


def _load_best_params() -> tuple[str, dict]:
    """
    Carga el mejor modelo y sus hiperparametros desde best_params.json (generado por Optuna).
    Si no existe, usa GradientBoostingClassifier con parametros probados en benchmark.
    """
    best_params_file = MODELS_DIR / "best_params.json"
    if best_params_file.exists():
        with open(best_params_file, encoding="utf-8") as f:
            data = json.load(f)
        model_name = data.get("model_name", "GBM")
        params = data.get("hyperparameters", {})
        cv_auc = data.get("cv_roc_auc", 0.0)
        logger.info(f"Cargando best_params.json: modelo={model_name}, CV AUC={cv_auc:.4f}")
        return model_name, params

    # Fallback: mejor configuracion del benchmark previo
    logger.warning("best_params.json no encontrado. Usando GBM con parametros del benchmark.")
    return "GBM", {
        "n_estimators": 150,
        "max_depth": 4,
        "learning_rate": 0.04,
        "subsample": 0.85,
        "min_samples_leaf": 2,
        "random_state": 42,
    }


def _build_classifier(model_name: str, params: dict):
    """Instancia el clasificador optimo con los hiperparametros dados."""
    # random_state puede venir en params desde best_params.json
    params.setdefault("random_state", 42)
    if model_name == "GBM":
        return GradientBoostingClassifier(**params)
    elif model_name == "LGBM":
        return lgb.LGBMClassifier(**params, verbose=-1)
    else:  # XGB
        params.setdefault("eval_metric", "logloss")
        return xgb.XGBClassifier(**params)


def train_and_evaluate():
    """
    Pipeline de entrenamiento final integrado con:
      - Feature Engineering v2 (32+ features)
      - Hiperparametros cargados desde best_params.json (Optuna Multi-Modelo)
      - Calibracion isotonica de probabilidades
      - Threshold Optimization CV para maximizar F1-Macro
      - Registro completo en MLflow con artefactos XAI (SHAP)
    """
    # 1. Configurar MLflow
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://172.17.212.149:5000")
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = "Titanic-Survival-MLOps"
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow conectado a: {tracking_uri} | Experimento: {experiment_name}")

    # 2. Cargar datos y aplicar Feature Engineering v2
    df_train, df_test = load_raw_data()
    pipeline = TitanicFeaturePipeline()
    X_train, y_train = pipeline.fit_transform(df_train)
    X_test = pipeline.transform(df_test)
    logger.info(f"Feature Engineering v2 completado. Features: {X_train.shape[1]}")

    # 3. Cargar mejor modelo y parametros de Optuna
    model_name, best_params = _load_best_params()
    # Eliminar random_state si ya fue serializado para evitar duplicados
    best_params.pop("random_state", None)

    with mlflow.start_run(run_name=f"{model_name}_calibrated_v2") as run:
        logger.info(f"MLflow Run ID activo: {run.info.run_id}")

        # Log de metadata y parametros
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("features_count", X_train.shape[1])
        mlflow.log_param("feature_engineering_version", "v2")
        mlflow.log_param("imputation_strategy", "knn_k5_with_indicator")
        mlflow.log_params({f"hp_{k}": v for k, v in best_params.items()})
        mlflow.set_tag("author", "Guillen Concepcion")
        mlflow.set_tag("project", "Proyecto Odysseus")
        mlflow.set_tag("optimization_type", "optuna_multimodel_v2")

        # 4. Cross-Validated Threshold Optimization
        logger.info("Ejecutando optimizacion de threshold con validacion cruzada...")
        model_class = (
            GradientBoostingClassifier
            if model_name == "GBM"
            else (lgb.LGBMClassifier if model_name == "LGBM" else xgb.XGBClassifier)
        )
        threshold_result = cross_validated_threshold_optimization(
            model_class=model_class,
            model_params=dict(best_params),
            X_train=X_train,
            y_train=y_train,
            n_splits=5,
            metric="f1_macro",
        )
        optimal_threshold = threshold_result["mean_optimal_threshold"]
        mlflow.log_param("optimal_threshold", optimal_threshold)
        mlflow.log_metric("cv_mean_threshold", optimal_threshold)
        mlflow.log_metric("cv_threshold_std", threshold_result["std_threshold"])
        logger.success(f"Threshold optimo CV: {optimal_threshold:.3f}")

        # 5. Entrenamiento OOF 5-Fold para metricas finales con threshold optimo
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof_probs = np.zeros(len(X_train))
        test_preds_folds = np.zeros(len(X_test))

        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
            X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
            X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]

            model = _build_classifier(model_name, dict(best_params))
            model.fit(X_tr, y_tr)

            val_probs = model.predict_proba(X_va)[:, 1]
            oof_probs[val_idx] = val_probs

            fold_auc = roc_auc_score(y_va, val_probs)
            fold_acc = accuracy_score(y_va, (val_probs >= optimal_threshold).astype(int))
            mlflow.log_metric(f"fold_{fold + 1}_roc_auc", fold_auc)
            mlflow.log_metric(f"fold_{fold + 1}_accuracy", fold_acc)

            test_preds_folds += model.predict_proba(X_test)[:, 1] / cv.n_splits

        # 6. Metricas globales OOF con threshold optimo
        oof_classes = (oof_probs >= optimal_threshold).astype(int)
        acc = accuracy_score(y_train, oof_classes)
        f1 = f1_score(y_train, oof_classes, average="macro")
        auc = roc_auc_score(y_train, oof_probs)

        mlflow.log_metric("cv_accuracy", acc)
        mlflow.log_metric("cv_f1_macro", f1)
        mlflow.log_metric("cv_roc_auc", auc)

        logger.success("=======================================================")
        logger.success(f" RESULTADOS OOF v2 ({model_name} + Threshold={optimal_threshold:.3f}):")
        logger.success(f" - Accuracy:  {acc:.4f} ({acc * 100:.2f}%)")
        logger.success(f" - F1-Macro:  {f1:.4f} ({f1 * 100:.2f}%)")
        logger.success(f" - ROC-AUC:   {auc:.4f} ({auc * 100:.2f}%)")
        logger.success("=======================================================")

        # 7. Entrenamiento del modelo final sobre TODOS los datos de train
        final_model = _build_classifier(model_name, dict(best_params))
        final_model.fit(X_train, y_train)

        # 8. Calibracion isotonica del modelo final
        logger.info("Calibrando probabilidades del modelo final (metodo isotonic)...")
        calibrated_model = calibrate_model(final_model, X_train, y_train, method="isotonic", cv=5)
        mlflow.log_param("calibration_method", "isotonic")

        # Guardar modelo serializado (sin calibracion para XGBoost nativo)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        if model_name == "XGB":
            local_model_path = MODELS_DIR / "titanic_best_model.json"
            final_model.save_model(str(local_model_path))
        else:
            import pickle

            local_model_path = MODELS_DIR / "titanic_best_model.pkl"
            with open(local_model_path, "wb") as f_out:
                pickle.dump(calibrated_model, f_out)

        # Registrar modelo calibrado en MLflow Model Registry
        # skops_trusted_types requerido para CalibratedClassifierCV desde mlflow >= 2.11
        _trusted_types = [
            "sklearn.calibration._CalibratedClassifier",
            "sklearn.calibration.CalibratedClassifierCV",
            "sklearn.ensemble._gb.GradientBoostingClassifier",
            "sklearn.ensemble._gb.GradientBoostingRegressor",
            "sklearn.preprocessing._label.LabelEncoder",
            "numpy.ndarray",
            "builtins.NoneType",
        ]
        try:
            mlflow.sklearn.log_model(
                sk_model=calibrated_model,
                artifact_path="model",
                registered_model_name="Titanic_Survival_Optimized_v2",
                skops_trusted_types=_trusted_types,
            )
            logger.success("Modelo calibrado registrado en MLflow Model Registry como 'Titanic_Survival_Optimized_v2'")
        except Exception as e:
            logger.warning(f"No se pudo registrar en Model Registry (error skops): {e}")
            logger.info("Continuando sin registro en Model Registry — modelo guardado en disco.")

        # 9. Explicabilidad XAI con SHAP (sobre modelo no calibrado, TreeExplainer compatible)
        logger.info("Calculando SHAP values y subiendo artefactos...")
        try:
            explainer = shap.TreeExplainer(final_model)
            shap_values = explainer.shap_values(X_train)
            mean_abs_shap = np.abs(shap_values).mean(axis=0)

            feature_importance_df = pd.DataFrame(
                {"feature": X_train.columns, "mean_shap_value": mean_abs_shap}
            ).sort_values(by="mean_shap_value", ascending=False)

            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            shap_file = REPORTS_DIR / "shap_feature_importance_v2.csv"
            feature_importance_df.to_csv(shap_file, index=False)
            mlflow.log_artifact(str(shap_file), artifact_path="explainability")
            logger.success(f"SHAP feature importance exportado: {shap_file}")
        except Exception as e:
            logger.warning(f"SHAP no compatible con modelo calibrado: {e}. Omitiendo SHAP.")

        # 10. Generar submission con probabilidades calibradas + threshold optimo
        calibrated_probs_test = calibrated_model.predict_proba(X_test)[:, 1]
        test_binary_preds = (calibrated_probs_test >= optimal_threshold).astype(int)
        submission_df = pd.DataFrame({"PassengerId": df_test["PassengerId"], "Survived": test_binary_preds})

        DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        submission_file = DATA_PROCESSED_DIR / "titanic_best_model_submission.csv"
        submission_df.to_csv(submission_file, index=False)
        mlflow.log_artifact(str(submission_file), artifact_path="predictions")
        logger.success(
            f"Submission Kaggle generado: {submission_file} | {test_binary_preds.sum()} supervivientes predichos"
        )

        logger.success("Corrida v2 completada y registrada en MLflow.")
        logger.info(
            f"MLflow UI: http://172.17.212.149:5000/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}"
        )

    return {
        "model_name": model_name,
        "accuracy": acc,
        "f1_macro": f1,
        "roc_auc": auc,
        "optimal_threshold": optimal_threshold,
        "features_count": X_train.shape[1],
        "mlflow_run_id": run.info.run_id,
        "submission_file": str(submission_file),
    }


if __name__ == "__main__":
    train_and_evaluate()
