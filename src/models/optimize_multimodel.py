"""
optimize_multimodel.py — Estudio Bayesiano Multi-Modelo con Optuna.

Busca simultaneamente el mejor algoritmo (GBM, LGBM, XGBoost) y sus hiperparametros
usando TPE Sampler + HyperbandPruner. Exporta best_params.json a models/ y registra
cada trial en MLflow experimento 'Titanic-Optuna-MultiModel'.
"""

import json
import os
from pathlib import Path

import lightgbm as lgb
import mlflow
import optuna
import xgboost as xgb
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.data.make_dataset import load_raw_data
from src.features.build_features import TitanicFeaturePipeline
from src.utils.logger import logger

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def build_model(trial, model_name: str):
    """Construye el clasificador con los hiperparametros sugeridos por Optuna."""
    if model_name == "GBM":
        return GradientBoostingClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 500, step=50),
            max_depth=trial.suggest_int("max_depth", 2, 6),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 30),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            random_state=42,
        )

    elif model_name == "LGBM":
        return lgb.LGBMClassifier(
            num_leaves=trial.suggest_int("num_leaves", 15, 127),
            n_estimators=trial.suggest_int("n_estimators", 100, 600, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 50),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            random_state=42,
            verbose=-1,
        )

    else:  # XGB
        return xgb.XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 400, step=50),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 8),
            gamma=trial.suggest_float("gamma", 0.0, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            random_state=42,
            eval_metric="logloss",
        )


def run_multi_model_optuna_study(n_trials: int = 100) -> dict:
    """
    Ejecuta el estudio bayesiano multi-modelo.

    Args:
        n_trials: Numero de trials de Optuna (default: 100).

    Returns:
        Diccionario con best_model_name, best_params y best_cv_auc.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://172.17.212.149:5000")
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = "Titanic-Optuna-MultiModel"
    mlflow.set_experiment(experiment_name)

    df_train, _ = load_raw_data()
    pipeline = TitanicFeaturePipeline()
    X_train, y_train = pipeline.fit_transform(df_train)

    logger.info(f"Features v2 generadas: {X_train.shape[1]}")
    logger.info(f"Iniciando estudio Optuna multi-modelo | {n_trials} trials | HyperbandPruner + TPE")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial):
        model_name = trial.suggest_categorical("model", ["GBM", "LGBM", "XGB"])
        model = build_model(trial, model_name)

        oof_probs = []
        oof_labels = []

        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr = X_train.iloc[train_idx]
            y_tr = y_train.iloc[train_idx]
            X_va = X_train.iloc[val_idx]
            y_va = y_train.iloc[val_idx]

            model.fit(X_tr, y_tr)
            probs = model.predict_proba(X_va)[:, 1]
            oof_probs.extend(probs)
            oof_labels.extend(y_va)

        cv_auc = roc_auc_score(oof_labels, oof_probs)
        return cv_auc

    sampler = TPESampler(seed=42, n_startup_trials=20)
    pruner = HyperbandPruner(min_resource=1, max_resource=n_trials, reduction_factor=3)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name="titanic_multimodel_v2",
    )

    # Silenciar logs verbosos de Optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    logger.info(f"Ejecutando {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_trial = study.best_trial
    best_model_name = best_trial.params["model"]
    best_params = {k: v for k, v in best_trial.params.items() if k != "model"}
    best_cv_auc = study.best_value

    logger.success("=" * 70)
    logger.success(" ESTUDIO OPTUNA COMPLETADO")
    logger.success(f" Mejor Modelo:  {best_model_name}")
    logger.success(f" Mejor CV AUC:  {best_cv_auc:.4f} ({best_cv_auc * 100:.2f}%)")
    logger.success(f" Hiperparametros: {best_params}")
    logger.success("=" * 70)

    # Exportar best_params.json para consumo en train_and_evaluate.py
    output = {
        "model_name": best_model_name,
        "cv_roc_auc": round(best_cv_auc, 6),
        "hyperparameters": best_params,
        "n_trials": n_trials,
        "features_count": X_train.shape[1],
    }
    best_params_path = MODELS_DIR / "best_params.json"
    with open(best_params_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.success(f"best_params.json exportado a: {best_params_path}")

    # Registrar el mejor trial en MLflow
    with mlflow.start_run(run_name=f"optuna_best_{best_model_name}"):
        mlflow.log_param("model_name", best_model_name)
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("features_count", X_train.shape[1])
        mlflow.log_params(best_params)
        mlflow.log_metric("best_cv_roc_auc", best_cv_auc)
        mlflow.set_tag("author", "Guillen Concepcion")
        mlflow.set_tag("optimization_type", "multi_model_optuna_v2")
        mlflow.log_artifact(str(best_params_path), artifact_path="hyperparameters")

    return output


if __name__ == "__main__":
    result = run_multi_model_optuna_study(n_trials=100)
    logger.info(f"Resultado exportado: {result}")
