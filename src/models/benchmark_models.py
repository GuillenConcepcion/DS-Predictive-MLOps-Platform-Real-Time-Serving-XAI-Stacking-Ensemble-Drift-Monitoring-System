"""
Odysseus AI - Multi-Model Benchmark & Ensemble Optimization Suite v2.
Evaluates RandomForest, GradientBoosting, ExtraTrees, LightGBM, XGBoost,
Voting (Soft Blending) and StackingEnsemble v2 (4 base learners + LR/XGB meta-learner).
All models trained on Feature Engineering v2 (32+ features).
Logs all models, metrics, and comparisons to MLflow and reports artifacts.
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.data.make_dataset import load_raw_data
from src.features.build_features import TitanicFeaturePipeline
from src.utils.logger import logger


def run_multi_model_benchmark():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://172.17.212.149:5000")
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = "Titanic-MultiModel-Benchmark"
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow Benchmark conectado a: {tracking_uri} | Experimento: {experiment_name}")

    df_train, df_test = load_raw_data()
    pipeline = TitanicFeaturePipeline()
    X_train, y_train = pipeline.fit_transform(df_train)
    X_test = pipeline.transform(df_test)

    # Definición de Algoritmos Candidatos con Hiperparámetros Optimizados
    models_dict = {
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),
        "GradientBoostingClassifier": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.04,
            max_depth=4,
            subsample=0.85,
            min_samples_split=4,
            random_state=42,
        ),
        "ExtraTreesClassifier": ExtraTreesClassifier(
            n_estimators=250,
            max_depth=7,
            min_samples_split=3,
            min_samples_leaf=2,
            max_features="sqrt",
            bootstrap=True,
            random_state=42,
            n_jobs=-1,
        ),
        "LightGBMClassifier": lgb.LGBMClassifier(
            n_estimators=180,
            learning_rate=0.03,
            max_depth=4,
            num_leaves=15,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            verbose=-1,
        ),
        "XGBoostClassifier": xgb.XGBClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=2,
            gamma=0.1,
            random_state=42,
            eval_metric="logloss",
        ),
    }

    # ---- Voting Ensemble v2 — pesos basados en CV ROC-AUC del benchmark previo ----
    voting_ensemble = VotingClassifier(
        estimators=[
            ("gb", models_dict["GradientBoostingClassifier"]),  # mejor CV AUC: 88.52
            ("xgb", models_dict["XGBoostClassifier"]),  # CV AUC: 88.23
            ("lgb", models_dict["LightGBMClassifier"]),  # CV AUC: 87.80
            ("et", models_dict["ExtraTreesClassifier"]),  # CV AUC: 87.21
            ("rf", models_dict["RandomForestClassifier"]),  # CV AUC: 87.19
        ],
        voting="soft",
        weights=[1.5, 1.4, 1.3, 1.1, 1.1],  # ponderado por CV AUC del benchmark
    )
    models_dict["VotingEnsemble (Soft Blending)"] = voting_ensemble

    # ---- StackingEnsemble v2 — meta-learner LR optimizado ----
    # Base learners: 4 algoritmos diversificados (evita redundancia RF+ExtraTrees)
    stacking_ensemble = StackingClassifier(
        estimators=[
            ("gb", models_dict["GradientBoostingClassifier"]),
            ("lgb", models_dict["LightGBMClassifier"]),
            ("xgb", models_dict["XGBoostClassifier"]),
            ("et", models_dict["ExtraTreesClassifier"]),
        ],
        final_estimator=LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42),
        cv=5,
        stack_method="predict_proba",
        n_jobs=-1,
    )
    models_dict["StackingEnsemble (Meta-Learner)"] = stacking_ensemble

    # ---- StackingEnsemble v2b — meta-learner XGB shallow ----
    stacking_xgb_meta = StackingClassifier(
        estimators=[
            ("gb", models_dict["GradientBoostingClassifier"]),
            ("lgb", models_dict["LightGBMClassifier"]),
            ("xgb", models_dict["XGBoostClassifier"]),
            ("et", models_dict["ExtraTreesClassifier"]),
        ],
        final_estimator=xgb.XGBClassifier(
            n_estimators=50,
            max_depth=2,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
            eval_metric="logloss",
        ),
        cv=5,
        stack_method="predict_proba",
        n_jobs=-1,
    )
    models_dict["StackingEnsemble (XGB Meta-Learner)"] = stacking_xgb_meta

    # --------------------------------------------------------------------------
    # Validación Cruzada Estratificada 5-Fold para cada Modelo
    # --------------------------------------------------------------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    benchmark_results = []

    best_model_name = None
    best_roc_auc = 0.0
    best_fitted_model = None

    for model_name, model in models_dict.items():
        logger.info(f"Evaluando algoritmo: {model_name}...")
        oof_probs = np.zeros(len(X_train))

        with mlflow.start_run(run_name=f"benchmark_{model_name}"):
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("cv_folds", 5)

            for _fold, (tr_idx, va_idx) in enumerate(cv.split(X_train, y_train)):
                X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
                X_va = X_train.iloc[va_idx]

                model.fit(X_tr, y_tr)
                val_probs = model.predict_proba(X_va)[:, 1]
                oof_probs[va_idx] = val_probs

            oof_preds = (oof_probs >= 0.5).astype(int)
            cv_acc = accuracy_score(y_train, oof_preds)
            cv_f1 = f1_score(y_train, oof_preds, average="macro")
            cv_auc = roc_auc_score(y_train, oof_probs)

            # Entrenar sobre todo el dataset de entrenamiento para medir ajuste
            model.fit(X_train, y_train)
            train_probs = model.predict_proba(X_train)[:, 1]
            train_acc = accuracy_score(y_train, (train_probs >= 0.5).astype(int))
            train_auc = roc_auc_score(y_train, train_probs)

            mlflow.log_metric("cv_accuracy", cv_acc)
            mlflow.log_metric("cv_f1_macro", cv_f1)
            mlflow.log_metric("cv_roc_auc", cv_auc)
            mlflow.log_metric("train_accuracy", train_acc)
            mlflow.log_metric("train_roc_auc", train_auc)

            logger.info(f" -> {model_name} | CV Acc: {cv_acc:.4f} | CV AUC: {cv_auc:.4f} | Train AUC: {train_auc:.4f}")

            benchmark_results.append(
                {
                    "Algorithm": model_name,
                    "CV Accuracy": round(cv_acc * 100.0, 2),
                    "CV F1-Macro": round(cv_f1 * 100.0, 2),
                    "CV ROC-AUC": round(cv_auc * 100.0, 2),
                    "Train Accuracy": round(train_acc * 100.0, 2),
                    "Train ROC-AUC": round(train_auc * 100.0, 2),
                }
            )

            if cv_auc > best_roc_auc:
                best_roc_auc = cv_auc
                best_model_name = model_name
                best_fitted_model = model

    # --------------------------------------------------------------------------
    # Crear Reporte Tabular y Visual
    # --------------------------------------------------------------------------
    df_results = pd.DataFrame(benchmark_results).sort_values(by="CV ROC-AUC", ascending=False)

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_report_path = reports_dir / "multi_model_benchmark_results.csv"
    df_results.to_csv(csv_report_path, index=False)

    # Gráfico de Comparación de Modelos
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    df_plot = pd.melt(
        df_results,
        id_vars=["Algorithm"],
        value_vars=["CV Accuracy", "CV ROC-AUC", "Train ROC-AUC"],
        var_name="Metric",
        value_name="Score (%)",
    )

    palette = {"CV Accuracy": "#38bdf8", "CV ROC-AUC": "#2563eb", "Train ROC-AUC": "#10b981"}
    sns.barplot(data=df_plot, x="Score (%)", y="Algorithm", hue="Metric", palette=palette, ax=ax)

    ax.set_title(
        "Benchmarking Multimodelo: RandomForest, GradientBoosting, ExtraTrees, XGBoost & Ensembles", fontsize=13, pad=15
    )
    ax.set_xlim(75, 100)
    ax.set_xlabel("Puntuación (%)")
    ax.set_ylabel("")
    plt.legend(loc="lower right")
    plt.tight_layout()

    plot_path = reports_dir / "model_benchmark_comparison.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()

    # Generar predicciones con el mejor modelo seleccionado
    test_preds = (best_fitted_model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
    submission_df = pd.DataFrame({"PassengerId": df_test["PassengerId"], "Survived": test_preds})
    sub_path = Path("data/processed/titanic_best_ensemble_submission.csv")
    submission_df.to_csv(sub_path, index=False)

    logger.success("=========================================================================")
    logger.success(f" 🏆 MEJOR MODELO SELECCIONADO: {best_model_name}")
    logger.success(f" -> CV ROC-AUC: {best_roc_auc * 100.0:.2f}% | Predicciones guardadas en: {sub_path}")
    logger.success("=========================================================================")

    return {
        "best_model": best_model_name,
        "best_roc_auc": round(best_roc_auc * 100.0, 2),
        "results_table": df_results.to_dict(orient="records"),
        "csv_path": str(csv_report_path),
        "plot_path": str(plot_path),
        "submission_path": str(sub_path),
    }


if __name__ == "__main__":
    run_multi_model_benchmark()
