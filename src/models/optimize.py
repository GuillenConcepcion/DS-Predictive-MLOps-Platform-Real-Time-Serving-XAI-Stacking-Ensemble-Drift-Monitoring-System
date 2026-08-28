import optuna
import xgboost as xgb
from optuna.pruners import MedianPruner
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.data.make_dataset import load_raw_data
from src.features.build_features import TitanicFeaturePipeline
from src.utils.logger import logger


def run_optuna_study(n_trials: int = 40):
    df_train, _ = load_raw_data()
    pipeline = TitanicFeaturePipeline()
    X_train, y_train = pipeline.fit_transform(df_train)

    logger.info(f"Iniciando estudio bayesiano con Optuna ({n_trials} trials)...")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=25),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 6),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "random_state": 42,
            "eval_metric": "logloss",
        }

        clf = xgb.XGBClassifier(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="roc_auc")
        return scores.mean()

    study = optuna.create_study(direction="maximize", pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5))
    study.optimize(objective, n_trials=n_trials)

    logger.success(f"Estudio completado. Mejor ROC-AUC: {study.best_value:.4f}")
    logger.info(f"Hiperparámetros óptimos: {study.best_params}")
    return study.best_params


if __name__ == "__main__":
    run_optuna_study()
