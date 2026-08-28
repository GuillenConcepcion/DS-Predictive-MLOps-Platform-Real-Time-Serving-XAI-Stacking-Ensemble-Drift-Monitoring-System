"""
target_encoder.py — Bayesian Target Encoding con m-estimate smoothing y OOF Cross-Validation.

Garantiza cero Data Leakage mediante particionado estratificado interno en entrenamiento
y mapeo determinístico a posteriori con fallback de prior global para inferencia en producción.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold

from src.utils.logger import logger


class BayesianTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Codificador Bayesiano de Target (m-estimate smoothing).

    Fórmula:
        S_k = (count_k * mean_k + m * global_mean) / (count_k + m)

    Parámetros:
        cols: Lista de columnas categóricas a codificar.
        m: Factor de suavizado (peso a priori). Mayor 'm' = mayor regularización hacia global_mean.
        cv: Número de splits para Out-of-Fold en fit_transform.
        random_state: Semilla aleatoria para particionado estratificado.
    """

    def __init__(
        self,
        cols: list[str] | None = None,
        m: float = 10.0,
        cv: int = 5,
        random_state: int = 42,
    ):
        self.cols = cols or []
        self.m = float(m)
        self.cv = int(cv)
        self.random_state = int(random_state)
        self.global_mean_: float = 0.0
        self.encoding_maps_: dict[str, dict[str, float]] = {}

    def _compute_bayes_map(self, series: pd.Series, target: pd.Series, global_mean: float) -> dict[str, float]:
        """Calcula el diccionario de codificación Bayesiana para una columna dada."""
        stats = target.groupby(series, observed=False).agg(["count", "mean"])
        bayes_encoded = ((stats["count"] * stats["mean"]) + (self.m * global_mean)) / (stats["count"] + self.m)
        return bayes_encoded.to_dict()

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Ajusta los diccionarios de codificación globales sobre todo el conjunto de datos.
        Utilizado para inferencia en test / producción.
        """
        self.global_mean_ = float(y.mean())
        self.encoding_maps_ = {}

        for col in self.cols:
            if col in X.columns:
                series_str = X[col].astype(str)
                self.encoding_maps_[col] = self._compute_bayes_map(series_str, y, self.global_mean_)

        logger.debug(
            f"BayesianTargetEncoder ajustado en {len(self.encoding_maps_)} columnas (global_mean={self.global_mean_:.4f}, m={self.m})"
        )
        return self

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Calcula las codificaciones Out-of-Fold (OOF) para evitar sobreajuste y data leakage,
        y ajusta simultáneamente los mapas globales para inferencias posteriores.
        """
        self.fit(X, y)
        df_encoded = pd.DataFrame(index=X.index)

        # Preparar estructura de salida
        for col in self.cols:
            if col in X.columns:
                df_encoded[f"{col}_TE"] = np.nan

        # Manejo adaptativo de CV para muestras pequeñas (e.g., tests con n_samples < 5)
        class_counts = y.value_counts()
        min_class_samples = int(class_counts.min()) if len(class_counts) > 1 else 0
        effective_cv = min(self.cv, len(X), min_class_samples)

        if effective_cv < 2:
            # Fallback determinístico con mapa global si no es posible particionar
            for col in self.cols:
                if col in X.columns and col in self.encoding_maps_:
                    series_str = X[col].astype(str)
                    df_encoded[f"{col}_TE"] = (
                        series_str.map(self.encoding_maps_[col]).fillna(self.global_mean_).astype(float)
                    )
            return df_encoded

        # Generar codificaciones OOF con Stratified CV
        skf = StratifiedKFold(n_splits=effective_cv, shuffle=True, random_state=self.random_state)

        for train_idx, val_idx in skf.split(X, y):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_va = X.iloc[val_idx]
            fold_global_mean = float(y_tr.mean())

            for col in self.cols:
                if col in X.columns:
                    tr_series = X_tr[col].astype(str)
                    va_series = X_va[col].astype(str)

                    fold_map = self._compute_bayes_map(tr_series, y_tr, fold_global_mean)
                    encoded_val = va_series.map(fold_map).fillna(fold_global_mean).astype(float)
                    df_encoded.iloc[val_idx, df_encoded.columns.get_loc(f"{col}_TE")] = encoded_val

        # Fallback de seguridad si quedara algún NaN
        for col in self.cols:
            target_col = f"{col}_TE"
            if target_col in df_encoded.columns:
                df_encoded[target_col] = df_encoded[target_col].fillna(self.global_mean_)

        logger.success(f"Target Encoding Bayesiano OOF generado para {len(self.cols)} variables.")
        return df_encoded

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforma datos nuevos / en producción utilizando los mapas globales aprendidos en fit().
        Las categorías no vistas se imputan con la tasa de supervivencia global (prior).
        """
        df_encoded = pd.DataFrame(index=X.index)

        for col in self.cols:
            target_col = f"{col}_TE"
            if col in X.columns and col in self.encoding_maps_:
                series_str = X[col].astype(str)
                bayes_map = self.encoding_maps_[col]
                encoded_series = series_str.map(bayes_map).fillna(self.global_mean_).astype(float)
                df_encoded[target_col] = encoded_series
            else:
                df_encoded[target_col] = self.global_mean_

        return df_encoded
