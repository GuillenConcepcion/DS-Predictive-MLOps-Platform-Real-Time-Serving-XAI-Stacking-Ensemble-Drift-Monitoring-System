from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, MissingIndicator, SimpleImputer
from sklearn.linear_model import BayesianRidge


def littles_mcar_test(
    df: pd.DataFrame, columns: list[str] | None = None, max_iter: int = 100, tol: float = 1e-4
) -> dict[str, Any]:
    """
    Roderick J. A. Little (1988) Chi-Square Test of Missing Completely at Random (MCAR).

    Hipótesis:
      H0: Los datos faltantes son Missing Completely at Random (MCAR).
      H1: Los datos NO son MCAR (Mecanismo MAR o MNAR).

    Parámetros:
      df: DataFrame con variables numéricas/continuas a evaluar.
      columns: Subconjunto de columnas numéricas (por defecto, todas las numéricas con nulos o varianza).
      max_iter: Máximo de iteraciones para el algoritmo Expectation-Maximization (EM).
      tol: Tolerancia de convergencia de parámetros mu y sigma.

    Retorna:
      Diccionario con estadístico d^2 (chi_square), grados de libertad (df), p-value,
      veredicto (is_mcar), patrones de ausencia y recomendación de imputación.
    """
    if columns is None:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        num_cols = [c for c in columns if c in df.columns and np.issubdtype(df[c].dtype, np.number)]

    if not num_cols:
        return {
            "is_mcar": True,
            "chi_square_stat": 0.0,
            "degrees_of_freedom": 0,
            "p_value": 1.0,
            "verdict": "Sin variables numéricas válidas para contrastar.",
            "recommended_strategy": "mean",
            "missing_patterns": [],
        }

    data = df[num_cols].values.astype(float)
    N, P = data.shape
    nan_mask = np.isnan(data)

    # Si no hay valores nulos en el conjunto
    if not nan_mask.any():
        return {
            "is_mcar": True,
            "chi_square_stat": 0.0,
            "degrees_of_freedom": 0,
            "p_value": 1.0,
            "verdict": "Dataset completo sin valores nulos (MCAR trivial).",
            "recommended_strategy": "passthrough",
            "missing_patterns": [{"pattern_id": 1, "pattern": [True] * P, "count": N, "pct": 100.0}],
        }

    # 1. Identificar patrones únicos de datos faltantes (False = observado, True = nulo)
    unique_patterns, pattern_inverse, pattern_counts = np.unique(
        nan_mask, axis=0, return_inverse=True, return_counts=True
    )

    # 2. Algoritmo EM para estimación Maximum Likelihood de mu y Sigma
    # Inicialización con medias y matriz de covarianza
    mu = np.nanmean(data, axis=0)
    mu = np.nan_to_num(mu, nan=0.0)

    # Rellenar temporalmente con media para calcular covarianza inicial
    temp_data = np.where(nan_mask, mu, data)
    sigma = np.cov(temp_data, rowvar=False)
    if sigma.ndim == 0:
        sigma = np.array([[float(sigma)]])

    # Regularización ridge mínima para estabilidad numérica
    sigma += np.eye(P) * 1e-6

    for _ in range(max_iter):
        mu_old = mu.copy()
        sum_y = np.zeros(P)
        sum_yy = np.zeros((P, P))

        for i in range(N):
            obs_idx = np.where(~nan_mask[i])[0]
            mis_idx = np.where(nan_mask[i])[0]
            y_i = data[i].copy()

            if len(mis_idx) == 0:
                sum_y += y_i
                sum_yy += np.outer(y_i, y_i)
            elif len(obs_idx) == 0:
                sum_y += mu
                sum_yy += sigma + np.outer(mu, mu)
            else:
                mu_obs = mu[obs_idx]
                mu_mis = mu[mis_idx]
                sigma_oo = sigma[np.ix_(obs_idx, obs_idx)]
                sigma_mm = sigma[np.ix_(mis_idx, mis_idx)]
                sigma_mo = sigma[np.ix_(mis_idx, obs_idx)]
                sigma_om = sigma[np.ix_(obs_idx, mis_idx)]

                inv_sigma_oo = np.linalg.pinv(sigma_oo)
                beta = sigma_mo @ inv_sigma_oo

                # E[y_mis | y_obs]
                e_mis = mu_mis + beta @ (y_i[obs_idx] - mu_obs)
                var_mis = sigma_mm - beta @ sigma_om

                y_filled = y_i.copy()
                y_filled[mis_idx] = e_mis

                sum_y += y_filled

                # Matriz E[y y^T | y_obs]
                yy_i = np.outer(y_filled, y_filled)
                yy_i[np.ix_(mis_idx, mis_idx)] += var_mis
                sum_yy += yy_i

        mu = sum_y / N
        sigma = (sum_yy / N) - np.outer(mu, mu)
        sigma = (sigma + sigma.T) / 2.0  # Simetría
        sigma += np.eye(P) * 1e-6  # Regularización

        if np.max(np.abs(mu - mu_old)) < tol:
            break

    # 3. Cálculo del estadístico d^2 de Little
    d2 = 0.0
    total_obs_vars = 0

    patterns_summary = []
    for p_idx, (pat, count) in enumerate(zip(unique_patterns, pattern_counts, strict=False)):
        obs_idx = np.where(~pat)[0]
        p_s = len(obs_idx)
        total_obs_vars += p_s

        pat_info = {
            "pattern_id": p_idx + 1,
            "observed_columns": [num_cols[k] for k in obs_idx],
            "missing_columns": [num_cols[k] for k in np.where(pat)[0]],
            "count": int(count),
            "percentage": round(float(count / N) * 100.0, 2),
        }
        patterns_summary.append(pat_info)

        if 0 < p_s < P:
            y_obs_pattern = data[pattern_inverse == p_idx][:, obs_idx]
            y_bar_s = np.mean(y_obs_pattern, axis=0)
            mu_s = mu[obs_idx]
            sigma_s = sigma[np.ix_(obs_idx, obs_idx)]

            diff = y_bar_s - mu_s
            try:
                inv_sigma_s = np.linalg.pinv(sigma_s)
                d2 += count * (diff.T @ inv_sigma_s @ diff)
            except np.linalg.LinAlgError:
                pass

    # Grados de libertad: df = sum(p_s) - P
    df_stat = max(0, total_obs_vars - P)

    if df_stat > 0:
        p_val = float(stats.chi2.sf(d2, df_stat))
    else:
        p_val = 1.0

    is_mcar = bool(p_val > 0.05)

    if is_mcar:
        verdict = f"Acepta H0 (p={p_val:.4f} > 0.05): Ausencia MCAR (Missing Completely at Random). La falta de datos es aleatoria e independiente."
        rec_strategy = "mean" if (nan_mask.sum() / (N * P)) < 0.05 else "knn"
    else:
        verdict = f"Rechaza H0 (p={p_val:.4e} <= 0.05): Ausencia NO-MCAR (Mecanismo MAR / MNAR). Existe dependencia multivariada en los patrones de ausencia."
        rec_strategy = "iterative_mice" if P > 3 else "knn"

    return {
        "is_mcar": is_mcar,
        "chi_square_stat": round(float(d2), 4),
        "degrees_of_freedom": int(df_stat),
        "p_value": round(float(p_val), 6),
        "significance_level": 0.05,
        "verdict": verdict,
        "evaluated_columns": num_cols,
        "total_records": N,
        "recommended_strategy": rec_strategy,
        "patterns_count": len(unique_patterns),
        "missing_patterns": patterns_summary,
    }


class AdvancedDataImputer(BaseEstimator, TransformerMixin):
    """
    Imputador avanzado con soporte para MCAR, MAR y MNAR, diagnóstico formal
    de Little's MCAR Test y trazabilidad de indicadores booleanos.
    """

    def __init__(self, strategy="knn", n_neighbors=5, add_indicator=True, random_state=42):
        self.strategy = strategy
        self.n_neighbors = n_neighbors
        self.add_indicator = add_indicator
        self.random_state = random_state
        self.imputer_ = None
        self.indicator_ = None
        self.feature_names_in_ = None
        self.mcar_diagnostics_: dict[str, Any] | None = None

    @staticmethod
    def test_mcar(df: pd.DataFrame, columns: list[str] | None = None) -> dict[str, Any]:
        """Ejecuta el test estadístico de Little (1988) sobre el DataFrame."""
        return littles_mcar_test(df, columns=columns)

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self.feature_names_in_ = list(X.columns)

        # Ejecutar diagnóstico de ausencia
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            self.mcar_diagnostics_ = littles_mcar_test(X, columns=num_cols)

        if self.add_indicator:
            self.indicator_ = MissingIndicator(features="missing-only")
            self.indicator_.fit(X)

        if self.strategy in ["mean", "median", "most_frequent", "constant"]:
            self.imputer_ = SimpleImputer(strategy=self.strategy)
            self.imputer_.fit(X)
        elif self.strategy == "knn":
            self.imputer_ = KNNImputer(n_neighbors=self.n_neighbors, weights="distance")
            self.imputer_.fit(X)
        elif self.strategy == "iterative_mice":
            self.imputer_ = IterativeImputer(estimator=BayesianRidge(), random_state=self.random_state)
            self.imputer_.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        else:
            X = X.copy()

        arr = self.imputer_.transform(X)
        imputed_df = pd.DataFrame(arr, columns=self.feature_names_in_, index=X.index)

        if self.add_indicator and self.indicator_ is not None:
            ind_arr = self.indicator_.transform(X)
            if ind_arr.shape[1] > 0:
                missing_cols = [f"{col}_is_na" for col in np.array(self.feature_names_in_)[self.indicator_.features_]]
                ind_df = pd.DataFrame(ind_arr, columns=missing_cols, index=X.index)
                imputed_df = pd.concat([imputed_df, ind_df], axis=1)

        return imputed_df

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
