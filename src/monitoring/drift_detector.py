import os
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as stats
from loguru import logger


def calculate_psi(reference: np.ndarray, current: np.ndarray, num_bins: int = 10, epsilon: float = 1e-4) -> float:
    """
    Calcula el Population Stability Index (PSI) entre dos distribuciones continuas.
    Interpretación estándar en la industria bancaria y MLOps:
      - PSI < 0.10: Distribución estable (Sin cambio significativo / No drift).
      - 0.10 <= PSI < 0.20: Cambio moderado (Monitoreo preventivo / Moderate drift).
      - PSI >= 0.20: Desvío significativo (Alerta de reentrenamiento / Significant drift).
    """
    ref_clean = reference[~np.isnan(reference)]
    curr_clean = current[~np.isnan(current)]

    if len(ref_clean) < 10 or len(curr_clean) < 10:
        return 0.0

    # Adaptar cantidad de bins al tamaño muestral para evitar ruido en muestras pequeñas
    effective_bins = min(num_bins, max(3, len(ref_clean) // 25))
    quantiles = np.linspace(0, 100, effective_bins + 1)
    bin_edges = np.percentile(ref_clean, quantiles)
    bin_edges = np.unique(bin_edges)

    if len(bin_edges) < 2:
        return 0.0

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref_clean, bins=bin_edges)
    curr_counts, _ = np.histogram(curr_clean, bins=bin_edges)

    ref_pct = (ref_counts / len(ref_clean)) + epsilon
    curr_pct = (curr_counts / len(curr_clean)) + epsilon

    ref_pct /= ref_pct.sum()
    curr_pct /= curr_pct.sum()

    psi_value = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
    return float(np.clip(psi_value, 0.0, 10.0))


def calculate_categorical_drift(
    reference_series: pd.Series, current_series: pd.Series, alpha: float = 0.05
) -> dict[str, Any]:
    """
    Calcula drift en variables categóricas mediante Chi-Square Goodness-of-Fit
    y Total Variation Distance (TVD).
    """
    ref_counts = reference_series.value_counts(dropna=False)
    curr_counts = current_series.value_counts(dropna=False)

    all_categories = sorted(list(set(ref_counts.index).union(set(curr_counts.index))), key=lambda x: str(x))

    ref_freq = np.array([ref_counts.get(cat, 0) for cat in all_categories], dtype=float)
    curr_freq = np.array([curr_counts.get(cat, 0) for cat in all_categories], dtype=float)

    ref_prob = (ref_freq / ref_freq.sum()) if ref_freq.sum() > 0 else np.ones(len(all_categories)) / len(all_categories)
    curr_prob = (
        (curr_freq / curr_freq.sum()) if curr_freq.sum() > 0 else np.ones(len(all_categories)) / len(all_categories)
    )

    # Total Variation Distance: TVD = 0.5 * sum(|P - Q|) in [0, 1]
    tvd = float(0.5 * np.sum(np.abs(curr_prob - ref_prob)))

    # Chi-Square Test (con frecuencias esperadas ajustadas)
    expected_freq = ref_prob * curr_freq.sum()
    valid_idx = expected_freq > 0
    if valid_idx.sum() > 1 and curr_freq.sum() >= 10:
        stat, p_val = stats.chisquare(curr_freq[valid_idx], f_exp=expected_freq[valid_idx])
        p_value = float(p_val)
        chi2_stat = float(stat)
    else:
        p_value = 1.0
        chi2_stat = 0.0

    is_drift = bool((p_value < alpha and tvd >= 0.10) or tvd >= 0.20)
    return {
        "test_name": "Chi-Square & TVD",
        "statistic": round(chi2_stat, 4),
        "tvd": round(tvd, 4),
        "p_value": round(p_value, 6),
        "threshold": alpha,
        "is_drift": is_drift,
        "drift_score": round(tvd, 4),
    }


def calculate_numerical_drift(reference: np.ndarray, current: np.ndarray, alpha: float = 0.05) -> dict[str, Any]:
    """
    Calcula drift en variables numéricas mediante:
      1. Kolmogorov-Smirnov Test (KS 2-sample)
      2. Wasserstein Distance (Magnitud de shift)
      3. Population Stability Index (PSI)
    """
    ref_clean = reference[~np.isnan(reference)]
    curr_clean = current[~np.isnan(current)]

    if len(ref_clean) < 5 or len(curr_clean) < 5:
        return {
            "test_name": "KS-Test & PSI",
            "ks_statistic": 0.0,
            "p_value": 1.0,
            "wasserstein_distance": 0.0,
            "psi": 0.0,
            "is_drift": False,
            "drift_score": 0.0,
        }

    # KS Test
    ks_res = stats.ks_2samp(ref_clean, curr_clean)
    ks_stat = float(ks_res.statistic)
    p_val = float(ks_res.pvalue)

    # Wasserstein Distance
    wasserstein_dist = float(stats.wasserstein_distance(ref_clean, curr_clean))

    # PSI
    psi_val = calculate_psi(ref_clean, curr_clean)

    # Criterio estadístico riguroso de Drift:
    # Se detecta drift si el p-value de Kolmogorov-Smirnov es estadísticamente significativo (p < alpha)
    # y el PSI indica al menos un cambio perceptible (PSI >= 0.05), o si el PSI excede 0.25 con significancia marginal (p < 0.10).
    is_drift = bool((p_val < alpha and psi_val >= 0.05) or (p_val < 0.10 and psi_val >= 0.25))

    return {
        "test_name": "Kolmogorov-Smirnov & PSI",
        "ks_statistic": round(ks_stat, 4),
        "p_value": round(p_val, 6),
        "wasserstein_distance": round(wasserstein_dist, 4),
        "psi": round(psi_val, 4),
        "threshold": alpha,
        "is_drift": is_drift,
        "drift_score": round(psi_val, 4),
    }


class DataDriftDetector:
    """
    Motor integral de auditoría y monitoreo continuo de Data & Concept Drift.
    """

    def __init__(
        self,
        reference_data: pd.DataFrame | None = None,
        drift_share_threshold: float = 0.33,
        alpha: float = 0.05,
        reference_probabilities: np.ndarray | None = None,
        reference_predictions: np.ndarray | None = None,
    ):
        self.reference_data = reference_data.copy() if reference_data is not None else pd.DataFrame()
        self.drift_share_threshold = drift_share_threshold
        self.alpha = alpha
        self.reference_probabilities_: np.ndarray | None = (
            np.asarray(reference_probabilities, dtype=float) if reference_probabilities is not None else None
        )
        self.reference_predictions_: np.ndarray | None = (
            np.asarray(reference_predictions, dtype=int) if reference_predictions is not None else None
        )

    def set_reference_predictions(self, reference_probabilities: np.ndarray, reference_predictions: np.ndarray):
        """Establece las predicciones de referencia generadas sobre el conjunto de entrenamiento."""
        self.reference_probabilities_ = np.asarray(reference_probabilities, dtype=float)
        self.reference_predictions_ = np.asarray(reference_predictions, dtype=int)

    def calculate_prediction_drift(
        self, current_probabilities: np.ndarray, current_predictions: np.ndarray, alpha: float = 0.05
    ) -> dict[str, Any]:
        """
        Calcula el Prediction Drift evaluando:
          1. Probabilidad Continua: PSI, KS 2-Sample Test, Distancia de Wasserstein.
          2. Decisión Binaria: Chi-Square Goodness-of-Fit y TVD sobre la tasa de supervivencia predicha.
        """
        if self.reference_probabilities_ is None or len(self.reference_probabilities_) < 10:
            return {
                "status": "insufficient_reference_data",
                "is_drift_detected": False,
                "overall_status": "NO_BASELINE",
            }

        curr_probs = np.asarray(current_probabilities, dtype=float)
        curr_preds = np.asarray(current_predictions, dtype=int)

        if len(curr_probs) < 5:
            return {
                "sample_size": len(curr_probs),
                "baseline_samples": len(self.reference_probabilities_),
                "probability_drift": {"status": "insufficient_sample", "psi": 0.0},
                "decision_drift": {"status": "insufficient_sample", "tvd": 0.0},
                "overall_status": "COLLECTING_DATA",
                "is_drift_detected": False,
                "summary": f"Muestra insuficiente ({len(curr_probs)}/5 registros requeridos para inferencia estadística).",
            }

        # 1. Continuous Probability Drift
        prob_drift = calculate_numerical_drift(self.reference_probabilities_, curr_probs, alpha=alpha)

        # 2. Binary Decision Drift
        ref_series = pd.Series(self.reference_predictions_)
        curr_series = pd.Series(curr_preds)
        decision_drift = calculate_categorical_drift(ref_series, curr_series, alpha=alpha)

        # 3. Criterio de Severidad
        psi = prob_drift["psi"]
        p_val = prob_drift["p_value"]
        is_prob_drift = prob_drift["is_drift"]
        is_decision_drift = decision_drift["is_drift"]

        if psi >= 0.25 or (is_prob_drift and is_decision_drift):
            overall_status = "CRITICAL_DRIFT"
            is_drift_detected = True
            summary = f"Alerta Crítica: Prediction Drift detectado (PSI={psi:.4f}, KS p-val={p_val:.4f}). Requiere inspección o reentrenamiento."
        elif psi >= 0.10 or is_prob_drift or is_decision_drift:
            overall_status = "MODERATE_DRIFT"
            is_drift_detected = True
            summary = f"Advertencia: Desplazamiento moderado en las predicciones (PSI={psi:.4f}). Aumentar frecuencia de monitoreo."
        else:
            overall_status = "STABLE"
            is_drift_detected = False
            summary = f"Distribución de predicciones estable (PSI={psi:.4f}, sin drift significativo)."

        return {
            "sample_size": len(curr_probs),
            "baseline_samples": len(self.reference_probabilities_),
            "probability_drift": prob_drift,
            "decision_drift": decision_drift,
            "overall_status": overall_status,
            "is_drift_detected": is_drift_detected,
            "summary": summary,
        }

    def detect_drift(self, current_data: pd.DataFrame, features: list[str] | None = None) -> dict[str, Any]:
        """
        Ejecuta el análisis de Data Drift entre el baseline de referencia y los datos entrantes.
        """
        if features is None:
            common_cols = [c for c in self.reference_data.columns if c in current_data.columns]
        else:
            common_cols = [c for c in features if c in self.reference_data.columns and c in current_data.columns]

        drift_by_column = {}
        drifting_features_count = 0

        for col in common_cols:
            ref_col = self.reference_data[col]
            curr_col = current_data[col]

            if np.issubdtype(ref_col.dtype, np.number):
                col_metric = calculate_numerical_drift(
                    ref_col.values.astype(float), curr_col.values.astype(float), alpha=self.alpha
                )
            else:
                col_metric = calculate_categorical_drift(ref_col, curr_col, alpha=self.alpha)

            col_metric["column_type"] = "numerical" if np.issubdtype(ref_col.dtype, np.number) else "categorical"
            col_metric["reference_missing_pct"] = round(float(ref_col.isnull().mean() * 100), 2)
            col_metric["current_missing_pct"] = round(float(curr_col.isnull().mean() * 100), 2)

            if col_metric["is_drift"]:
                drifting_features_count += 1

            drift_by_column[col] = col_metric

        total_features = len(common_cols)
        drift_share = float(drifting_features_count / total_features) if total_features > 0 else 0.0
        dataset_drift = bool(drift_share >= self.drift_share_threshold)

        return {
            "dataset_drift": dataset_drift,
            "drift_share": round(drift_share, 4),
            "drift_share_threshold": self.drift_share_threshold,
            "number_of_features": total_features,
            "number_of_drifted_features": drifting_features_count,
            "reference_rows": len(self.reference_data),
            "current_rows": len(current_data),
            "alpha": self.alpha,
            "drift_by_column": drift_by_column,
        }

    def detect_prediction_drift(
        self,
        reference_preds: np.ndarray,
        current_preds: np.ndarray,
        reference_probs: np.ndarray | None = None,
        current_probs: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """
        Monitorea el Prediction & Concept Drift comparando la distribución de
        probabilidades y clases generadas por el modelo en producción.
        """
        # Drift en clases predichas
        pred_ref_s = pd.Series(reference_preds)
        pred_curr_s = pd.Series(current_preds)
        class_drift = calculate_categorical_drift(pred_ref_s, pred_curr_s, alpha=self.alpha)

        # Drift en distribución de probabilidades continuas
        prob_drift = None
        if reference_probs is not None and current_probs is not None:
            prob_drift = calculate_numerical_drift(
                np.array(reference_probs, dtype=float), np.array(current_probs, dtype=float), alpha=self.alpha
            )

        is_concept_drift = bool(class_drift["is_drift"] or (prob_drift["is_drift"] if prob_drift else False))

        return {
            "prediction_drift": is_concept_drift,
            "class_drift": class_drift,
            "probability_drift": prob_drift,
            "reference_positive_rate": round(float(np.mean(reference_preds)), 4),
            "current_positive_rate": round(float(np.mean(current_preds)), 4),
        }

    def generate_html_report(
        self,
        current_data: pd.DataFrame,
        output_path: str = "reports/drift_report.html",
        features: list[str] | None = None,
    ) -> str:
        """
        Genera un informe visual interactivo y autónomo en formato HTML5.
        """
        report_data = self.detect_drift(current_data, features=features)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        badge_class = "badge-danger" if report_data["dataset_drift"] else "badge-success"
        badge_text = "DRIFT DETECTADO" if report_data["dataset_drift"] else "DISTRIBUCIÓN ESTABLE"

        rows_html = ""
        for col, res in report_data["drift_by_column"].items():
            status_badge = (
                '<span class="status-pill pill-drift">DRIFT</span>'
                if res["is_drift"]
                else '<span class="status-pill pill-ok">OK</span>'
            )
            p_val_display = f"{res['p_value']:.4e}" if res["p_value"] < 0.001 else f"{res['p_value']:.4f}"
            score_metric = (
                f"PSI: {res.get('psi', 'N/A')}"
                if res["column_type"] == "numerical"
                else f"TVD: {res.get('tvd', 'N/A')}"
            )

            rows_html += f"""
            <tr>
                <td><strong>{col}</strong></td>
                <td><span class="type-tag">{res["column_type"]}</span></td>
                <td>{res["test_name"]}</td>
                <td>{score_metric}</td>
                <td>{p_val_display}</td>
                <td>{status_badge}</td>
            </tr>
            """

        html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Odysseus MLOps - Data Drift Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 30px; }}
        .container {{ max-width: 1100px; margin: 0 auto; background: #1e293b; padding: 30px; border-radius: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); border: 1px solid #334155; }}
        h1 {{ color: #38bdf8; margin-top: 0; display: flex; justify-content: space-between; align-items: center; }}
        .badge {{ padding: 6px 14px; border-radius: 20px; font-size: 14px; font-weight: bold; }}
        .badge-success {{ background: #10b981; color: #fff; }}
        .badge-danger {{ background: #ef4444; color: #fff; }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 25px 0; }}
        .card {{ background: #0f172a; padding: 20px; border-radius: 8px; border: 1px solid #334155; text-align: center; }}
        .card-value {{ font-size: 28px; font-weight: bold; color: #38bdf8; margin-top: 5px; }}
        .card-label {{ font-size: 13px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th, td {{ padding: 12px 16px; text-align: left; border-bottom: 1px solid #334155; }}
        th {{ background: #0f172a; color: #94a3b8; font-weight: 600; text-transform: uppercase; font-size: 12px; }}
        tr:hover {{ background: #24344d; }}
        .status-pill {{ padding: 4px 10px; border-radius: 12px; font-size: 11px; font-weight: bold; display: inline-block; }}
        .pill-ok {{ background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #10b981; }}
        .pill-drift {{ background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444; }}
        .type-tag {{ background: #334155; padding: 2px 8px; border-radius: 4px; font-size: 11px; color: #cbd5e1; }}
        .footer {{ margin-top: 30px; font-size: 12px; color: #64748b; text-align: center; border-top: 1px solid #334155; padding-top: 15px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>
            <span>🚀 Odysseus AI - Auditoría de Data Drift</span>
            <span class="badge {badge_class}">{badge_text}</span>
        </h1>
        <p style="color: #94a3b8;">Monitoreo estadístico de distribuciones con Kolmogorov-Smirnov, Population Stability Index (PSI) y Chi-Cuadrado.</p>

        <div class="summary-grid">
            <div class="card">
                <div class="card-label">Features en Drift</div>
                <div class="card-value">{report_data["number_of_drifted_features"]} / {report_data["number_of_features"]}</div>
            </div>
            <div class="card">
                <div class="card-label">% Share de Drift</div>
                <div class="card-value">{report_data["drift_share"] * 100:.1f}%</div>
            </div>
            <div class="card">
                <div class="card-label">Registros Referencia</div>
                <div class="card-value">{report_data["reference_rows"]}</div>
            </div>
            <div class="card">
                <div class="card-label">Registros Actuales</div>
                <div class="card-value">{report_data["current_rows"]}</div>
            </div>
        </div>

        <table>
            <thead>
                <tr>
                    <th>Columna</th>
                    <th>Tipo</th>
                    <th>Prueba Estadística</th>
                    <th>Métrica de Estabilidad</th>
                    <th>P-Valor</th>
                    <th>Estado</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>

        <div class="footer">
            Generado automáticamente por Odysseus AI MLOps Platform | Lead Architect: Guillén Concepción
        </div>
    </div>
</body>
</html>
"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.success(f"Reporte HTML de Data Drift generado exitosamente en: {output_path}")
        return output_path
