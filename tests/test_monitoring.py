import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.monitoring.drift_detector import (
    DataDriftDetector,
    calculate_categorical_drift,
    calculate_numerical_drift,
    calculate_psi,
)
from src.serving.app import app

client = TestClient(app)


def test_psi_identical_distributions():
    np.random.seed(42)
    ref = np.random.normal(100, 15, 1000)
    curr = np.random.normal(100, 15, 1000)
    psi = calculate_psi(ref, curr)
    assert psi < 0.10, f"El PSI para distribuciones idénticas debería ser < 0.10, obtenido: {psi}"


def test_psi_drifted_distribution():
    np.random.seed(42)
    ref = np.random.normal(100, 15, 1000)
    curr = np.random.normal(140, 25, 1000)  # Desvío severo
    psi = calculate_psi(ref, curr)
    assert psi >= 0.20, f"El PSI para distribución desfasada debería ser >= 0.20, obtenido: {psi}"


def test_numerical_drift_detection():
    np.random.seed(42)
    ref = np.random.exponential(10, 500)
    curr_stable = np.random.exponential(10, 500)
    curr_drift = np.random.exponential(50, 500)

    res_stable = calculate_numerical_drift(ref, curr_stable)
    assert res_stable["is_drift"] is False
    assert res_stable["p_value"] > 0.05

    res_drift = calculate_numerical_drift(ref, curr_drift)
    assert res_drift["is_drift"] is True
    assert res_drift["p_value"] < 0.05


def test_categorical_drift_detection():
    ref_s = pd.Series(["S"] * 700 + ["C"] * 200 + ["Q"] * 100)
    curr_s_stable = pd.Series(["S"] * 70 + ["C"] * 20 + ["Q"] * 10)
    curr_s_drift = pd.Series(["S"] * 10 + ["C"] * 80 + ["Q"] * 10)

    res_stable = calculate_categorical_drift(ref_s, curr_s_stable)
    assert res_stable["is_drift"] is False

    res_drift = calculate_categorical_drift(ref_s, curr_s_drift)
    assert res_drift["is_drift"] is True


def test_drift_detector_end_to_end(tmp_path):
    np.random.seed(42)
    df_ref = pd.DataFrame(
        {
            "Age": np.random.normal(30, 10, 200),
            "Fare": np.random.exponential(32, 200),
            "Sex": np.random.choice(["male", "female"], 200, p=[0.6, 0.4]),
        }
    )

    # Dataset idéntico sin drift
    df_curr_stable = pd.DataFrame(
        {
            "Age": np.random.normal(30, 10, 100),
            "Fare": np.random.exponential(32, 100),
            "Sex": np.random.choice(["male", "female"], 100, p=[0.6, 0.4]),
        }
    )

    detector = DataDriftDetector(reference_data=df_ref, drift_share_threshold=0.33)
    res_stable = detector.detect_drift(df_curr_stable)
    assert res_stable["dataset_drift"] is False
    assert res_stable["drift_share"] < 0.33

    # Generación de HTML
    html_path = tmp_path / "drift_test.html"
    detector.generate_html_report(df_curr_stable, output_path=str(html_path))
    assert html_path.exists()
    assert "Odysseus AI - Auditoría de Data Drift" in html_path.read_text(encoding="utf-8")


def test_serving_drift_endpoints():
    with TestClient(app) as test_client:
        # 1. Test POST /monitoring/drift
        payload = {
            "passengers": [
                {
                    "PassengerId": 1,
                    "Pclass": 1,
                    "Name": "Cumings, Mrs. John Bradley",
                    "Sex": "female",
                    "Age": 38.0,
                    "SibSp": 1,
                    "Parch": 0,
                    "Ticket": "PC 17599",
                    "Fare": 71.2833,
                    "Cabin": "C85",
                    "Embarked": "C",
                },
                {
                    "PassengerId": 2,
                    "Pclass": 3,
                    "Name": "Heikkinen, Miss. Laina",
                    "Sex": "female",
                    "Age": 26.0,
                    "SibSp": 0,
                    "Parch": 0,
                    "Ticket": "STON/O2. 3101282",
                    "Fare": 7.925,
                    "Cabin": None,
                    "Embarked": "S",
                },
            ]
        }
        resp = test_client.post("/monitoring/drift", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "dataset_drift" in data
        assert "drift_share" in data
        assert "html_dashboard_url" in data

        # 2. Test GET /monitoring/drift/dashboard
        dash_resp = test_client.get("/monitoring/drift/dashboard")
        assert dash_resp.status_code == 200
        assert "text/html" in dash_resp.headers["content-type"]
        assert "Odysseus AI - Auditoría de Data Drift" in dash_resp.text
