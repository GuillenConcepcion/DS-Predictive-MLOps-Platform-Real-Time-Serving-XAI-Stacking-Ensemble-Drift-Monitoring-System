import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.monitoring.drift_detector import DataDriftDetector
from src.serving.app import app
from src.serving.schemas import PassengerInput


def test_passenger_input_valid_normalization():
    p = PassengerInput(
        PassengerId=1,
        Pclass=1,
        Name="Smith, Mr. John",
        Sex="  MALE  ",
        Age=30.0,
        SibSp=0,
        Parch=0,
        Ticket="12345",
        Fare=50.0,
        Cabin="C12",
        Embarked="  c  ",
    )
    assert p.Sex == "male"
    assert p.Embarked == "C"


def test_passenger_input_invalid_sex():
    with pytest.raises(ValidationError):
        PassengerInput(
            PassengerId=1,
            Pclass=1,
            Name="Smith, Mr. John",
            Sex="alien",
            Age=30.0,
        )


def test_passenger_input_invalid_pclass():
    with pytest.raises(ValidationError):
        PassengerInput(
            PassengerId=1,
            Pclass=5,
            Name="Smith, Mr. John",
            Sex="male",
            Age=30.0,
        )


def test_passenger_input_invalid_embarked():
    with pytest.raises(ValidationError):
        PassengerInput(
            PassengerId=1,
            Pclass=1,
            Name="Smith, Mr. John",
            Sex="male",
            Embarked="X",
        )


def test_prediction_drift_detector_synthetic_shift():
    np.random.seed(42)
    # Baseline: 891 samples with ~38% survival probability
    ref_probs = np.random.beta(2, 5, size=891)
    ref_preds = (ref_probs >= 0.34).astype(int)

    # Shifted: 100 samples with ~90% survival probability
    live_probs = np.random.beta(8, 2, size=100)
    live_preds = (live_probs >= 0.34).astype(int)

    detector = DataDriftDetector(
        reference_data=None,
        reference_probabilities=ref_probs,
        reference_predictions=ref_preds,
    )

    report = detector.calculate_prediction_drift(live_probs, live_preds)

    assert report["is_drift_detected"] is True
    assert report["overall_status"] in ["CRITICAL_DRIFT", "MODERATE_DRIFT"]
    assert report["probability_drift"]["psi"] > 0.10


def test_prediction_drift_and_metrics_endpoints():
    with TestClient(app) as client:
        # 1. Check initial metrics
        resp_m = client.get("/monitoring/inference-metrics")
        assert resp_m.status_code == 200

        # 2. Perform a batch of inferences
        payload = {
            "passengers": [
                {
                    "PassengerId": i,
                    "Pclass": 1 if i % 2 == 0 else 3,
                    "Name": f"Passenger, Test {i}",
                    "Sex": "female" if i % 2 == 0 else "male",
                    "Age": 25.0 + i,
                    "SibSp": 0,
                    "Parch": 0,
                    "Ticket": f"TKT-{i}",
                    "Fare": 50.0 + (i * 2),
                    "Embarked": "S",
                }
                for i in range(15)
            ]
        }
        resp_p = client.post("/predict/batch", json=payload)
        assert resp_p.status_code == 200

        # 3. Query prediction drift endpoint
        resp_d = client.get("/monitoring/prediction-drift")
        assert resp_d.status_code == 200
        drift_data = resp_d.json()
        assert drift_data["sample_size"] >= 15
        assert "overall_status" in drift_data
        assert "probability_drift" in drift_data
        assert "decision_drift" in drift_data

        # 4. Query updated inference metrics
        resp_m2 = client.get("/monitoring/inference-metrics")
        assert resp_m2.status_code == 200
        metrics_data = resp_m2.json()
        assert metrics_data["total_inferences"] >= 15
        assert metrics_data["current_buffer_size"] >= 15
        assert 0.0 <= metrics_data["average_survival_probability"] <= 1.0
