from fastapi.testclient import TestClient

from src.serving.app import app

client = TestClient(app)


def test_health_check():
    with TestClient(app) as test_client:
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True


def test_model_metadata():
    with TestClient(app) as test_client:
        response = test_client.get("/model/metadata")
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "Titanic_Survival_Production_Pipeline"
        assert data["cv_accuracy"] >= 0.80


def test_predict_single_high_probability_female_pclass1():
    with TestClient(app) as test_client:
        payload = {
            "PassengerId": 101,
            "Pclass": 1,
            "Name": "Astor, Mrs. John Jacob (Madeleine Talmadge Force)",
            "Sex": "female",
            "Age": 18.0,
            "SibSp": 1,
            "Parch": 0,
            "Ticket": "PC 17757",
            "Fare": 227.525,
            "Cabin": "C62",
            "Embarked": "C",
        }
        response = test_client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["passenger_id"] == 101
        assert data["prediction"] == 1
        assert data["status"] == "Survived"
        assert data["survival_probability"] > 0.50


def test_predict_single_low_probability_male_pclass3():
    with TestClient(app) as test_client:
        payload = {
            "PassengerId": 102,
            "Pclass": 3,
            "Name": "Braund, Mr. Owen Harris",
            "Sex": "male",
            "Age": 22.0,
            "SibSp": 1,
            "Parch": 0,
            "Ticket": "A/5 21171",
            "Fare": 7.25,
            "Cabin": None,
            "Embarked": "S",
        }
        response = test_client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["passenger_id"] == 102
        assert data["prediction"] == 0
        assert data["status"] == "Did Not Survive"
        assert data["survival_probability"] < 0.50


def test_predict_batch():
    with TestClient(app) as test_client:
        payload = {
            "passengers": [
                {
                    "PassengerId": 1,
                    "Pclass": 1,
                    "Name": "Cumings, Mrs. John Bradley (Florence Briggs Thayer)",
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
                    "Name": "Dooley, Mr. Patrick",
                    "Sex": "male",
                    "Age": 32.0,
                    "SibSp": 0,
                    "Parch": 0,
                    "Ticket": "370376",
                    "Fare": 7.75,
                    "Cabin": None,
                    "Embarked": "Q",
                },
            ]
        }
        response = test_client.post("/predict/batch", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["total_samples"] == 2
        assert len(data["predictions"]) == 2
        assert 0.0 <= data["survival_rate"] <= 1.0


def test_eda_bi_dashboard():
    with TestClient(app) as test_client:
        response = test_client.get("/monitoring/eda/dashboard")
        assert response.status_code == 200
        assert "<title>Odysseus AI - Executive EDA & Business Intelligence Dashboard</title>" in response.text
        assert "Supervivencia Global" in response.text
