
import pytest

from fastapi.testclient import TestClient

from main_v2 import create_app


# -------------------------------
# Fake ML Model
# -------------------------------

class FakeIrisModel:

    def predict(self, features):

        return [0]


# -------------------------------
# Fake Dependencies
# -------------------------------

def fake_model_loader():

    return FakeIrisModel()


def fake_version_loader():

    return "fake-version"


# -------------------------------
# Test Fixture
# -------------------------------

@pytest.fixture
def client():

    app = create_app(
        model_loader=fake_model_loader,
        version_loader=fake_version_loader
    )

    with TestClient(app) as test_client:

        yield test_client


# -------------------------------
# Health Endpoint Test
# -------------------------------

def test_health_endpoint(client):

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_name"] == "iris-classifier"
    assert data["model_version"] == "fake-version"


# -------------------------------
# Prediction Endpoint Test
# -------------------------------

def test_valid_iris_prediction(client):

    payload = {
        "sepal_length": 3.4,
        "sepal_width": 1.4,
        "petal_length": 3.4,
        "petal_width": 0.4
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] == 0
    assert result["class_name"] == "setosa"


# -------------------------------
# Invalid Input Test
# -------------------------------

def test_invalid_iris_prediction(client):

    payload = {
        "sepal_length": -3.4,
        "sepal_width": 1.4,
        "petal_length": 3.4,
        "petal_width": 0.4
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 422
