
import os
from contextlib import asynccontextmanager
from typing import Callable

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field


# -------------------------------
# MLflow Configuration
# -------------------------------

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:6001"
)

MODEL_NAME = "iris-classifier"
MODEL_ALIAS = os.getenv("MLFLOW_MODEL_ALIAS", "production")

MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"


# -------------------------------
# MLflow Model Loader
# -------------------------------

def load_mlflow_model():

    # Lazy imports: MLflow is not imported during test collection
    import mlflow
    import mlflow.sklearn

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model = mlflow.sklearn.load_model(MODEL_URI)

    return model


def load_model_version():

    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    client = mlflow.MlflowClient()

    model_version = client.get_model_version_by_alias(
        name=MODEL_NAME,
        alias=MODEL_ALIAS
    )

    return str(model_version.version)


# -------------------------------
# Request Schema
# -------------------------------

class IrisRequest(BaseModel):

    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)


# -------------------------------
# Application Factory
# -------------------------------

def create_app(
    model_loader: Callable = load_mlflow_model,
    version_loader: Callable = load_model_version
):

    @asynccontextmanager
    async def lifespan(app: FastAPI):

        app.state.model = model_loader()
        app.state.model_version = version_loader()

        print(
            f"Model loaded successfully. "
            f"Version: {app.state.model_version}"
        )

        yield

    app = FastAPI(
        title="Iris ML API",
        description="Iris classification using MLflow",
        version="1.0.0",
        lifespan=lifespan
    )

    # -------------------------------
    # Root Endpoint
    # -------------------------------

    @app.get("/")
    def root():

        return {
            "message": "Iris ML API is running"
        }

    # -------------------------------
    # Health Endpoint
    # -------------------------------

    @app.get("/health")
    def health():

        return {
            "status": "healthy",
            "model_name": MODEL_NAME,
            "model_version": app.state.model_version
        }

    # -------------------------------
    # Prediction Endpoint
    # -------------------------------

    @app.post("/predict")
    def predict(data: IrisRequest, request: Request):

        model = request.app.state.model

        features = [[
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
        ]]

        prediction = int(model.predict(features)[0])

        classes = [
            "setosa",
            "versicolor",
            "virginica"
        ]

        return {
            "prediction": prediction,
            "class_name": classes[prediction]
        }

    return app


# -------------------------------
# Production Application
# -------------------------------

app = create_app()
