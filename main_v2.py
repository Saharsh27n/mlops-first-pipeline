from fastapi import FastAPI, Request, HTTPException
from flask import app
from pydantic import BaseModel, Field
import mlflow
import mlflow.sklearn
import os
import contextlib
from typing import callable



# app = FastAPI(title="Iris Model API")

# MLflow configuration
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:6001"
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

MODEL_URI = "models:/iris-classifier@champion"


def load_mlflow_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return mlflow.sklearn.load_model(MODEL_URI)


class IrisRequest(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)


#application factory

def create_app(model_loader: callable=load_mlflow_model(),version_loader: callable=None):
    if version_loader is None:
        def version_loader():
            client = mlflow.MlflowClient()
            champion_version = client.get_model_version_by_alias(
                name="iris-classifier",
                alias="champion"
            )
            return champion_version



    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        #load the model once during startup
        #injected loader can be real MLFLOW or a test model
        try:
            app.state.model=model_loader()
        except Exception as ex:
            raise RuntimeError(
                f"failed to load mlflow champion model..\n{ex}"
            )
        finally:
            pass
        app.state.model_version = version_loader()
        print(f"Loaded model: iris-classifier:{app.state.model_version}")
        yield

    app.router.lifespan_context = lifespan

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_name": "iris-classifier",
        "model_version": app.state.model_version
    }


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

    return {
        "prediction": prediction,
        "class_name": [
            "setosa",
            "versicolor",
            "virginica"
        ][prediction]
    }
