from pathlib import Path
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

import mlflow
import mlflow.sklearn

# Configure MLflow tracking
mlflow.set_tracking_uri("http://127.0.0.1:6001")
mlflow.set_experiment("iris_classification")

n_estimators = 100
random_state = 42
test_size = 0.2

# Load and split dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data,
    iris.target,
    test_size=test_size,
    random_state=random_state
)

# Parameters to track
model_parameters = {
    "model_type": "RandomForestClassifier",
    "n_estimators": n_estimators,
    "random_state": random_state,
    "test_size": test_size
}

# Start MLflow run
with mlflow.start_run() as run:
    mlflow.log_params(model_parameters)

    # Train model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state
    )
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, average="macro", zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, average="macro", zero_division=0
    )

    # Log metrics
    mlflow.log_metrics({
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall
    })

    # Log trained model to MLflow
    mlflow.sklearn.log_model(
    sk_model=model,
    name="iris_model",
    serialization_format="pickle",
    registered_model_name="IrisRandomForestModel"
    )

    # Save local model copy
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "iris_model.joblib")

    print("MLflow Run ID:", run.info.run_id)
    print("Test Accuracy:", accuracy)
    print("Test Precision:", precision)
    print("Test Recall:", recall)
    print("Model saved to:", model_dir / "iris_model.joblib")

