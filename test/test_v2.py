from fastapi.testclient import TestClient


from main_v2 import create_app

class fakeirismodel:
    def predict(self, features):
        return [0]  # always return setosa

def fake_model_loader():
    return fakeirismodel()

def fake_version_loader():
    return "fake-version"

app = create_app(model_loader=fake_model_loader, version_loader=fake_version_loader)
