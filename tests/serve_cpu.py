# serve_cpu.py
import torch
import mlflow.pytorch
from fastapi import FastAPI, Request

MODEL_URI = "/home/tamil/DL-animal-10/src/dl_animal_10/models/animal_classifier_with_sig"

app = FastAPI()

@app.on_event("startup")
def load_model():
    # map_location forces CPU load
    global model
    model = mlflow.pytorch.load_model(MODEL_URI, map_location=torch.device("cpu"))

@app.post("/invocations")
async def predict(request: Request):
    payload = await request.json()
    # assume input is a list of lists or similar
    tensor = torch.tensor(payload["instances"])
    with torch.no_grad():
        out = model(tensor).cpu().numpy().tolist()
    return {"predictions": out}
