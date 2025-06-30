# serve_cpu.py

import torch
import mlflow.pytorch
from fastapi import FastAPI, Request,  Body
from contextlib import asynccontextmanager

MODEL_URI = "/home/tamil/DL-animal-10/src/dl_animal_10/models/animal_classifier_with_preprocessing_step"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model into CPU memory on startup
    global model
    model = mlflow.pyfunc.load_model(MODEL_URI)
    yield
    # (Optional) add any teardown/cleanup here

app = FastAPI(lifespan=lifespan)

@app.post("/invocations")
async def predict(data: bytes = Body(..., media_type="application/octet-stream")):
    # assume input is a list of lists or similar
    with torch.no_grad():
        out = model.predict(data)
    return {"predictions": out}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "serve_cpu:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )