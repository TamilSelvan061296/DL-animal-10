import os
import numpy as np
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature, ModelSignature
from mlflow.types.schema import Schema, TensorSpec
import torch

# 1. Download the registered model artifacts
client     = MlflowClient()
model_uri  = "/home/ubuntu/work/DL-animal-10/mlartifacts/371523339529422074/models/m-e87b716f66be497eae0f4d640a2966c2/artifacts"
local_dir  = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)  # :contentReference[oaicite:0]{index=0}

# 2. Load the model locally
model = mlflow.pytorch.load_model(local_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

example_tensor = torch.rand(1, 3, 224, 224, device=device)

with torch.no_grad():
    output_tensor = model(example_tensor)

example_input_np = example_tensor.cpu().numpy()
output_np        = output_tensor.cpu().numpy()

signature = infer_signature(example_input_np, output_np)

# 4. Save the model with the new signature (and optional input_example)
save_path = "/home/ubuntu/work/DL-animal-10/src/models/animal_classifier_with_sig"
mlflow.pytorch.save_model(
    pytorch_model=model,
    path=save_path,
    signature=signature,
    input_example=example_input_np
)                                                                   # :contentReference[oaicite:3]{index=3}

# 5. Register it as a new version
new_uri = f"file://{os.path.abspath(save_path)}"
new_ver = client.create_model_version(
    name="animal_classifier",
    source=new_uri,
    run_id=None
)
print(f"Registered new version: {new_ver.version}")
