import os
from mlflow.tracking import MlflowClient

client = MlflowClient()
model_name = "animal_classifier"   # ← replace with your registered model name

for mv in client.search_model_versions(f"name = '{model_name}'"):
    # mv.version is the integer version (1, 2, …)
    # mv.source  is the artifact URI, ending in models/m-<hash>
    hash_dir = os.path.basename("/home/ubuntu/DL-animal-10/mlartifacts/371523339529422074/models")
    print(f"version {mv.version:>2} → folder `{hash_dir}`")
