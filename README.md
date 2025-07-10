 # DL Animal 10

## Project Overview

This project demonstrates the full cycle of deep learning model development: from data ETL, training, and experiment tracking, to model versioning and serving for inference. The goal is to build robust, scalable, and flexible training pipelines using industry-standard practices.

- **Dataset:** [Animal-10](https://www.kaggle.com/datasets/alessiocorrado99/animals10) from Kaggle
- **Model:** DenseNet121 (PyTorch)
- **Tracking/Serving:** MLflow
- **Dependency Management:** [uv](https://github.com/astral-sh/uv)
- **Cloud:** AWS (for GPU training)

## Features
- Modularized code for ETL, training, and configuration
- MLflow for experiment tracking, model versioning, and serving
- Custom preprocessing and model signature for robust inference
- Example config and input files for reproducibility

---

## Setup

### 1. Install uv (Dependency Manager)
[uv](https://github.com/astral-sh/uv) is a fast Python package/dependency manager. Install it globally:

```bash
pip install uv
```

### 2. Install Project Dependencies

```bash
uv pip install -r pyproject.toml
```

Or, for editable/development mode:

```bash
uv pip install -e .
```

---

## Dataset Download

Download the Animal-10 dataset from Kaggle using the provided script:

```bash
python data/kaggle_data_download.py
```

This will download the dataset and print the path to the files. Update the `data.path` in `src/dl_animal_10/config/config.yaml` if needed.

---

## Configuration

Edit `src/dl_animal_10/config/config.yaml` to set paths, training hyperparameters, and MLflow tracking server info:

```yaml
data:
  path: '/path/to/your/dataset'
  test_size: 0.2
  batch_size: 64
  split_ratio: 0.8
train:
  learning_rate: 0.003
  epochs: 3
mlflow:
  log_every: 5
  experiment_name: "animal_classifier"
  run_name: "final-training-run"
  url: "http://0.0.0.0"
  port: 5000
```

---

## Training the Model

Run the training pipeline using the Typer CLI:

```bash
python -m src.dl_animal_10.main src/dl_animal_10/config/config.yaml
```

Or, if installed as a script:

```bash
train-animal-10 src/dl_animal_10/config/config.yaml
```

Training progress, metrics, and artifacts will be logged to MLflow.

---

## MLflow Tracking & Model Serving

### 1. Start MLflow Tracking Server

```bash
mlflow server --host 0.0.0.0 --port 5000
```

Visit [http://localhost:5000](http://localhost:5000) to view experiments and runs. Note that this server should be started before starting the training.

### 2. Serve the Model with MLflow

After training, serve the best model (example for the model with signature):

```bash
mlflow models serve -m src/dl_animal_10/models/animal_classifier_with_sig -p 1234 --host 0.0.0.0
```

- The model directory may vary depending on your run and configuration.
- The `MLmodel` file and `requirements.txt` in the model directory define the environment and input signature.

---

## Querying the Model

Send a POST request to the served model endpoint. Example (using `requests` in Python):

```python
import requests
import numpy as np

# Prepare input as per the model signature (see serving_input_example.json for format)
input_data = np.random.rand(1, 3, 224, 224).astype(np.float32).tolist()

response = requests.post(
    "http://localhost:1234/invocations",
    json={"inputs": input_data}
)
print(response.json())
```

- See `src/dl_animal_10/models/animal_classifier_with_sig/serving_input_example.json` for a real input example.

---

## Notes
- All dependency management is handled via `uv` and `pyproject.toml`.
- MLflow is used for both experiment tracking and model serving.
- Test scripts in the `tests/` directory are for development/debugging and not part of the main workflow.
- For cloud training, update paths and configs as needed for your environment.

---

## Project Insights

See `README-for-my-approach.md` for a detailed write-up on the project journey, challenges, and design decisions.


**Note:** This README was written using an AI tool(Cursor) and verified by me for corrections.
