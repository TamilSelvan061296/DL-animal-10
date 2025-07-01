import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data
DATA_DIR = '/home/tamil/.cache/kagglehub/datasets/alessiocorrado99/animals10/versions/2/raw-img'
TRAIN_RATIO = 0.8
BATCH_SIZE = 32

# Model
MODEL_NAME = 'densenet121'
PRETRAINED = True
FREEZE_PARAMS = True

# Training
EPOCHS = 1
PRINT_EVERY = 5
LEARNING_RATE = 0.003

# MLflow
MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT_NAME = "animal_classifier"
