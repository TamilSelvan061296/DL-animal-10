
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import models
from collections import OrderedDict

import time
import mlflow
import mlflow.pytorch
import logging

logger = logging.getLogger(__name__)


class train:
    def __init__(self):
        self.load_pretrained_model()

    def load_pretrained_model(self) -> None:
        self.model = models.densenet121(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False

        classifier = nn.Sequential(OrderedDict([
                                    ('fc1', nn.Linear(1024, 500)),
                                    ('relu1', nn.ReLU()),
                                    ('fc2', nn.Linear(500, 10)),
                                    ('output', nn.LogSoftmax(dim=1))
                                    ]))

        self.model.classifier = classifier


    def train_the_model(self, epochs, learning_rate, 
                        train_dataloader, test_dataloader):

        logger.info("Training started. For training logs visit 'http://localhost:5000/'")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        criterion = nn.NLLLoss()
        optimizer = optim.Adam(self.model.classifier.parameters(), lr = learning_rate)

        self.model = self.model.to(device)

        # registering in mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("animal_classifier")

        log_every = 5

        hyperparams = {
            "epochs": epochs,
            "log_every": log_every,
            "optimizer": optimizer.__class__.__name__,
            "lr": optimizer.param_groups[0]['lr'],
            "criterion": criterion.__class__.__name__
        }

        with mlflow.start_run(run_name = "run_2"):

            mlflow.log_params(hyperparams)

            steps = 0
            running_loss = 0

            for epoch in range(epochs):
                start = time.perf_counter()
                self.model.train()

                for inputs, labels in train_dataloader:
                    steps += 1
                    inputs, labels = inputs.to(device), labels.to(device)

                    optimizer.zero_grad()

                    logps = self.model.forward(inputs)
                    loss = criterion(logps, labels)

                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                    if steps % log_every == 0:
                        self.model.eval()
                        test_loss = 0
                        accuracy = 0 

                        with torch.no_grad():
                            for inputs, labels in test_dataloader:
                                inputs, labels = inputs.to(device), labels.to(device)

                                logps = self.model.forward(inputs)
                                batch_loss = criterion(logps, labels)

                                test_loss += batch_loss.item()
                                ps = torch.exp(logps)

                                top_p, top_class = ps.topk(1, dim = 1)
                                equals = top_class == labels.view(*top_class.shape)
                                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                        elapsed = time.perf_counter() - start

                        # 2) Build a quick grid of, say, the first 4 images in this batch
                        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
                        preds = self.model(inputs).argmax(1).cpu()
                        for i, ax in enumerate(axes.flatten()):
                            img = inputs[i].cpu().permute(1, 2, 0)  # C×H×W → H×W×C
                            ax.imshow(img)
                            ax.set_title(f"True: {labels[i].item()}  Pred: {preds[i].item()}")
                            ax.axis("off")
                        plt.tight_layout()

                        # 3) Log that figure to MLflow under this run
                        mlflow.log_figure(fig, f"batch_{steps:04d}_preds.png")
                        plt.close(fig)

                        # Log metrics to MLflow
                        mlflow.log_metric("train_loss", running_loss, step=steps)
                        mlflow.log_metric("test_loss", test_loss / len(test_dataloader), step=steps)
                        mlflow.log_metric("accuracy", accuracy / len(test_dataloader), step=steps)
                        mlflow.log_metric("epoch_time_s", elapsed, step=steps)

                        print(f"Epoch {epoch+1}/{epochs} took {elapsed:.2f} s;"
                            f"Train_loss: {running_loss};"
                            f"Test_loss: {test_loss/len(test_dataloader):.3f}; "
                            f"Accuracy: {accuracy/len(test_dataloader):.3f}")
                        running_loss = 0
                        self.model.train()
