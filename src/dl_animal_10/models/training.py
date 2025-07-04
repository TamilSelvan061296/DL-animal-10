
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
from mlflow.models.signature import infer_signature
import logging
from dl_animal_10.config.config_loader import Config

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


    def train_the_model(self, train_dataloader, test_dataloader, config: Config):

        # load the config
        train_cfg = config.get("train", {})
        mlflow_cfg = config.get("mlflow", {})

        logger.info("Training started. For training logs visit 'http://localhost:5000/'")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        criterion = nn.NLLLoss()
        optimizer = optim.Adam(self.model.classifier.parameters(), lr = train_cfg["learning_rate"])

        self.model = self.model.to(device)

        # registering in mlflow
        mlflow.set_tracking_uri(f"{mlflow_cfg["url"]}:{mlflow_cfg["port"]}")
        mlflow.set_experiment(mlflow_cfg["experiment_name"])

        log_every = mlflow_cfg["log_every"]

        hyperparams = {
            "epochs": train_cfg["epochs"],
            "log_every": log_every,
            "optimizer": optimizer.__class__.__name__,
            "lr": optimizer.param_groups[0]['lr'],
            "criterion": criterion.__class__.__name__
        }

        # inferring signature for the model:
        example_tensor = torch.rand(1, 3, 224, 224, device=device)

        with torch.no_grad():
            output_tensor = self.model(example_tensor)

        example_input_np = example_tensor.cpu().numpy()
        output_np        = output_tensor.cpu().numpy()

        signature = infer_signature(example_input_np, output_np)

        # start model training
        with mlflow.start_run(run_name = mlflow_cfg["run_name"]) as run:

            mlflow.log_params(hyperparams)

            steps = 0
            running_loss = 0
            required_max_accuracy = 0.85

            for epoch in range(train_cfg["epochs"]):
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

                        print(f"Epoch {epoch+1}/{train_cfg["epochs"]} took {elapsed:.2f} s;"
                            f"Train_loss: {running_loss};"
                            f"Test_loss: {test_loss/len(test_dataloader):.3f}; "
                            f"Accuracy: {accuracy/len(test_dataloader):.3f}")
                        running_loss = 0
                        self.model.train()


                        if accuracy/len(test_dataloader) > required_max_accuracy: # save the version of the model if accuracy > 0.85
                            
                            mlflow.pytorch.log_model(
                                pytorch_model=self.model,
                                artifact_path="models",
                                registered_model_name="animal_classifier",
                                signature=signature,
                                )
                            print("Model registered under run:", run.info.run_id)
                            required_max_accuracy = accuracy/len(test_dataloader)
                        
                        self.model.train()
            
            
            mlflow.pytorch.log_model(
                            pytorch_model=self.model,
                            artifact_path="models",
                            registered_model_name="animal_classifier",
                            signature=signature,
                            )
            print("Model registered under run:", run.info.run_id)
            
            mlflow.pytorch.save_model(
                pytorch_model = self.model,
                path          = "/home/ubuntu/work/DL-animal-10/src/models/animal_classifier",
                signature=signature,
                )

            print("Local model saved to /home/ubuntu/work/DL-animal-10/src/models/animal_classifier")              
