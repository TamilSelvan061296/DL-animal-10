import time
import torch
from torch import nn, optim
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
from src.dl_animal_10.config import config

class Trainer:
    def __init__(self, model, train_dataloader, test_dataloader):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.device = config.DEVICE
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.classifier.parameters(), lr=config.LEARNING_RATE)
        self.model.to(self.device)

    def train(self):
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

        hyperparams = {
            "epochs": config.EPOCHS,
            "print_every": config.PRINT_EVERY,
            "optimizer": self.optimizer.__class__.__name__,
            "lr": self.optimizer.param_groups[0]['lr'],
            "criterion": self.criterion.__class__.__name__
        }

        with mlflow.start_run():
            mlflow.log_params(hyperparams)

            steps = 0
            running_loss = 0

            for epoch in range(config.EPOCHS):
                start = time.perf_counter()
                self.model.train()

                for inputs, labels in self.train_dataloader:
                    steps += 1
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    self.optimizer.zero_grad()

                    logps = self.model.forward(inputs)
                    loss = self.criterion(logps, labels)

                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item()

                    if steps % config.PRINT_EVERY == 0:
                        self.model.eval()
                        test_loss = 0
                        accuracy = 0

                        with torch.no_grad():
                            for inputs, labels in self.test_dataloader:
                                inputs, labels = inputs.to(self.device), labels.to(self.device)

                                logps = self.model.forward(inputs)
                                batch_loss = self.criterion(logps, labels)

                                test_loss += batch_loss.item()
                                ps = torch.exp(logps)

                                top_p, top_class = ps.topk(1, dim=1)
                                equals = top_class == labels.view(*top_class.shape)
                                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                        elapsed = time.perf_counter() - start

                        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
                        preds = self.model(inputs).argmax(1).cpu()
                        for i, ax in enumerate(axes.flatten()):
                            img = inputs[i].cpu().permute(1, 2, 0)
                            ax.imshow(img)
                            ax.set_title(f"True: {labels[i].item()}  Pred: {preds[i].item()}")
                            ax.axis("off")
                        plt.tight_layout()

                        mlflow.log_figure(fig, f"batch_{steps:04d}_preds.png")
                        plt.close(fig)

                        mlflow.log_metric("train_loss", running_loss, step=steps)
                        mlflow.log_metric("test_loss", test_loss / len(self.test_dataloader), step=steps)
                        mlflow.log_metric("accuracy", accuracy / len(self.test_dataloader), step=steps)
                        mlflow.log_metric("epoch_time_s", elapsed, step=steps)

                        print(f"Epoch {epoch + 1}/{config.EPOCHS} took {elapsed:.2f} s;"
                              f"Train_loss: {running_loss};"
                              f"Test_loss: {test_loss / len(self.test_dataloader):.3f}; "
                              f"Accuracy: {accuracy / len(self.test_dataloader):.3f}")
                        running_loss = 0
                        self.model.train()
