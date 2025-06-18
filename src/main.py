import typer
from typing import Type
from data.data_etl import load_and_transform
from models.training import train
import logging

logging.basicConfig(
    filename="log_file.log",
    format="%(asctime)s %(message)s",
    filemode="w",
    level=logging.INFO,
    force=True,          # Python 3.8+
)


app = typer.Typer()

@app.command()
def pipeline(data_path: str, split_ratio: float, 
             epochs: int, lr: float):

    # load and transform the training data
    train_dataloader, test_dataloader = load_and_transform(path=data_path, split_ratio=split_ratio)

    # train the DL model
    trainer = train()
    trainer.train_the_model(epochs=epochs, train_dataloader=train_dataloader, 
                          test_dataloader=test_dataloader, learning_rate = lr)


def main():
    app()


if __name__ == "__main__":
    app()



