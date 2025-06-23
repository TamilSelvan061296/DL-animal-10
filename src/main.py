import typer
from typing import Type
from data.data_etl import load_and_transform
from config.config_loader import Config
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
def pipeline(config_path: str):
    
    # load the config
    config = Config(config_path)
    # load and transform the training data
    train_dataloader, test_dataloader = load_and_transform(config)

    # train the DL model
    trainer = train()
    trainer.train_the_model(train_dataloader=train_dataloader, 
                          test_dataloader=test_dataloader, config=config)


def main():
    app()


if __name__ == "__main__":
    app()



