from test_serve import preprocess_image, predict_via_rest

from dl_animal_10.data.data_etl import load_and_transform, verify_the_dataset
from dl_animal_10.config.config_loader import Config
from dl_animal_10.models.training import train
import torch

cfg = Config('/home/tamil/DL-animal-10/src/dl_animal_10/config/config.yaml')

train, test = load_and_transform(cfg)

trainer = train()
trainer.train_the_model(train_dataloader=train, 
                        test_dataloader=test, config=cfg)
