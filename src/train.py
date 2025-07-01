from src.dl_animal_10.data.data_loader import get_dataloaders
from src.dl_animal_10.models.model import get_model
from src.dl_animal_10.trainer.trainer import Trainer

def main():
    train_dataloader, test_dataloader = get_dataloaders()
    model = get_model()
    trainer = Trainer(model, train_dataloader, test_dataloader)
    trainer.train()

if __name__ == '__main__':
    main()
