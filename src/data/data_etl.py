import torch
from torchvision import datasets, transforms

from collections import Counter
import logging
from utils.helper_functions import save_tensor_image

logger = logging.getLogger(__name__)

def load_and_transform(path: str, split_ratio: float) -> None:

    dataset = datasets.ImageFolder(path)
    train_size: float = int(split_ratio * len(dataset))
    test_size: float = len(dataset) - train_size

    train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])

    logger.info(f"Loaded the data from {path}")

    train_data.dataset.transform = transforms.Compose([transforms.RandomRotation(30),
                                                    transforms.RandomResizedCrop(224),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor()])

    test_data.dataset.transform = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.ToTensor()])

    logger.info(f"Transformed the data")


    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

    logger.info(f"Loaded the data into torch dataloader after transformation")
    logger.info(f"Information about the dataset")

    verify_the_dataset(train_loader=train_dataloader, test_loader=test_dataloader)

    return train_dataloader, test_dataloader


def verify_the_dataset(train_loader, test_loader = None) -> None:

    data, target = next(iter(train_loader))
    logger.info(f"Data shape: {data.shape}")
    logger.info(f"Target shape: {target.shape}")

    train_dataset = train_loader.dataset.dataset
    all_labels = train_dataset.targets    # e.g. [0, 5, 3, 0, 1, …]
    class_names = train_dataset.classes   # e.g. ["cat","dog","elephant",…]

    logger.info(f"Class Names: {class_names}")

    counts = Counter(all_labels)
    for idx, cnt in sorted(counts.items()):
        logger.info(f"{class_names[idx]:10s} → {cnt}")

    # now print names for that batch:
    for i, idx in enumerate(target):
        logger.info(f"Image {i:2d}: index={idx.item():2d}  →  {class_names[idx]}")


    # test the data loader
    images, labels = next(iter(train_loader))
    path = save_tensor_image(image=images[0], save_dir="/home/tamil/DL-animal-10/tmp_artifacts", 
                      normalize=False)
    
    logger.info(f"Take a look at an image in the train_dataloader here: '{path}'")