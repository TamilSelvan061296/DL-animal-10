from torchvision import models
from torch import nn
from src.dl_animal_10.config import config

def get_model():
    model = getattr(models, config.MODEL_NAME)(pretrained=config.PRETRAINED)

    if config.FREEZE_PARAMS:
        for param in model.parameters():
            param.requires_grad = False

    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(1024, 500)),
                                ('relu1', nn.ReLU()),
                                ('fc2', nn.Linear(500, 10)),
                                ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier
    return model
