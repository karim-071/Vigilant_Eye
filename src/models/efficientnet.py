import torch.nn as nn
from torchvision import models

def get_model():
    model = models.efficientnet_b0(pretrained=True)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, 1
    )
    return model
