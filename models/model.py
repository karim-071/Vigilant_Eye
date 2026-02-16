import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


def get_model(weight_path=None, device="cpu"):

    model = efficientnet_b0(
        weights=EfficientNet_B0_Weights.IMAGENET1K_V1
    )

    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.classifier[1].in_features, 1)
    )

    # LOAD TRAINED WEIGHTS
    if weight_path is not None:
        state_dict = torch.load(weight_path, map_location=device)
        model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    return model
