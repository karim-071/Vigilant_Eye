from torch.utils.data import DataLoader
from src.data.hf_dataset import DeepfakeDataset
from src.data.transforms import get_train_transforms, get_val_transforms

def get_dataloaders(batch_size=32):
    train_ds = DeepfakeDataset("train", get_train_transforms())
    val_ds   = DeepfakeDataset("validation", get_val_transforms())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
