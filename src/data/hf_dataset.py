from datasets import load_dataset
from torch.utils.data import Dataset

class DeepfakeDataset(Dataset):
    def __init__(self, split, transform=None):
        self.dataset = load_dataset(
            "RohanRamesh/ff-images-dataset",
            split=split
        )
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["image"]
        label = sample["label"]

        if self.transform:
            image = self.transform(image)

        return image, label
