import torch
import torch.nn as nn
from src.data.dataloader import get_dataloaders
from src.models.efficientnet import get_model
from src.models.trainer import train_one_epoch, evaluate

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = get_dataloaders(batch_size=32)

    model = get_model().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 3
    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_acc, val_auc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")

    torch.save(model.state_dict(), "weights/vigilant_eye.pth")
    print("Model saved to weights/vigilant_eye.pth")

if __name__ == "__main__":
    main()
