from sklearn.preprocessing import LabelEncoder
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import time
import torch.nn as nn

from data import load_csv, preprocessing, create_dataloader, save_le, splits
from model import create_model, freeze_backbone_keep_fc_trainable, unfreeze_layer3_layer4_and_fc, make_param_groups
from config import MODEL_PATH, TRAIN_EPOCHS, HEAD_EPOCHS, TRAINING_LOSS, CSV_PATH

def head_training(model: nn.Module, train_loader: DataLoader, device: torch.device, epochs=HEAD_EPOCHS, loss_fn=TRAINING_LOSS) -> None:
    """
    Trains only the final fully connected (FC) layer of the model for a few epochs.
    Used as the head training phase before fine-tuning deeper layers.
    """
    optimizer_head = torch.optim.AdamW(model.fc.parameters(), lr=1e-3, weight_decay=0.01)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X, y in tqdm(train_loader, desc=f"Head epoch {epoch + 1}", leave=False):
            X, y = X.to(device), y.to(device)
            # zero the parameter gradients
            optimizer_head.zero_grad()

            # forward + backward + optimize
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer_head.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        print(f"Head epoch {epoch + 1} complete. Avg Loss: {avg_loss:.4f}")
    print("Head training completed!")


def save_model(model: nn.Module, model_path = MODEL_PATH) -> None:
    """
    Saves the model's state dictionary to the specified file path.
    """
    torch.save(model.state_dict(), model_path)


def main_training(model: nn.Module, train_loader: DataLoader, param_groups: dict,
                  device: torch.device, epochs=TRAIN_EPOCHS, loss_fn=TRAINING_LOSS) -> None:
    """
    Fine-tunes the model using the given parameter groups, optimizer, and scheduler.
    Trains for the specified number of epochs and saves the final model.
    """
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs * len(train_loader))
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X, y in tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False):
            X, y = X.to(device), y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1} complete. Avg Loss: {avg_loss:.4f}")
    print("Training Complete")
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTraining complete in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    save_model(model)


def train_model() -> None:
    """
    Full training pipeline: loads and preprocesses data, splits into subsets,
    initializes the model, runs head training, fine-tuning, and saves the model.
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    df = load_csv()
    le = LabelEncoder()
    df_clean = preprocessing(df, le, fit=True)
    df_clean = splits(df_clean)
    df_clean.to_csv(CSV_PATH, index=False)
    df_train = df_clean[df_clean["subset"] == "train"].reset_index(drop=True)
    train_loader = create_dataloader(df_train, train=True)
    model = create_model(num_classes=len(le.classes_), device=device)
    freeze_backbone_keep_fc_trainable(model)
    head_training(model, train_loader, device)
    unfreeze_layer3_layer4_and_fc(model)
    pg = make_param_groups(model)
    save_le(le)
    main_training(model, train_loader, pg, device)

if __name__ == "__main__":
    train_model()
