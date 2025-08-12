from sklearn.preprocessing import LabelEncoder
import torch.optim.lr_scheduler as lr_scheduler
import torch
from tqdm import tqdm
import time

from data import load_csv, preprocessing, create_dataloader, save_le
from model import create_model, freeze_backbone_keep_fc_trainable, unfreeze_layer3_layer4_and_fc, make_param_groups
from config import MODEL_PATH, TRAIN_EPOCHS, HEAD_EPOCHS, TRAINING_LOSS

def head_training(model, train_loader, device, epochs=HEAD_EPOCHS, loss_fn=TRAINING_LOSS):
    optimizer_head = torch.optim.AdamW(model.fc.parameters(), lr=1e-3, weight_decay=0.01)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X, y in tqdm(train_loader, desc=f"Head epoch {epoch + 1}", leave=False):
            start_batch = time.time()
            X, y = X.to(device), y.to(device)
            # zero the parameter gradients
            optimizer_head.zero_grad()

            # forward + backward + optimize
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer_head.step()
            epoch_loss += loss.item()
            end_batch = time.time()
        avg_loss = epoch_loss / len(train_loader)
        print(f"Head epoch {epoch + 1} complete. Avg Loss: {avg_loss:.4f}")
    print("Head training completed!")


def save_model(model, model_path = MODEL_PATH):
    torch.save(model.state_dict(), model_path)


def main_training(model, train_loader, param_groups, device, epochs=TRAIN_EPOCHS, loss_fn=TRAINING_LOSS):
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


def train_model():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    df = load_csv()
    le = LabelEncoder()
    df_clean = preprocessing(df, le, fit=True)
    df_train = df_clean[df_clean["subset"] == "train"].reset_index(drop=True)
    df_test = df_clean[df_clean["subset"] == "test"].reset_index(drop=True)
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