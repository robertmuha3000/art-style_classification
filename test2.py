import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torchvision.io import decode_image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
import time
import string
from sklearn.utils.class_weight import compute_class_weight
from torchvision.transforms import InterpolationMode, RandAugment
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, data, img_dir, transform = None):
        self.data = data
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        label = self.data.iloc[idx]["encoded_genre"]
        if self.transform:
            image = self.transform(image)
        return image, label

def main():
    start_time = time.time()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(device)

    image_root = "painting_classify/archive-3/"
    df = pd.read_csv("classes.csv")

    print(sum(df["subset"] == "train"))
    print(sum(df["subset"] == "test"))

    pattern = r'^[a-zA-Z0-9\s.,!?;:\'\"@#&$%^*()_+\-=\[\]{}<>/\\|~`]*$'

    df_clean = df[df["filename"].str.match(pattern, na=False)]


    print(sum(df_clean["subset"] == "train"))
    print(sum(df_clean["subset"] == "test"))

    """
    with Image.open(image_root+df_clean["filename"][1220]) as im:
        im.show()
        print(im.size)
    """

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BICUBIC, antialias=True),
        transforms.RandomHorizontalFlip(),
        RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        transforms.RandomErasing(p=0.25)
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        transforms.RandomErasing(p=0.25)
    ])

    # blend genres together later
    safe_genres = {"['Impressionism']", "['Realism']", "['Romanticism']", "['Expressionism']", "['Post Impressionism']",
                "['Baroque']", "['Art Nouveau Modern']", "['Symbolism']", "['Abstract Expressionism']", "['Northern Renaissance']",
                "['Naive Art Primitivism']", "['Rococo']", "['Cubism']", "['Color Field Painting']", "['Pop Art']", "['Early Renaissance']",
                "['High Renaissance']", "['Minimalism']", "['Ukiyo e']", "['Fauvism']", "['Pointillism']", "['Contemporary Realism']",
                "['New Realism']", "['Synthetic Cubism']", "['Analytical Cubism']", "['Action painting']"}

    genre_merge_map = {
        "['Post Impressionism']": "['Impressionism']",
        "['Analytical Cubism']": "['Cubism']",
        "['Synthetic Cubism']": "['Cubism']",
        "['Color Field Painting']": "['Abstract Expressionism']",
        "['Action painting']": "['Abstract Expressionism']",
        "['Early Renaissance']": "['Renaissance']",
        "['High Renaissance']": "['Renaissance']",
        "['New Realism']": "['Realism']",
        "['Contemporary Realism']": "['Realism']"
    }

    print("!!!\n!!!")

    le = LabelEncoder()

    df_clean = df_clean[df_clean["genre"].isin(safe_genres)]
    df_clean["genre_merged"] = df_clean["genre"].replace(genre_merge_map)
    df_clean["encoded_genre"] = le.fit_transform(df_clean["genre_merged"])

    print(df_clean["encoded_genre"])


    train = df_clean[df_clean["subset"] == "train"]
    test = df_clean[df_clean["subset"] == "test"]
    print(train)
    train_dataset = CustomDataset(data=train, img_dir=image_root, transform=transform_train)
    test_dataset = CustomDataset(data=test, img_dir=image_root, transform=transform_test)

    print(train_dataset)

    num_workers = max(os.cpu_count() - 1, 0)
    persistent_workers = num_workers > 0

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size, shuffle = True, num_workers=num_workers, persistent_workers=persistent_workers)
    test_loader = DataLoader(test_dataset, batch_size, shuffle = False, num_workers=num_workers, persistent_workers=persistent_workers)

    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, len(le.classes_))

    model = model.to(device)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    loss_fn = torch.nn.CrossEntropyLoss( label_smoothing=0.1)
    optimizer_head = torch.optim.AdamW(model.fc.parameters(), lr=1e-3, weight_decay=0.01)

    for epoch in range(3):
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


    for p in model.layer4.parameters():
        p.requires_grad = True
    for p in model.layer3.parameters():
        p.requires_grad = True

    param_groups = [
        {"params": [p for p in model.layer4.parameters() if p.requires_grad], "lr": 1e-4},
        {"params": [p for p in model.layer3.parameters() if p.requires_grad], "lr": 1e-4},
        {"params": [p for p in model.fc.parameters() if p.requires_grad],     "lr": 3e-4}
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    epochs = 12
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs * len(train_loader))

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X, y in tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False):
            start_batch = time.time()
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
            end_batch = time.time()
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1} complete. Avg Loss: {avg_loss:.4f}")
    print("Training Complete")

    model.eval()   # turn off dropout, batchnorm, etc.
    correct = total = 0

    with torch.no_grad():  # disable autograd for faster inference
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

    torch.save(model.state_dict(), "resnet34_wikiart_final.pth")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTraining complete in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()