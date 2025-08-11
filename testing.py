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
import torchvision.models as models
from collections import defaultdict
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

    transform_test = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
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
        "['Contemporary Realism']": "['Realism']",
        "['Fauvism']": "['Expressionism']"
    }

    print("!!!\n!!!")

    le = LabelEncoder()

    df_clean = df_clean[df_clean["genre"].isin(safe_genres)]
    df_clean["genre_merged"] = df_clean["genre"].replace(genre_merge_map)
    df_clean["encoded_genre"] = le.fit_transform(df_clean["genre_merged"])

    print(df_clean["encoded_genre"])


    test = df_clean[df_clean["subset"] == "test"]
    test_dataset = CustomDataset(data=test, img_dir=image_root, transform=transform_test)


    num_workers = max(os.cpu_count() - 1, 0)
    persistent_workers = num_workers > 0

    batch_size = 32
    test_loader = DataLoader(test_dataset, batch_size, shuffle = False, num_workers=num_workers, persistent_workers=persistent_workers)

    state = torch.load("resnet34_wikiart_second_final.pth", map_location=device)
    num_classes = state["fc.weight"].shape[0]       # infer from checkpoint

    model = models.resnet34(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(state)
    model.to(device)
    y_true, y_pred = [], []

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labels.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    num_classes = len(le.classes_)
    per_class_total = np.bincount(y_true, minlength=num_classes)
    per_class_correct = np.bincount(y_true[y_true == y_pred], minlength=num_classes)
    per_class_acc = per_class_correct / np.maximum(per_class_total, 1)

    print("\nPer-class accuracy:")
    for cls_idx, acc in sorted(enumerate(per_class_acc), key=lambda x: -x[1]):
        print(f"{le.classes_[cls_idx]:30s}  n={per_class_total[cls_idx]:4d}  acc={acc*100:6.2f}%")

    overall = (y_true == y_pred).mean()
    print(f"\nOverall accuracy: {overall*100:.2f}%")



if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
