import pandas as pd
import os
from PIL import Image
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, RandAugment
import pickle
from sklearn.model_selection import train_test_split
from config import CSV_PATH, IMAGE_ROOT, BATCH_SIZE, NUM_WORKERS, PERSISTENT_WORKERS, LABEL_ENCODER_PATH


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
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

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

def splits(df: pd.DataFrame, val_frac=0.15, test_frac=0.15, seed=42, strat_col="encoded_genre") -> pd.DataFrame:
    """
    Splits the dataset into train, validation, and test sets with stratification.
    Returns a copy of the DataFrame with a modified 'subset' column labeling each sample's split.
    """
    trainval_idx, test_idx = train_test_split(
        df.index, test_size=test_frac, stratify=df[strat_col], random_state=seed
    )
    val_rel = val_frac / (1.0 - test_frac)
    train_idx, val_idx = train_test_split(
        trainval_idx, test_size=val_rel, stratify=df.loc[trainval_idx, strat_col], random_state=seed
    )

    df = df.copy()
    df["subset"] = None
    df.loc[train_idx, "subset"] = "train"
    df.loc[val_idx, "subset"] = "val"
    df.loc[test_idx, "subset"] = "test"
    return df

def load_csv(csv_path=CSV_PATH) -> pd.DataFrame:
    """
    Loads the dataset CSV into a pandas DataFrame.
    """
    df = pd.read_csv(csv_path)
    return df

def preprocessing(df: pd.DataFrame, le: LabelEncoder, fit: bool) -> pd.DataFrame:
    """
    Cleans and filters the dataset, merges similar genres, and encodes labels.
    If 'fit' is True, fits the LabelEncoder, otherwise, transforms with an existing one.
    """
    pattern = r'^[a-zA-Z0-9\s.,!?;:\'\"@#&$%^*()_+\-=\[\]{}<>/\\|~`]*$'
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

    df_clean = df[df["filename"].str.match(pattern, na=False)]
    df_clean = df_clean[df_clean["genre"].isin(safe_genres)]
    df_clean["genre_merged"] = df_clean["genre"].replace(genre_merge_map)
    if fit:
        df_clean["encoded_genre"] = le.fit_transform(df_clean["genre_merged"])
    else:
        df_clean["encoded_genre"] = le.transform(df_clean["genre_merged"])

    return df_clean


def create_dataloader(df, train: bool, image_root=IMAGE_ROOT, batch_size= BATCH_SIZE, num_workers=NUM_WORKERS, persistent_workers=PERSISTENT_WORKERS) -> DataLoader:
    """
    Creates a PyTorch DataLoader for the given DataFrame split.
    Applies training or testing transformations depending on the 'train' flag.
    """
    if train:
        transform = transform_train
    else:
        transform = transform_test

    dataset = CustomDataset(data=df, img_dir=image_root, transform=transform)
    loader = DataLoader(dataset, batch_size, shuffle = train, num_workers=num_workers, persistent_workers=persistent_workers)
    return loader

def save_le(le, path=LABEL_ENCODER_PATH) -> None:
    """
    Saves the fitted LabelEncoder to disk using pickle.
    """
    with open(path, "wb") as f:
        pickle.dump(le, f)

def load_le(path=LABEL_ENCODER_PATH) -> LabelEncoder:
    """
    Loads a previously saved LabelEncoder from disk.
    """
    with open(path, 'rb') as f:
        return pickle.load(f)
