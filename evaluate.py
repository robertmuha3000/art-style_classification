from data import load_csv, load_le, preprocessing, create_dataloader
from model import load_state, create_model
from config import MODEL_PATH
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import json


def mc_dropout(model: nn.Module, device: torch.device, test_loader: DataLoader, samples=50) -> tuple:
    """
    Performs Monte Carlo Dropout to estimate predictive uncertainty by running multiple stochastic forward passes.
    Returns predicted classes, entropy values, and mean class probabilities.
    """
    model.to(device)
    model.eval()
    y_true = []
    with torch.no_grad():
        for _, labels in test_loader:
            y_true.extend(labels.cpu().numpy())

    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()
    all_preds = []
    with torch.no_grad():
        for sample in range(samples):
            per_pass_probs = []
            for X, y in tqdm(test_loader, desc=f"Sample {sample + 1}", leave=False):
                X = X.to(device)
                logits = model(X)
                probs = F.softmax(logits, dim=1)
                per_pass_probs.append(probs.cpu())
            per_pass_probs = torch.cat(per_pass_probs, dim=0)
            all_preds.append(per_pass_probs)
    all_probs = torch.stack(all_preds, dim=0)
    mean_probs = all_probs.mean(dim=0)
    pred = mean_probs.argmax(dim=1)
    entropy = -(mean_probs * (mean_probs.clamp_min(1e-12)).log()).sum(dim=1)
    y_true = np.array(y_true)
    mc_accuracy = (pred.numpy() == y_true).mean()
    print(f"MC accuracy: {mc_accuracy*100:.2f}%")
    model.eval()
    return pred, entropy, mean_probs


def main_evaluation(model: nn.Module, device: torch.device, test_loader: DataLoader, le: LabelEncoder):
    """
    Evaluates model accuracy on a dataset and prints per-class and overall accuracy statistics.
    Uses a LabelEncoder to map predictions to class names.
    """
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

    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    f1_macro = f1_score(y_true, y_pred, average="macro")
    print(f"Weighted F1: {f1_weighted*100:.2f}%")
    print(f"Macro F1: {f1_macro*100:.2f}%")

def evaluation():
    """
    Loads the trained model and dataset splits, then evaluates on both validation and test sets.
    Also runs MC Dropout to compute uncertainty thresholds for later use in prediction visualization.
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    df = load_csv()
    le = load_le()
    df_clean = preprocessing(df, le, fit=False)
    if "val" not in df_clean["subset"].unique():
        raise ValueError("Run training first before the evaluation!")
    df_val = df_clean[df_clean["subset"] == "val"].reset_index(drop=True)
    df_test = df_clean[df_clean["subset"] == "test"].reset_index(drop=True)
    val_loader = create_dataloader(df_val, train=False)
    test_loader = create_dataloader(df_test, train=False)
    num_classes = len(le.classes_)
    model = create_model(num_classes=num_classes, device=device, pretrained=False)
    model = load_state(model, checkpoint_path=MODEL_PATH, device=device)
    print("**Validation set evaluation**")
    main_evaluation(model, device, val_loader, le)
    print("**Test set evaluation**")
    main_evaluation(model, device, test_loader, le)
    pred, entropy, mean_probs = mc_dropout(model, device, val_loader)
    print("\nMC Dropout (first 10 items):")
    print("Top-1 preds:", pred[:10].tolist())
    print("Entropy   :", entropy[:10].tolist())
    entropy_vals = entropy.numpy()
    print(f"Entropy — min: {entropy_vals.min():.4f}, max: {entropy_vals.max():.4f}, mean: {entropy_vals.mean():.4f}, std: {entropy_vals.std():.4f}")

    low_thresh = np.percentile(entropy_vals, 33)
    med_thresh = np.percentile(entropy_vals, 66)
    print(f"Suggested cutoffs -> Low/Med: {low_thresh:.4f}, Med/High: {med_thresh:.4f}")
    import math
    low_thresh_norm = low_thresh / math.log(len(le.classes_))
    med_thresh_norm = med_thresh / math.log(len(le.classes_))
    print(f"Normalized thresholds -> Low/Med: {low_thresh_norm:.4f}, Med/High: {med_thresh_norm:.4f}")
    streamlit_thresh = {
    "low": float(low_thresh_norm),
    "med": float(med_thresh_norm)
    }
    with open('data/data.json', 'w') as f:
        json.dump(streamlit_thresh, f, indent=4)



if __name__ == "__main__":
    evaluation()
