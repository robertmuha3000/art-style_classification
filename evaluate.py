from data import load_csv, load_le, preprocessing, create_dataloader
from model import load_state, create_model
from config import MODEL_PATH
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

def mc_dropout(model, device, test_loader, samples=50):
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



def main_evaluation(model, device, test_loader, le):
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

def evaluation():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    df = load_csv()
    le = load_le()
    df_clean = preprocessing(df, le, fit=False)
    df_test = df_clean[df_clean["subset"] == "test"].reset_index(drop=True)
    test_loader = create_dataloader(df_test, train=False)
    num_classes = len(le.classes_)
    model = create_model(num_classes=num_classes, device=device, pretrained=False)
    model = load_state(model, checkpoint_path=MODEL_PATH, device=device)
    main_evaluation(model, device, test_loader, le)
    pred, entropy, mean_probs = mc_dropout(model, device, test_loader)
    print("\nMC Dropout (first 10 items):")
    print("Top-1 preds:", pred[:10].tolist())
    print("Entropy   :", entropy[:10].tolist())
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.hist(entropy.numpy(), bins=30, color='skyblue', edgecolor='black')
    plt.xlabel("Predictive Entropy")
    plt.ylabel("Number of Samples")
    plt.title("MC Dropout Uncertainty Distribution")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    confidences = mean_probs.max(dim=1).values.numpy()

    plt.figure(figsize=(8, 5))
    plt.scatter(confidences, entropy.numpy(), alpha=0.5, s=10)
    plt.xlabel("Mean Confidence (Top-1 Probability)")
    plt.ylabel("Predictive Entropy")
    plt.title("Confidence vs Uncertainty")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


if __name__ == "__main__":
    evaluation()
