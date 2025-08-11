from data import load_csv, load_le, preprocessing, create_dataloader
from model import load_state, create_model
from config import MODEL_PATH
import torch
import numpy as np

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


if __name__ == "__main__":
    evaluation()