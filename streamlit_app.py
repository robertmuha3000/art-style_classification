import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import pickle
import torch.nn as nn
import torch.nn.functional as F

LABEL_ENCODER_PATH = "label_encoder.pkl"
MODEL_PATH = "resnet34_wikiart_uncertainty.pth"

transform_test = transforms.Compose([
    transforms.Resize(256, interpolation=InterpolationMode.BICUBIC, antialias=True),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

@st.cache_resource
def load_le(path=LABEL_ENCODER_PATH):
    with open(path, 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def create_model(num_classes: int, device: torch.device, pretrained: bool = False):
    """
    Load ResNet34 from the same hub tag you used, replace the final layer,
    and move to device.
    """
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=pretrained)
    model.fc = nn.Sequential(nn.Dropout(p=0.25), nn.Linear(model.fc.in_features, num_classes))
    model = model.to(device)
    return model

def load_state(model, checkpoint_path: str, device: torch.device, strict: bool = True):
    """
    Load a state_dict saved with torch.save(model.state_dict(), ...).
    """
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state, strict=strict)
    model.to(device)
    model.eval()
    return model

def mc_dropout(img, model, device, samples=50):
    model.to(device)
    model.eval()

    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()
    all_preds = []
    with torch.no_grad():
        image = transform_test(img).unsqueeze(0).to(device)
        for _ in range(samples):
            logits = model(image)
            probs = F.softmax(logits, dim=1)
            all_preds.append(probs.cpu())
    all_probs = torch.stack(all_preds, dim=0)
    mean_probs = all_probs.mean(dim=0)
    pred = mean_probs.argmax(dim=1)
    entropy = -(mean_probs * (mean_probs.clamp_min(1e-12)).log()).sum(dim=1)

    model.eval()
    return pred, entropy, mean_probs

def main():
    st.title("🎨 Art Style Classifier")
    st.text("Upload an image of an art piece below!")

    uploaded_file = st.file_uploader("Upload a painting", type=["jpg","jpeg","png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded image", use_container_width=True)

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        le = load_le()
        num_classes = len(le.classes_)
        model = create_model(num_classes=num_classes, device=device, pretrained=False)
        model = load_state(model, checkpoint_path=MODEL_PATH, device=device)
        if st.button("Predict"):
            with st.spinner("Analyzing..."):
                pred, entropy, mean_probs = mc_dropout(img, model, device, samples= 50)
                mc_conf = mean_probs[0]
                mc_top_probs, mc_top_idxs = mc_conf.topk(3)

            st.subheader("Predictions")
            for p, idx in zip(mc_top_probs.tolist(), mc_top_idxs.tolist()):
                st.write(f"**{le.classes_[idx]}** — {p * 100:.2f}%")
            import math
            ent_norm = (entropy[0].item()) / math.log(len(le.classes_))
            conf = mc_conf.max().item()
            st.caption(f"Uncertainty (normalized entropy): {ent_norm:.2f}")
            if ent_norm <= 0.33:
                st.success(f"Confidence: {conf*100:.1f}% · Uncertainty: {ent_norm:.2f} (low)")
            elif ent_norm <= 0.66:
                st.warning(f"Confidence: {conf*100:.1f}% · Uncertainty: {ent_norm:.2f} (medium)")
            else:
                st.error(f"Confidence: {conf*100:.1f}% · Uncertainty: {ent_norm:.2f} (high)")

if __name__ == "__main__":
    main()
