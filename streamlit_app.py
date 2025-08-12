import streamlit as st
from PIL import Image
import torch
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import pickle
import torch.nn as nn
import torch.nn.functional as F

LABEL_ENCODER_PATH = "label_encoder.pkl"
MODEL_PATH = "resnet34_wikiart_third_final.pth"

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
    model.fc = nn.Linear(model.fc.in_features, num_classes)
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

def evaluate(img, device, model, topk=3):
    with torch.no_grad():
        image = transform_test(img).unsqueeze(0).to(device)
        logits = model(image)
        probs = torch.softmax(logits, dim=1)[0]
        top_probs, top_idxs = probs.topk(min(topk, probs.shape[0]))

    return top_probs.cpu().tolist(), top_idxs.cpu().tolist()

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
                top_probs, top_idxs = evaluate(img, device, model)

            st.subheader("Predictions")
            for p, idx in zip(top_probs, top_idxs):
                    st.write(f"**{le.classes_[idx]}** — {p * 100:.2f}%")

if __name__ == "__main__":
    main()
