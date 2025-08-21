from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from PIL import Image
import io, json, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import data, model
from config import MODEL_PATH

class ModelOutput(BaseModel):
    prediction_first: str
    prediction_second: str
    prediction_third: str
    confidence: float
    uncertainty: str


app = FastAPI(
    title="Painting Art Style Classification Model",
    summary=("ResNet-34 on WikiArt with MC-Dropout; the thresholds are derived "
             "from validation-set entropy.")
)

transform_test = transforms.Compose([
    transforms.Resize(256, interpolation=InterpolationMode.BICUBIC, antialias=True),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

LE = data.load_le()
NUM_CLASSES = len(LE.classes_)

with open("data/data.json", "r") as f:
    THRESHOLDS = json.load(f)
LOW_T = float(THRESHOLDS["low"])
MED_T = float(THRESHOLDS["med"])

MODEL = model.create_model(num_classes=NUM_CLASSES, device=DEVICE, pretrained=False)
MODEL = model.load_state(MODEL, checkpoint_path=MODEL_PATH, device=DEVICE)
MODEL.eval()

def mc_dropout_eval(img, model, device, samples=50):
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

@app.get("/", description="Root endpoint that redirects to documentation.")
async def root():
    return RedirectResponse(url="/docs")

@app.post(
    "/classification",
    summary="Art style classification",
    description="Upload an image (jpg, jpeg, png) to classify style with uncertainty via MC-Dropout.",
    response_model=ModelOutput
)
async def classification(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/jpg", "image/png"}:
        raise HTTPException(status_code=415, detail="Unsupported file type. Use jpg/jpeg/png.")

    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    pred_idx, entropy, mean_probs = mc_dropout_eval(img, MODEL, DEVICE, samples=50)

    top_probs, top_idxs = mean_probs.topk(k=min(3, NUM_CLASSES))
    top_probs = top_probs.squeeze(0).tolist()
    top_idxs = top_idxs.squeeze(0).tolist()

    ent_norm = float(entropy.item() / math.log(NUM_CLASSES))

    if ent_norm <= LOW_T:
        unc = "LOW"
    elif ent_norm <= MED_T:
        unc = "MEDIUM"
    else:
        unc = "HIGH"

    return ModelOutput(
        prediction_first = LE.classes_[top_idxs[0]],
        prediction_second = LE.classes_[top_idxs[1]],
        prediction_third = LE.classes_[top_idxs[2]],
        confidence = float(top_probs[0]),
        uncertainty = unc
    )
