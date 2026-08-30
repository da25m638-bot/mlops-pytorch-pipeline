

import io
import logging
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import transforms

from model import get_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("serve")


app = FastAPI(title="CIFAR-10 Classifier", version="1.0.0")

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


_state: dict = {"model": None, "device": None, "loaded": False}


def _inference_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616],
        ),
    ])


def _load_model() -> bool:
    checkpoint_dir = Path(os.getenv("CHECKPOINT_DIR", "/app/checkpoints"))
    model_name = os.getenv("MODEL_NAME", "classifier_v1.pt")
    architecture = os.getenv("MODEL_ARCHITECTURE", "resnet18")
    num_classes = int(os.getenv("NUM_CLASSES", "10"))

    checkpoint_path = checkpoint_dir / model_name
    if not checkpoint_path.exists():
        logger.error("Checkpoint not found: %s", checkpoint_path)
        return False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = get_model(architecture=architecture, num_classes=num_classes)
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device).eval()

        _state["model"] = model
        _state["device"] = device
        _state["loaded"] = True

        logger.info(
            "Model loaded | path=%s | epoch=%s | val_acc=%.4f | device=%s",
            checkpoint_path,
            ckpt.get("epoch", "?"),
            ckpt.get("val_accuracy", float("nan")),
            device,
        )
        return True
    except Exception as exc:
        logger.exception("Failed to load model: %s", exc)
        return False



@app.on_event("startup")
async def startup() -> None:
    logger.info("Server starting – loading model …")
    _load_model()


@app.get("/health")
async def health():
    """Kubernetes liveness & readiness probe."""
    if _state["loaded"]:
        return JSONResponse({"status": "healthy", "model_loaded": True})
    raise HTTPException(status_code=503, detail="Model not loaded")


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    """
    Accept a PNG/JPEG image and return CIFAR-10 class probabilities.

    Example:
        curl -X POST http://localhost:8080/predict -F "image=@cat.png"
    """
    if not _state["loaded"]:
        raise HTTPException(status_code=503, detail="Model not loaded")

    content_type = image.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (PNG/JPEG)")

    try:
        raw = await image.read()
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")

        tensor = _inference_transforms()(pil_img).unsqueeze(0).to(_state["device"])

        with torch.no_grad():
            logits = _state["model"](tensor)
            probs = F.softmax(logits, dim=1).squeeze(0)

        top_idx = int(probs.argmax())
        return JSONResponse({
            "predicted_class": CIFAR10_CLASSES[top_idx],
            "confidence": round(probs[top_idx].item(), 4),
            "class_probabilities": {
                CIFAR10_CLASSES[i]: round(probs[i].item(), 4)
                for i in range(len(CIFAR10_CLASSES))
            },
        })

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
