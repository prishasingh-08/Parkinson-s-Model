from pathlib import Path
from io import BytesIO

import numpy as np
import torch
from flask import Flask, render_template, request
from PIL import Image, ImageOps, UnidentifiedImageError
from torch import nn
from torchvision import models

ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = ROOT / "artifacts" / "resnet18_spiral.pth"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


def load_model():
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"ResNet18 checkpoint not found: {CHECKPOINT_PATH}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "resnet18":
        raise ValueError("Expected a ResNet18 model checkpoint.")
    if checkpoint.get("class_names") != ["Healthy", "Parkinson"]:
        raise ValueError("Checkpoint labels must be ordered Healthy, Parkinson.")

    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(nn.Dropout(0.35), nn.Linear(model.fc.in_features, 2))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    size = int(checkpoint.get("input_size", 224))
    mean = np.asarray(checkpoint.get("mean", (0.485, 0.456, 0.406)), dtype=np.float32)
    std = np.asarray(checkpoint.get("std", (0.229, 0.224, 0.225)), dtype=np.float32)
    if size != 224 or mean.shape != (3,) or std.shape != (3,) or np.any(std <= 0):
        raise ValueError("Unexpected image preprocessing metadata in checkpoint.")
    return model, size, mean, std


MODEL, IMAGE_SIZE, IMAGE_MEAN, IMAGE_STD = load_model()
print(f"Loaded ResNet18: {CHECKPOINT_PATH}")
print("Class mapping: 0=Healthy, 1=Parkinson")
print("Preprocessing: RGB, 224x224, ImageNet normalization; inference on CPU")


def analyze_image(payload):
    try:
        with Image.open(BytesIO(payload)) as source:
            if source.format not in {"JPEG", "PNG"}:
                raise ValueError("Please upload a JPG or PNG image.")
            source.load()
            image = ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("That file is not a readable JPG or PNG image.") from exc

    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    pixels = (pixels - IMAGE_MEAN) / IMAGE_STD
    tensor = torch.from_numpy(pixels.transpose(2, 0, 1)).unsqueeze(0)

    with torch.inference_mode():
        logits = MODEL(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    # The checkpoint records [Healthy, Parkinson], matching labels 0 and 1.
    healthy_score = float(probabilities[0].item())
    parkinson_score = float(probabilities[1].item())
    predicted_class = int(torch.argmax(probabilities).item())
    return predicted_class, healthy_score, parkinson_score


def render_result(*, prediction=None, probability=None, error=None, status=200):
    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        error=error,
        page="spiral",
        model_name="ResNet18",
        input_size=IMAGE_SIZE,
    ), status


@app.errorhandler(413)
def too_large(_error):
    return render_result(error="The image is too large. Please use a file under 25 MB.", status=413)


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "GET":
        return render_template("index.html", prediction=None, probability=None, error=None, page="dashboard", model_name="ResNet18", input_size=IMAGE_SIZE)

    uploaded = request.files.get("spiral")
    if uploaded is None or uploaded.filename == "":
        return render_result(error="Please choose a spiral image.")

    payload = uploaded.stream.read(MAX_UPLOAD_BYTES + 1)
    if not payload:
        return render_result(error="The selected file is empty.")
    if len(payload) > MAX_UPLOAD_BYTES:
        return render_result(error="The image is too large. Please use a file under 25 MB.", status=413)

    try:
        predicted_class, healthy_score, parkinson_score = analyze_image(payload)
    except ValueError as exc:
        return render_result(error=str(exc))
    except Exception as exc:
        app.logger.exception("ResNet18 inference failed")
        return render_result(error=f"Could not analyze this image: {exc}", status=500)

    print(f"Healthy class score: {healthy_score:.6f}; Parkinson class score: {parkinson_score:.6f}")
    prediction = (
        "Parkinson's pattern detected by the model"
        if predicted_class == 1
        else "Healthy pattern detected by the model"
    )
    return render_result(prediction=prediction, probability=parkinson_score)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
