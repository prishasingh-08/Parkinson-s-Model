import base64
from pathlib import Path
from io import BytesIO
import json
import csv

import numpy as np
import torch
from flask import Flask, render_template, request
from PIL import Image, ImageOps, UnidentifiedImageError
from skimage.feature import hog
from torch import nn
from torchvision import models
import joblib

from gait_model import extract_gait_features, parse_gait_record
from voice_model import extract_voice_features

ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = ROOT / "artifacts" / "resnet18_spiral.pth"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
VOICE_MODEL_PATH = ROOT / "artifacts" / "voice_model.joblib"
GAIT_MODEL_PATH = ROOT / "artifacts" / "gait_model.joblib"
HOG_MODEL_PATH = ROOT / "artifacts" / "hog_svm.joblib"
VOICE_MODEL = joblib.load(VOICE_MODEL_PATH)["pipeline"] if VOICE_MODEL_PATH.is_file() else None
GAIT_MODEL = joblib.load(GAIT_MODEL_PATH)["pipeline"] if GAIT_MODEL_PATH.is_file() else None
HOG_MODEL = joblib.load(HOG_MODEL_PATH) if HOG_MODEL_PATH.is_file() else None


def read_accuracy(filename, fallback=0.5):
    path = ROOT / "artifacts" / filename
    try:
        value = float(json.loads(path.read_text(encoding="utf-8"))["accuracy"])
        return value if 0.0 <= value <= 1.0 else fallback
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return fallback


def read_csv_accuracy(filename, fallback=0.5):
    path = ROOT / "artifacts" / filename
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            row = next(csv.DictReader(handle))
        value = float(row["accuracy"])
        return value if 0.0 <= value <= 1.0 else fallback
    except (OSError, StopIteration, KeyError, TypeError, ValueError):
        return fallback


def spiral_validation_accuracy():
    # This value belongs to the checkpoint actually loaded at runtime. The
    # separate fold-prediction artifact evaluates different fold models.
    return read_accuracy("test_metrics.json")


SPIRAL_ACCURACIES = {
    "resnet18": spiral_validation_accuracy(),
    "hog_svm": read_csv_accuracy("hog_svm_metrics.csv"),
}
MODEL_ACCURACIES = {
    "Spiral analysis": SPIRAL_ACCURACIES["resnet18"],
    "Voice analysis": read_accuracy("voice_metrics.json"),
    "Gait analysis": read_accuracy("gait_metrics.json"),
}


def demo_histories():
    path = ROOT / "static" / "demo_histories.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


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


def analyze_hog_image(payload):
    if HOG_MODEL is None:
        raise FileNotFoundError(f"HOG+SVM model not found: {HOG_MODEL_PATH}")
    try:
        with Image.open(BytesIO(payload)) as source:
            if source.format not in {"JPEG", "PNG"}:
                raise ValueError("Please upload a JPG or PNG image.")
            source.load()
            image = ImageOps.exif_transpose(source).convert("L").resize((256, 256))
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("That file is not a readable JPG or PNG image.") from exc
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    features = hog(
        pixels, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
        block_norm="L2-Hys",
    ).reshape(1, -1)
    classes = list(HOG_MODEL.classes_)
    probability = float(HOG_MODEL.predict_proba(features)[0, classes.index(1)])
    predicted_class = int(HOG_MODEL.predict(features)[0])
    return predicted_class, probability


def render_result(*, page="spiral", prediction=None, probability=None, error=None, status=200, model_name="ResNet18", model_accuracy=None, model_accuracies=None, spiral_model="resnet18", spiral_preview=None):
    page_model_accuracy = MODEL_ACCURACIES.get({"spiral": "Spiral analysis", "voice": "Voice analysis", "gait": "Gait analysis"}.get(page, ""))
    if model_accuracy is not None:
        page_model_accuracy = model_accuracy
    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        spiral_preview=spiral_preview,
        error=error,
        page=page,
        model_name=model_name,
        input_size=IMAGE_SIZE,
        gait_available=GAIT_MODEL is not None,
        model_accuracy=page_model_accuracy,
        model_accuracies=model_accuracies or MODEL_ACCURACIES,
        spiral_accuracies=SPIRAL_ACCURACIES,
        spiral_model=spiral_model,
        demo_histories=demo_histories(),
    ), status


@app.errorhandler(413)
def too_large(_error):
    if request.path == "/analyze/voice":
        return render_result(page="voice", error="The recording is too large. Please use a file under 25 MB.", status=413, model_name="Voice research model")
    if request.path == "/analyze/gait":
        return render_result(page="gait", error="The gait file is too large. Please use a file under 25 MB.", status=413, model_name="Gait sensor research model")
    return render_result(error="The image is too large. Please use a file under 25 MB.", status=413)


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "GET":
        page = request.args.get("page", "dashboard")
        if page not in {"dashboard", "spiral", "gait", "voice", "history", "profile", "settings", "about"}:
            page = "dashboard"
        return render_template("index.html", prediction=None, probability=None, error=None, page=page, model_name="ResNet18", input_size=IMAGE_SIZE, gait_available=GAIT_MODEL is not None, model_accuracy=MODEL_ACCURACIES.get({"spiral": "Spiral analysis", "voice": "Voice analysis", "gait": "Gait analysis"}.get(page, "")), model_accuracies=MODEL_ACCURACIES, spiral_accuracies=SPIRAL_ACCURACIES, spiral_model="resnet18", demo_histories=demo_histories())

    selected_model = request.form.get("spiral_model", "resnet18")
    if selected_model not in SPIRAL_ACCURACIES:
        selected_model = "resnet18"
    model_name = "HOG+SVM" if selected_model == "hog_svm" else "ResNet18"
    model_accuracies = {**MODEL_ACCURACIES, "Spiral analysis": SPIRAL_ACCURACIES[selected_model]}
    uploaded = request.files.get("spiral")
    if uploaded is None or uploaded.filename == "":
        return render_result(error="Please choose a spiral image.", model_name=model_name, model_accuracy=SPIRAL_ACCURACIES[selected_model], model_accuracies=model_accuracies, spiral_model=selected_model)

    payload = uploaded.stream.read(MAX_UPLOAD_BYTES + 1)
    if not payload:
        return render_result(error="The selected file is empty.", model_name=model_name, model_accuracy=SPIRAL_ACCURACIES[selected_model], model_accuracies=model_accuracies, spiral_model=selected_model)
    if len(payload) > MAX_UPLOAD_BYTES:
        return render_result(error="The image is too large. Please use a file under 25 MB.", status=413, model_name=model_name, model_accuracy=SPIRAL_ACCURACIES[selected_model], model_accuracies=model_accuracies, spiral_model=selected_model)

    healthy_score = None
    try:
        if selected_model == "hog_svm":
            predicted_class, parkinson_score = analyze_hog_image(payload)
        else:
            predicted_class, _healthy_score, parkinson_score = analyze_image(payload)
    except ValueError as exc:
        return render_result(error=str(exc), model_name=model_name, model_accuracy=SPIRAL_ACCURACIES[selected_model], model_accuracies=model_accuracies, spiral_model=selected_model)
    except Exception as exc:
        app.logger.exception("ResNet18 inference failed")
        return render_result(error=f"Could not analyze this image: {exc}", status=500, model_name=model_name, model_accuracy=SPIRAL_ACCURACIES[selected_model], model_accuracies=model_accuracies, spiral_model=selected_model)

    if healthy_score is not None:
        print(f"Healthy class score: {healthy_score:.6f}; Parkinson class score: {parkinson_score:.6f}")
    prediction = (
        "Parkinson's pattern detected by the model"
        if predicted_class == 1
        else "Healthy pattern detected by the model"
    )
    spiral_preview = "data:image/png;base64," + base64.b64encode(payload).decode("ascii")
    return render_result(prediction=prediction, probability=parkinson_score, model_name=model_name, model_accuracy=SPIRAL_ACCURACIES[selected_model], model_accuracies=model_accuracies, spiral_model=selected_model, spiral_preview=spiral_preview)


@app.post("/analyze/voice")
def analyze_voice():
    if VOICE_MODEL is None:
        return render_result(page="voice", error="The voice model file is missing. Re-download the complete project ZIP.", status=503, model_name="Voice research model")
    uploaded = request.files.get("voice")
    if uploaded is None or not uploaded.filename:
        return render_result(page="voice", error="Record or choose a WAV audio file first.", model_name="Voice research model")
    payload = uploaded.stream.read(MAX_UPLOAD_BYTES + 1)
    if not payload or len(payload) > MAX_UPLOAD_BYTES:
        return render_result(page="voice", error="The audio file is empty or larger than 25 MB.", status=413, model_name="Voice research model")
    try:
        features = extract_voice_features(payload).reshape(1, -1)
        classes = list(VOICE_MODEL.classes_)
        probability = float(VOICE_MODEL.predict_proba(features)[0, classes.index(1)])
    except ValueError as exc:
        return render_result(page="voice", error=str(exc), model_name="Voice research model")
    except Exception:
        app.logger.exception("Voice inference failed")
        return render_result(page="voice", error="The recording could not be analyzed. Please use a clear WAV recording of a steady ‘ah’ sound.", status=500, model_name="Voice research model")
    prediction = "Parkinson-like voice pattern in this sample" if probability >= .5 else "Healthy-like voice pattern in this sample"
    return render_result(page="voice", prediction=prediction, probability=probability, model_name="Voice research model")


@app.post("/analyze/gait")
def analyze_gait():
    if GAIT_MODEL is None:
        return render_result(page="gait", error="The gait model has not been trained yet. This release includes the training script and source information, but the public gait files could not be retrieved in this build environment.", status=503, model_name="Gait sensor research model")
    uploaded = request.files.get("gait")
    if uploaded is None or not uploaded.filename:
        return render_result(page="gait", error="Choose a compatible gait sensor TXT file first.", model_name="Gait sensor research model")
    payload = uploaded.stream.read(MAX_UPLOAD_BYTES + 1)
    if not payload or len(payload) > MAX_UPLOAD_BYTES:
        return render_result(page="gait", error="The gait file is empty or larger than 25 MB.", status=413, model_name="Gait sensor research model")
    try:
        data = parse_gait_record(payload)
        features = extract_gait_features(data).reshape(1, -1)
        classes = list(GAIT_MODEL.classes_)
        probability = float(GAIT_MODEL.predict_proba(features)[0, classes.index(1)])
    except ValueError as exc:
        return render_result(page="gait", error=str(exc), model_name="Gait sensor research model")
    except Exception:
        app.logger.exception("Gait inference failed")
        return render_result(page="gait", error="The gait sensor file could not be analyzed.", status=500, model_name="Gait sensor research model")
    prediction = "Parkinson-like gait pattern in this sensor sample" if probability >= .5 else "Healthy-like gait pattern in this sensor sample"
    return render_result(page="gait", prediction=prediction, probability=probability, model_name="Gait sensor research model")


if __name__ == "__main__":
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
