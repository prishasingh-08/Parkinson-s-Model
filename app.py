from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from flask import Flask, render_template, request
from skimage.feature import hog

ROOT = Path(__file__).resolve().parent

app = Flask(__name__)

MODEL = joblib.load(ROOT / "artifacts" / "hog_svm.joblib")


def extract_features(image_path):
    image = np.array(
        Image.open(image_path)
        .convert("L")
        .resize((256, 256))
    ) / 255.0

    features = hog(
        image,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    return features.reshape(1, -1)



@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None
    error = None
    page = "spiral" if request.method == "POST" else "dashboard"

    if request.method == "POST":
        if "spiral" not in request.files:
            error = "Please upload a spiral image."
        else:
            file = request.files["spiral"]

            if file.filename == "":
                error = "Please choose an image."
            else:
                upload_dir = ROOT / "uploads"
                upload_dir.mkdir(exist_ok=True)

                image_path = upload_dir / file.filename
                file.save(image_path)

                try:
                    features = extract_features(image_path)
                    probability = float(
                        MODEL.predict_proba(features)[0, 1]
                    )

                    if probability >= 0.5:
                        prediction = "Parkinson's risk pattern detected"
                    else:
                        prediction = "Healthy pattern detected"

                except Exception as e:
                    error = f"Could not analyze image: {e}"

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        error=error,
        page=page
    )


if __name__ == "__main__":
    app.run(debug=True)