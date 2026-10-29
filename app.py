
from pathlib import Path

import joblib
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

URL_MODEL_PATH = MODEL_DIR / "url_model.joblib"
HTML_MODEL_PATH = MODEL_DIR / "html_model.joblib"
IMAGE_MODEL_PATH = MODEL_DIR / "efficientnet_b0_phish360.pth"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

DEVICE = torch.device("cpu")
IMAGE_SIZE = 224

# Load the two scikit-learn pipelines.
url_model = joblib.load(URL_MODEL_PATH)
html_model = joblib.load(HTML_MODEL_PATH)

# Rebuild the same EfficientNet-B0 architecture used during training.
checkpoint = torch.load(
    IMAGE_MODEL_PATH,
    map_location=DEVICE,
    weights_only=True,
)

image_model = models.efficientnet_b0(weights=None)
feature_count = image_model.classifier[1].in_features
image_model.classifier[1] = nn.Linear(feature_count, 2)
image_model.load_state_dict(checkpoint["model_state_dict"])
image_model.to(DEVICE)
image_model.eval()

image_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def get_prediction(model, text):
    probabilities = model.predict_proba([text])[0]
    predicted_class = int(probabilities.argmax())

    return {
        "label": "Phishing" if predicted_class == 1 else "Legitimate",
        "confidence": float(probabilities[predicted_class]) * 100,
        "phishing_probability": float(probabilities[1]) * 100,
    }


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        analysis_type = request.form.get("analysis_type", "")

        try:
            if analysis_type == "url":
                url = request.form.get("url", "").strip()

                if not url:
                    raise ValueError("Enter a URL first.")

                result = get_prediction(url_model, url)
                result["type"] = "URL Analysis"
                result["input_preview"] = url[:300]

            elif analysis_type == "html":
                html_text = request.form.get("html_text", "").strip()
                html_file = request.files.get("html_file")

                if html_file and html_file.filename:
                    html_text = html_file.read().decode(
                        "utf-8", errors="replace"
                    ).strip()

                if not html_text:
                    raise ValueError(
                        "Paste HTML source or upload an HTML file."
                    )

                result = get_prediction(html_model, html_text)
                result["type"] = "HTML Analysis"
                result["input_preview"] = html_text[:300]

            elif analysis_type == "image":
                uploaded_image = request.files.get("screenshot")

                if not uploaded_image or not uploaded_image.filename:
                    raise ValueError("Choose a screenshot first.")

                try:
                    with Image.open(uploaded_image.stream) as image:
                        image = image.convert("RGB")
                        tensor = image_transform(image).unsqueeze(0)
                except (UnidentifiedImageError, OSError, ValueError):
                    raise ValueError(
                        "That file could not be opened as an image."
                    )

                with torch.no_grad():
                    outputs = image_model(tensor.to(DEVICE))
                    probabilities = torch.softmax(outputs, dim=1)[0]
                    predicted_class = int(probabilities.argmax())

                result = {
                    "type": "Screenshot Analysis",
                    "label": (
                        "Phishing" if predicted_class == 1
                        else "Legitimate"
                    ),
                    "confidence": float(
                        probabilities[predicted_class]
                    ) * 100,
                    "phishing_probability": float(
                        probabilities[1]
                    ) * 100,
                    "input_preview": uploaded_image.filename,
                }

            else:
                raise ValueError("Choose a valid analysis type.")

        except (ValueError, UnicodeError) as exc:
            error = str(exc)
        except Exception as exc:
            app.logger.exception("Prediction failed")
            error = (
                "Analysis failed. Check the terminal for details: "
                f"{type(exc).__name__}: {exc}"
            )

    return render_template("index.html", result=result, error=error)


@app.errorhandler(413)
def file_too_large(_error):
    return render_template(
        "index.html",
        result=None,
        error="File too large. Maximum upload size is 10 MB.",
    ), 413


if __name__ == "__main__":
    print("Phishify360 models loaded successfully.")
    print("Open http://127.0.0.1:5000 in your browser.")
    app.run(debug=False, host="127.0.0.1", port=5000)
