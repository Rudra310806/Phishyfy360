
from pathlib import Path
import joblib

MODEL_PATH = Path("models/url_model.joblib")

model = joblib.load(MODEL_PATH)

test_urls = [
    "https://www.google.com/",
    "https://www.wikipedia.org/",
    "http://paypal-login-security-check.example/",
]

for url in test_urls:
    prediction = int(model.predict([url])[0])
    probabilities = model.predict_proba([url])[0]

    label = "LIKELY PHISHING" if prediction == 1 else "LIKELY LEGITIMATE"
    confidence = float(probabilities[prediction]) * 100

    print(f"\nURL: {url}")
    print(f"Prediction: {label}")
    print(f"Model score: {confidence:.2f}%")
