import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, render_template_string, request
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)
MODEL_PATH = ARTIFACTS / "model_v1.joblib"
REGISTRY_PATH = ARTIFACTS / "model_registry.json"
LOG_PATH = ARTIFACTS / "prediction_log.csv"

app = Flask(__name__)
data = load_breast_cancer(as_frame=True)
FEATURES = list(data.data.columns)
TARGET_NAMES = list(data.target_names)

def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    # Safe fallback so the demo starts before notebooks have been run.
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
    ])
    model.fit(data.data, data.target)
    return model

model = load_model()

def model_version():
    if REGISTRY_PATH.exists():
        try:
            return json.loads(REGISTRY_PATH.read_text(encoding="utf-8")).get("version", "1.0.0")
        except (ValueError, OSError):
            pass
    return "1.0.0"

def log_prediction(prediction, probability):
    new_file = not LOG_PATH.exists()
    with LOG_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "timestamp", "model_version", "prediction", "positive_class_probability"
        ])
        if new_file:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_version": model_version(),
            "prediction": int(prediction),
            "positive_class_probability": float(probability),
        })

PAGE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Production ML System Monitor</title>
<style>
:root{color-scheme:dark;--bg:#0b1220;--panel:#121d30;--line:#29374e;--text:#edf3fc;--muted:#a9b8cc;--accent:#7dd3fc}
*{box-sizing:border-box}body{margin:0;background:linear-gradient(140deg,#0b1220,#14263a);font:16px/1.55 system-ui,Segoe UI,sans-serif;color:var(--text)}
main{max-width:1050px;margin:36px auto;padding:0 20px}.top{display:flex;justify-content:space-between;gap:20px;align-items:center;flex-wrap:wrap}
h1{font-size:clamp(28px,4vw,42px);line-height:1.1;margin:0 0 10px}.muted{color:var(--muted)}.pill{border:1px solid #2f6177;border-radius:999px;padding:7px 12px;color:var(--accent);font-size:13px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:16px;margin:24px 0}.card{background:#121d30dd;border:1px solid var(--line);border-radius:18px;padding:22px;box-shadow:0 14px 40px #0002}
h2{font-size:19px;margin-top:0}label{display:block;margin:12px 0 5px;color:var(--muted);font-size:14px}
input{width:100%;padding:11px;border-radius:9px;border:1px solid var(--line);background:#0b1424;color:var(--text)}
button{margin-top:18px;padding:12px 16px;border:0;border-radius:10px;background:var(--accent);color:#082032;font-weight:700;cursor:pointer}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}.links{display:flex;gap:16px;flex-wrap:wrap}.result{margin-top:15px;padding:14px;border-radius:10px;background:#0b1424;border:1px solid var(--line);display:none;white-space:pre-wrap}
.small{font-size:13px}.stat{font-size:24px;font-weight:700}.footer{margin:26px 0 40px}.status{color:#86efac}
</style></head>
<body><main>
<div class="top"><div><h1>Production ML System</h1><div class="muted">A compact MLOps demo: predictions, model versioning and monitoring.</div></div><span class="pill">Model v{{ version }}</span></div>
<div class="grid">
<section class="card"><h2>Service status</h2><div class="stat status">● Online</div><p class="muted small">Dataset: Breast Cancer Wisconsin (scikit-learn). The demo model predicts the dataset's encoded class.</p><a href="/health">View health JSON →</a></section>
<section class="card"><h2>Monitoring overview</h2><p>Predictions are appended to a local CSV log. The notebooks demonstrate feature drift checks, Evidently reports and alert thresholds.</p><p class="muted small">Drift is a signal to investigate, not proof of lower model quality.</p></section>
<section class="card"><h2>Project links</h2><div class="links"><a href="https://github.com/InfinitePraveen" target="_blank" rel="noreferrer">GitHub ↗</a><a href="https://www.linkedin.com/in/infinitepraveen/" target="_blank" rel="noreferrer">LinkedIn ↗</a></div><p class="muted small">Built as an educational portfolio project.</p></section>
</div>
<section class="card"><h2>Try a prediction</h2><p class="muted">Enter the first two features below. Remaining features use example median values so the form stays manageable. For the full API, send all 30 named features to <code>/predict</code>.</p>
<form id="predictForm"><div class="grid">
<div><label for="radius">Mean radius</label><input id="radius" type="number" step="any" value="14.2" required></div>
<div><label for="texture">Mean texture</label><input id="texture" type="number" step="any" value="20.1" required></div>
</div><button type="submit">Run prediction</button></form><div id="result" class="result"></div>
</section>
<div class="footer muted small">Demo only — not for medical use. Model output is a machine-learning example, not a diagnosis. Secure and harden before any real deployment.</div>
</main>
<script>
document.getElementById('predictForm').addEventListener('submit', async (event) => {
 event.preventDefault();
 const result = document.getElementById('result'); result.style.display='block'; result.textContent='Running prediction…';
 try {
   const response = await fetch('/predict-demo', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({radius:Number(document.getElementById('radius').value),texture:Number(document.getElementById('texture').value)})});
   const body = await response.json(); result.textContent = JSON.stringify(body,null,2);
 } catch (error) { result.textContent = 'Request failed: ' + error; }
});
</script></body></html>
"""

@app.get("/")
def home():
    return render_template_string(PAGE, version=model_version())

@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_version": model_version(), "feature_count": len(FEATURES)})

@app.post("/predict-demo")
def predict_demo():
    payload = request.get_json(silent=True) or {}
    row = data.data.median().to_dict()
    try:
        row[FEATURES[0]] = float(payload["radius"])
        row[FEATURES[1]] = float(payload["texture"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Provide numeric radius and texture values."}), 400
    frame = __import__("pandas").DataFrame([row], columns=FEATURES)
    prediction = int(model.predict(frame)[0])
    probability = float(model.predict_proba(frame)[0][1])
    log_prediction(prediction, probability)
    return jsonify({
        "prediction": TARGET_NAMES[prediction],
        "encoded_prediction": prediction,
        "positive_class_probability": round(probability, 4),
        "model_version": model_version(),
        "note": "Educational demo only; not a medical diagnosis."
    })

@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    features = payload.get("features")
    if not isinstance(features, dict):
        return jsonify({"error": "Expected JSON with a 'features' object."}), 400
    missing = [name for name in FEATURES if name not in features]
    if missing:
        return jsonify({"error": "Missing required features.", "missing_features": missing}), 400
    try:
        row = {name: float(features[name]) for name in FEATURES}
        frame = __import__("pandas").DataFrame([row], columns=FEATURES)
        if not np.isfinite(frame.to_numpy()).all():
            raise ValueError("Features must be finite numeric values.")
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    prediction = int(model.predict(frame)[0])
    probability = float(model.predict_proba(frame)[0][1])
    log_prediction(prediction, probability)
    return jsonify({
        "prediction": TARGET_NAMES[prediction],
        "encoded_prediction": prediction,
        "positive_class_probability": round(probability, 4),
        "model_version": model_version()
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
