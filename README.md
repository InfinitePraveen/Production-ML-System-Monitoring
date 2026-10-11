# Production ML System + Monitoring

A notebook-led MLOps portfolio project that demonstrates a small production-style machine-learning lifecycle: model training, versioned artifacts, a Flask web interface/API, prediction logging, drift checks, Docker packaging, and CI quality gates.

**Dataset:** Breast Cancer Wisconsin Diagnostic dataset provided by scikit-learn. It is small, open, and loads directly from the installed package. This is an educational classification example and is not intended for medical use.

## Repository structure

```text
production-ml-system-monitoring/
├── .github/workflows/ci.yml
├── artifacts/
├── notebooks/                 # exactly 13 notebooks
├── app.py                     # Flask UI + API
├── test_app.py
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── README.md
├── CONTRIBUTE.md
└── CHANGELOG.md
```

No `src/`, preprocessing folder, or separate helper/module scripts are used. The notebooks are the learning and experimentation workflow; `app.py` is the required web application.

## Run the web app locally

Python 3.11 is recommended.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000. The dashboard includes the GitHub and LinkedIn profile links.

- `GET /health` — health and model version metadata.
- `POST /predict-demo` — small web-form demonstration.
- `POST /predict` — full JSON prediction endpoint. Supply all 30 feature names under `features`.

Example request shape:

```json
{
  "features": {
    "mean radius": 14.2,
    "mean texture": 20.1
  }
}
```

The example above intentionally omits most fields and will return a 400 response listing missing features. For a valid API request, include every feature name; the dataset's feature names can be printed with `load_breast_cancer(as_frame=True).data.columns.tolist()`.

The app trains a small fallback model on startup if `artifacts/model_v1.joblib` does not exist, so the UI is easy to demonstrate immediately. Run notebooks 01–05 to create a versioned artifact and registry metadata. Generated artifacts and logs are local demo outputs and are not included in version control except for the empty `artifacts/.gitkeep`.

## Notebook order

1. `01_problem_and_project_setup.ipynb` — scope and acceptance criteria
2. `02_data_understanding.ipynb` — dataset and class balance
3. `03_data_validation_and_split.ipynb` — schema checks and split
4. `04_baseline_training_and_evaluation.ipynb` — model, metrics, artifact
5. `05_reproducibility_and_model_versioning.ipynb` — registry metadata
6. `06_prediction_logging_and_batch_inference.ipynb` — inference logs
7. `07_drift_detection_baseline.ipynb` — PSI drift example
8. `08_evidently_data_drift_report.ipynb` — optional Evidently report
9. `09_flask_api_contract_and_smoke_test.ipynb` — API behavior
10. `10_monitoring_alerts_and_thresholds.ipynb` — simulated alert
11. `11_docker_and_deployment_validation.ipynb` — container workflow
12. `12_cicd_quality_gates.ipynb` — CI and release gates
13. `13_bentoml_serving_and_runbook.ipynb` — serving concepts and runbook

## Docker

```bash
docker build -t production-ml-monitoring .
docker run --rm -p 5000:5000 production-ml-monitoring
```

Then open http://127.0.0.1:5000.

## Tests and CI

```bash
pytest -q
```

GitHub Actions runs basic Python checks, tests, and a Docker build on pushes and pull requests.

## Monitoring, model versions, and alerts

- Model artifacts use versioned names such as `model_v1.joblib`; do not overwrite a previously released model.
- `model_registry.json` records the active model version and environment metadata.
- Prediction requests append basic output metadata to `artifacts/prediction_log.csv`.
- Notebook 07 demonstrates PSI; notebook 08 optionally exports an Evidently report.
- Notebook 10 shows a threshold-based alert decision.

Evidently is evolving quickly, so its report API can vary by release. The notebook catches optional dependency/API issues and points back to the lightweight PSI baseline. Confirm the installed version and validate upgrades before production use.

## Production caveats

This is an interview demonstration, not a turnkey production platform. Before real deployment, add authentication and authorization, secure secret handling, durable storage, structured logs, request limits, model approval gates, dependency/security scanning, and a monitored alert destination. Drift is not itself proof of model failure; use delayed ground-truth labels to track quality where possible.

## Author

- GitHub: https://github.com/InfinitePraveen
- LinkedIn: https://www.linkedin.com/in/infinitepraveen/

See [CONTRIBUTE.md](CONTRIBUTE.md) and [CHANGELOG.md](CHANGELOG.md).
