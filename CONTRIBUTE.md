# Contribution Guide

Thanks for considering a contribution.

## Local setup
1. Use Python 3.11 or a compatible version.
2. Create and activate a virtual environment.
3. Install dependencies with `pip install -r requirements.txt`.
4. Run checks with `pytest -q`.
5. Start the demo using `python app.py`.

## How to contribute
- Keep changes focused and explain the reason in the pull request.
- For model changes, report the dataset split, random seed, and evaluation metrics.
- Do not commit secrets, private data, or large generated files.
- Add or update tests when changing API behavior.
- Keep previous model artifacts when introducing a new version.

## Pull requests
Include a concise summary, testing performed, screenshots for UI changes when useful, and any limitations or follow-up work.
