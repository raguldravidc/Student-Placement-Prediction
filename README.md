# Student Placement Prediction System — Week 4 Capstone

## Overview
This project demonstrates an end-to-end machine learning workflow and a FastAPI prediction service. It predicts a binary placement label (`Placed = 1`, `Not Placed = 0`) from academic and skill-related features.

**Important dataset note:** the bundled `dataset.csv` is a reproducible synthetic demonstration dataset. It is not real student data and must not be described as real placement evidence. For the final internship submission, replace it with your approved original dataset and rerun training. If your original feature names differ, update `FEATURES`, the API schema, and this README consistently.

## Features
- Dataset validation and target cleaning
- Train/test split with stratification
- Missing-value imputation and scaling inside a scikit-learn Pipeline
- Logistic Regression classifier
- Accuracy, precision, recall, F1, ROC-AUC, confusion matrix and classification report
- Joblib serialization of the entire preprocessing + model pipeline
- FastAPI endpoints: `/`, `/health`, and `POST /predict`
- Interactive API docs at `/docs`

## Project structure
```text
Student_Placement_Prediction_Week4/
├── app.py
├── train_model.py
├── dataset.csv
├── requirements.txt
├── README.md
├── metrics.json          # created after training
└── placement_model.joblib # created after training
```

## Features and target
- `CGPA` (0–10)
- `Aptitude_Score` (0–100)
- `Technical_Score` (0–100)
- `Communication_Score` (0–100)
- `Internships` (count)
- `Projects` (count)
- `Backlogs` (count)
- Target: `Placed` (0 or 1)

## Run locally
Use Python 3.10+ (Python 3.11 is a safe choice for a clean environment).

```bash
python -m venv .venv
```
Activate the environment, then:
```bash
pip install -r requirements.txt
python train_model.py
uvicorn app:app --host 127.0.0.1 --port 8000
```
Open `http://127.0.0.1:8000/docs`.

## Google Colab workflow
1. Upload `dataset.csv`, `train_model.py`, `app.py`, and `requirements.txt` into Colab.
2. Run `!pip install -r requirements.txt`.
3. Run `!python train_model.py`.
4. Start the API in a separate cell with `!uvicorn app:app --host 0.0.0.0 --port 8000`. Colab's runtime is temporary; for API calls from outside the notebook, use an approved tunnel or deploy the repository to a hosting service. Do not publish API keys or tunnel tokens.
5. Test `/health` and `/predict`, and capture screenshots.

## Example API request
`POST /predict` JSON body:
```json
{
  "CGPA": 8.2,
  "Aptitude_Score": 75,
  "Technical_Score": 80,
  "Communication_Score": 78,
  "Internships": 1,
  "Projects": 2,
  "Backlogs": 0
}
```
Example response shape:
```json
{
  "prediction": "Placed",
  "placed": 1,
  "placement_probability": 0.82,
  "notice": "This is an educational estimate, not a guarantee of an actual placement outcome."
}
```
The example probability is illustrative; actual values depend on the dataset and trained model.

## Deployment
For a stable public URL, push the code to GitHub and deploy it to a Python hosting service that supports a web service. Set the build/install command to `pip install -r requirements.txt` and the start command to `uvicorn app:app --host 0.0.0.0 --port $PORT` if that platform supports `$PORT`. Ensure the model artifact is available at startup: either commit the trained `placement_model.joblib` if permitted, or configure a build step to run `python train_model.py`. Check the platform's current pricing, sleep behavior, file persistence, and free-tier limits before choosing it.

## Evaluation and limitations
Review `metrics.json` after training. Do not copy metrics from an earlier presentation unless they are reproduced with the current dataset, code, and test split. A model trained on synthetic labels only demonstrates the technical pipeline; it cannot establish real-world placement accuracy. Placement predictions should not be used to make high-stakes decisions about students.

## Troubleshooting
- `Model is not loaded`: run `python train_model.py` before starting FastAPI.
- Missing columns: make sure the CSV headers match the feature names above.
- Only one target class: check the `Placed` values; both 0 and 1 are required.
- API validation error: ensure numeric fields are in the stated ranges and JSON keys match exactly.
