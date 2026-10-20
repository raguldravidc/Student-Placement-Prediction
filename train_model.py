"""
Train and evaluate the Student Placement Prediction model.
If dataset.csv is the included sample, results are illustrative because its labels are synthetic.
Replace it with your approved project dataset using the same columns before final submission.
"""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset.csv"
MODEL_PATH = BASE_DIR / "placement_model.joblib"
METRICS_PATH = BASE_DIR / "metrics.json"

FEATURES = [
    "CGPA", "Aptitude_Score", "Technical_Score",
    "Communication_Score", "Internships", "Projects", "Backlogs"
]
TARGET = "Placed"

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "dataset.csv was not found. Upload your placement dataset and save it as dataset.csv."
        )

    df = pd.read_csv(DATA_PATH)
    missing_columns = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    df = df[FEATURES + [TARGET]].copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET].isin([0, 1])]
    if df[TARGET].nunique() != 2:
        raise ValueError("Target 'Placed' must contain both classes: 0 and 1.")

    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    y = df[TARGET].astype(int)
    if y.value_counts().min() < 2:
        raise ValueError("Need at least two examples in each target class for a stratified split.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # All preprocessing is fitted on training data only, preventing data leakage.
    pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "dataset_rows_after_target_cleaning": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)) if y_test.nunique() == 2 else None,
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(
            y_test, predictions, target_names=["Not Placed", "Placed"],
            output_dict=True, zero_division=0
        ),
        "features": FEATURES,
        "target": TARGET,
        "note": "If using the bundled dataset.csv, these are synthetic demonstration results, not real placement evidence."
    }

    # Save a single object containing preprocessing + model, to keep inference consistent.
    joblib.dump({"pipeline": pipeline, "features": FEATURES, "target": TARGET}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Model saved to:", MODEL_PATH)
    print("Metrics saved to:", METRICS_PATH)
    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))

if __name__ == "__main__":
    main()
