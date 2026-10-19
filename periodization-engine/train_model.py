from sklearn.impute import SimpleImputer
from pathlib import Path
import json
import re

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "raw" / "programs_detailed_boostcamp_kaggle.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "prescription_model.joblib"

# Limit training size so the first experiment is practical on a laptop.
MAX_TRAIN_ROWS = 120_000
RANDOM_STATE = 42

CATEGORICAL_FEATURES = [
    "goal",
    "experience",
    "equipment",
    "exercise_name",
]

NUMERIC_FEATURES = [
    "program_length",
    "time_per_workout",
    "week",
    "day",
]

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def normalize_goal(value):
    text = str(value).lower()

    if any(x in text for x in [
        "powerlifting", "powerbuilding", "strength"
    ]):
        return "strength"

    if any(x in text for x in [
        "bodybuilding", "hypertrophy", "muscle", "sculpting"
    ]):
        return "hypertrophy"

    if "athletics" in text:
        return "strength"

    return "general_fitness"


def normalize_experience(value):
    text = str(value).lower()

    if "beginner" in text or "novice" in text:
        return "beginner"
    if "intermediate" in text:
        return "intermediate"
    if "advanced" in text:
        return "advanced"

    return "unknown"


def normalize_equipment(value):
    text = str(value).lower()

    if "dumbbell" in text:
        return "dumbbell only"
    if "garage" in text:
        return "garage gym"
    if "home" in text:
        return "at home"
    if "gym" in text:
        return "full gym"

    return "unknown"


def number_from_value(value):
    """Extract a numeric value from values such as '8 weeks'."""
    if pd.isna(value):
        return np.nan

    match = re.search(r"\d+(?:\.\d+)?", str(value))
    return float(match.group()) if match else np.nan


def prepare_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.drop_duplicates().copy()

    required = [
        "title", "exercise_name", "sets", "reps",
        "goal", "level", "equipment",
    ]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    df["exercise_name"] = df["exercise_name"].fillna("").astype(str).str.strip()
    df["sets"] = pd.to_numeric(df["sets"], errors="coerce")
    df["reps"] = pd.to_numeric(df["reps"], errors="coerce")

    df = df[
        df["title"].notna()
        & df["exercise_name"].ne("")
        & df["sets"].between(1, 10)
    ].copy()

    # Negative reps represent timed prescriptions in this dataset.
    # Do not treat their absolute values as repetition counts.
    df["goal"] = df["goal"].map(normalize_goal)
    df["experience"] = df["level"].map(normalize_experience)
    df["equipment"] = df["equipment"].map(normalize_equipment)

    for col in ["program_length", "time_per_workout", "week", "day"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = df[col].map(number_from_value)

    df["reps"] = df["reps"].where(df["reps"] > 0, np.nan)

    # Restrict rep labels to a reasonable data-cleaning range.
    df["reps"] = df["reps"].where(df["reps"].between(1, 50), np.nan)

    df["title"] = df["title"].astype(str)

    return df



def build_pipeline():
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(
                        strategy="constant",
                        fill_value="unknown",
                    )),
                    ("encoder", OneHotEncoder(
                        handle_unknown="ignore"
                    )),
                ]),
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                ]),
                NUMERIC_FEATURES,
            ),
        ]
    )

    return Pipeline([
        ("preprocessing", preprocessing),
        ("model", RandomForestRegressor(
            n_estimators=60,
            min_samples_leaf=3,
            max_features=0.8,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )),
    ])


def train_target(df, target, train_indices, test_indices):
    usable = df.dropna(subset=[target]).copy()

    # The title-based split is prepared before this function.
    train = usable.loc[usable.index.intersection(train_indices)]
    test = usable.loc[usable.index.intersection(test_indices)]

    if len(train) < 100 or len(test) < 20:
        raise ValueError(
            f"Not enough rows to train/evaluate {target}. "
            f"Train={len(train)}, test={len(test)}"
        )

    # Sample training rows only. Keep the test set separate.
    if len(train) > MAX_TRAIN_ROWS:
        train = train.sample(
            n=MAX_TRAIN_ROWS,
            random_state=RANDOM_STATE,
        )

    model = build_pipeline()
    model.fit(train[FEATURES], train[target])

    predictions = model.predict(test[FEATURES])

    metrics = {
        "target": target,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "mae": round(float(mean_absolute_error(
            test[target], predictions
        )), 3),
        "rmse": round(float(np.sqrt(mean_squared_error(
            test[target], predictions
        ))), 3),
    }

    return model, metrics


def main():
    df = prepare_data()

    # Keep entire program titles together across train/test.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    train_pos, test_pos = next(
        splitter.split(df, groups=df["title"])
    )

    train_indices = df.index[train_pos]
    test_indices = df.index[test_pos]

    sets_model, sets_metrics = train_target(
        df, "sets", train_indices, test_indices
    )

    # Only positive, plausible repetition counts are used for this model.
    reps_model, reps_metrics = train_target(
        df, "reps", train_indices, test_indices
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    bundle = {
        "sets_model": sets_model,
        "reps_model": reps_model,
        "features": FEATURES,
        "metrics": {
            "sets": sets_metrics,
            "reps": reps_metrics,
        },
        "training_description": (
            "Random Forest models trained on program exercise records. "
            "They predict dataset prescription patterns, not athlete outcomes."
        ),
    }

    joblib.dump(bundle, MODEL_PATH)

    metrics_path = MODEL_DIR / "training_metrics.json"
    metrics_path.write_text(
        json.dumps(bundle["metrics"], indent=2),
        encoding="utf-8",
    )

    print("\nTraining complete.")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {metrics_path}")
    print("\nEvaluation on held-out program titles:")
    print(json.dumps(bundle["metrics"], indent=2))


if __name__ == "__main__":
    main()
