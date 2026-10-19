from __future__ import annotations
import ast, math, re
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import csv
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
DATA_PATH = (  ROOT / "data" / "raw" / "programs_detailed_boostcamp_kaggle.csv")

MODEL_PATH = ROOT / "models" / "prescription_model.joblib"
ML_BUNDLE = None
app = FastAPI(title="Periodization Engine API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000", "http://127.0.0.1:3000"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

GOAL_TERMS = {
    "strength": ["strength", "powerlifting", "powerbuilding", "athletics"],
    "hypertrophy": ["bodybuilding", "muscle", "sculpting", "hypertrophy", "powerbuilding"],
    "general_fitness": ["fitness", "athletics", "general", "muscle"],
}
LEVEL_ALIASES = {"beginner": ["beginner", "novice"], "novice": ["novice", "beginner"], "intermediate": ["intermediate"], "advanced": ["advanced"]}
EQUIPMENT_ALIASES = {"full gym": ["full gym", "gym"], "garage gym": ["garage gym", "garage"], "at home": ["at home", "home"], "dumbbell only": ["dumbbell only", "dumbbell"]}

class AthleteInput(BaseModel):
    goal: str = Field(default="strength", description="strength, hypertrophy, or general_fitness")
    experience: str = Field(default="beginner", description="beginner, intermediate, or advanced")
    equipment: str = Field(default="Full Gym", description="Equipment available to athlete")
    days_per_week: int = Field(default=3, ge=2, le=6)
    weeks: int = Field(default=8, ge=4, le=12)
    exercises_per_session: int = Field(default=5, ge=3, le=8)
class WorkoutFeedback(BaseModel):
    athlete_id: str
    exercise_name: str
    goal: str = "strength"
    experience: str = "beginner"

    planned_sets: int = Field(ge=1, le=10)
    planned_reps: int = Field(ge=1, le=50)

    completed_sets: int = Field(ge=0, le=10)
    average_completed_reps: float = Field(ge=0, le=50)

    effort: int = Field(
        ge=1,
        le=10,
        description="Athlete-reported effort, from 1 (easy) to 10 (maximal)"
    )

def safe_text(value: Any) -> str:
    if value is None: return ""
    try:
        if pd.isna(value): return ""
    except (TypeError, ValueError): pass
    if isinstance(value, (list, tuple, set)): return " ".join(str(x) for x in value)
    value = str(value).strip()
    if value.startswith("[") and value.endswith("]"):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple, set)): return " ".join(str(x) for x in parsed)
        except (ValueError, SyntaxError): pass
    return value

def normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", safe_text(value).lower()).strip()

def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lookup = {str(col).strip().lower(): col for col in df.columns}
    return next((lookup[c.lower()] for c in candidates if c.lower() in lookup), None)

def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Copy fitness_programs.csv into data/raw/.")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    required = ["exercise_name", "sets", "reps"]
    missing = [c for c in required if c not in df.columns]
    if missing: raise ValueError(f"CSV is missing expected columns: {missing}. Found columns: {list(df.columns)}")
    df = df.drop_duplicates().copy()
    df["exercise_name"] = df["exercise_name"].map(safe_text).str.strip()
    df = df[df["exercise_name"].ne("")].copy()
    for col in ["sets", "reps", "intensity", "week", "day"]:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
    # In this dataset negative reps encode duration in seconds, not negative repetitions.
    df["duration_seconds"] = np.where(df["reps"] < 0, df["reps"].abs(), np.nan)
    df["rep_count"] = df["reps"].where(df["reps"] > 0, np.nan)
    return df

def list_match(value: Any, terms: list[str]) -> bool:
    value_norm = normalize(value)
    return any(term in value_norm for term in terms)

def candidate_rows(df: pd.DataFrame, athlete: AthleteInput) -> pd.DataFrame:
    candidates = df.copy()
    goal_col = find_column(candidates, ["goal", "goals"])
    level_col = find_column(candidates, ["level", "experience", "experience_level"])
    equipment_col = find_column(candidates, ["equipment", "equipment_type"])
    if goal_col:
        terms = GOAL_TERMS.get(normalize(athlete.goal), [normalize(athlete.goal)])
        mask = candidates[goal_col].map(lambda x: list_match(x, terms))
        if mask.any(): candidates = candidates[mask].copy()
    if level_col:
        terms = LEVEL_ALIASES.get(normalize(athlete.experience), [normalize(athlete.experience)])
        mask = candidates[level_col].map(lambda x: list_match(x, terms))
        if mask.any(): candidates = candidates[mask].copy()
    if equipment_col:
        terms = EQUIPMENT_ALIASES.get(normalize(athlete.equipment), [normalize(athlete.equipment)])
        mask = candidates[equipment_col].map(lambda x: list_match(x, terms))
        if mask.any(): candidates = candidates[mask].copy()
    return candidates

def summarize_exercises(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, group in candidates.groupby("exercise_name", dropna=True):
        set_values = pd.to_numeric(group["sets"], errors="coerce")
        set_values = set_values[set_values.between(1, 10)]
        rep_values = pd.to_numeric(group["rep_count"], errors="coerce")
        rep_values = rep_values[rep_values.between(1, 50)]
        duration_values = pd.to_numeric(group["duration_seconds"], errors="coerce")
        duration_values = duration_values[duration_values.between(1, 3600)]
        intensity_values = pd.to_numeric(group["intensity"], errors="coerce") if "intensity" in group.columns else pd.Series(dtype=float)
        if not rep_values.empty:
            prescription_type, target_reps, duration_seconds = "reps", int(round(float(rep_values.median()))), None
        elif not duration_values.empty:
            prescription_type, target_reps, duration_seconds = "time", None, int(round(float(duration_values.median())))
        else: continue
        rows.append({"exercise_name": str(name), "sets": int(round(float(set_values.median()))) if not set_values.empty else 3,
                     "prescription_type": prescription_type, "reps": target_reps, "duration_seconds": duration_seconds,
                     "dataset_intensity_0_10": round(float(intensity_values.median()), 1) if not intensity_values.dropna().empty else None,
                     "record_count": int(len(group))})
    result = pd.DataFrame(rows)
    if result.empty: return result
    return result.sort_values(["record_count", "exercise_name"], ascending=[False, True])


def normalize_ml_goal(goal: str) -> str:
    value = normalize(goal)

    if any(x in value for x in [
        "strength", "powerlifting", "powerbuilding", "athletics"
    ]):
        return "strength"

    if any(x in value for x in [
        "hypertrophy", "bodybuilding", "muscle", "sculpting"
    ]):
        return "hypertrophy"

    return "general_fitness"


def normalize_ml_experience(experience: str) -> str:
    value = normalize(experience)

    if value in ("beginner", "novice"):
        return "beginner"
    if value == "intermediate":
        return "intermediate"
    if value == "advanced":
        return "advanced"

    return "unknown"


def normalize_ml_equipment(equipment: str) -> str:
    value = normalize(equipment)

    if "dumbbell" in value:
        return "dumbbell only"
    if "garage" in value:
        return "garage gym"
    if "home" in value:
        return "at home"
    if "gym" in value:
        return "full gym"

    return "unknown"


def load_ml_bundle():
    global ML_BUNDLE

    if ML_BUNDLE is None and MODEL_PATH.exists():
        ML_BUNDLE = joblib.load(MODEL_PATH)

    return ML_BUNDLE


def predict_prescription(
    athlete: AthleteInput,
    exercise: dict,
    week_no: int,
    day_no: int,
):
    bundle = load_ml_bundle()

    # If the model has not been trained yet, retain the existing
    # dataset-based prescription as a fallback.
    if bundle is None:
        return None

    features = pd.DataFrame([{
        "goal": normalize_ml_goal(athlete.goal),
        "experience": normalize_ml_experience(athlete.experience),
        "equipment": normalize_ml_equipment(athlete.equipment),
        "exercise_name": exercise["exercise_name"],
        "program_length": athlete.weeks,
        "time_per_workout": np.nan,
        "week": week_no,
        "day": day_no,
    }])

    predicted_sets = bundle["sets_model"].predict(features)[0]
    predicted_sets = int(np.clip(round(float(predicted_sets)), 1, 10))

    predicted_reps = None

    if exercise["prescription_type"] == "reps":
        predicted_reps = bundle["reps_model"].predict(features)[0]
        predicted_reps = int(
            np.clip(round(float(predicted_reps)), 1, 50)
        )

    return {
        "sets": predicted_sets,
        "reps": predicted_reps,
        "model_used": True,
    }

def make_plan(athlete: AthleteInput) -> dict[str, Any]:
    try: df = load_dataset()
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc: raise HTTPException(status_code=500, detail=str(exc)) from exc
    catalogue = summarize_exercises(candidate_rows(df, athlete))
    if catalogue.empty:
        catalogue = summarize_exercises(df)
        selection_note = "Requested filters did not match usable records; used the overall dataset catalogue. Check metadata labels."
    else:
        selection_note = "Candidates were filtered using available goal, experience, and equipment metadata."
    if catalogue.empty: raise HTTPException(status_code=422, detail="No usable exercises found. Check exercise_name, sets, and reps in the CSV.")
    # Keep the full ranked catalogue; select a different subset for each day.
    catalogue = catalogue.reset_index(drop=True)
    weeks = []
    for week_no in range(1, athlete.weeks + 1):
        is_deload = week_no % 4 == 0 or week_no == athlete.weeks
        if is_deload:
            phase, phase_note, set_multiplier = "Deload / recovery", "Reduce volume; use easy, technically controlled effort.", 0.6
        elif week_no <= 3:
            phase, phase_note, set_multiplier = "Accumulation", "Build consistent training volume; progress only after well-completed sessions.", 1.0
        elif week_no <= max(5, athlete.weeks - 2):
            phase, phase_note, set_multiplier = "Intensification", "Prioritize quality work and gradually increase difficulty when appropriate.", 1.0
        else:
            phase, phase_note, set_multiplier = "Realization / consolidation", "Consolidate progress without forcing maximal attempts.", 1.0
        
        sessions = []

        
        for day_no in range(1, athlete.days_per_week + 1):
            exercises = []
            
            # Select a different slice of the ranked catalogue for each day.
            start = (day_no - 1) * athlete.exercises_per_session

            day_catalogue = catalogue.iloc[
                start : start + athlete.exercises_per_session
            ]

            chosen = day_catalogue.to_dict(orient="records")

            # Assign exercises to different day groups instead of repeating
            # the same top-ranked exercises in every session.
            day_catalogue = catalogue.iloc[
                (day_no - 1)::athlete.days_per_week
            ].head(athlete.exercises_per_session)

            chosen = day_catalogue.to_dict(orient="records")

            for item in chosen:
                ml_prediction = predict_prescription(
                    athlete=athlete,
                    exercise=item,
                    week_no=week_no,
                    day_no=day_no,
                )

                # Use ML predictions if the trained model is available.
                if ml_prediction is not None:
                    base_sets = ml_prediction["sets"]
                else:
                    base_sets = int(item["sets"])

                final_sets = max(
                    1,
                    int(math.floor(base_sets * set_multiplier))
                )

                entry = {
                    "exercise_name": item["exercise_name"],
                    "sets": final_sets,
                    "prescription_type": item["prescription_type"],
                    "dataset_intensity_0_10": item["dataset_intensity_0_10"],
                    "prescription_source": (
                        "random_forest_ml"
                        if ml_prediction is not None
                        else "dataset_median_fallback"
                    ),
                }

                if item["prescription_type"] == "time":
                    entry["duration_seconds"] = item["duration_seconds"]
                else:
                    if ml_prediction is not None:
                        entry["reps"] = ml_prediction["reps"]
                    else:
                        entry["reps"] = item["reps"]

                entry["progression_rule"] = (
                    "Deload: do not increase load; prioritize recovery and technique."
                    if is_deload
                    else
                    "If all work is completed with good technique and manageable "
                    "effort, increase the next exposure conservatively; otherwise "
                    "repeat or reduce."
                )

                exercises.append(entry)

            sessions.append({
                "day": day_no,
                "exercises": exercises
            })

        weeks.append({
            "week": week_no,
            "phase": phase,
            "is_deload": is_deload,
            "phase_note": phase_note,
            "sessions": sessions
        })

    return {
        "athlete": athlete.model_dump(),
        "engine_type": "Random Forest prescription prediction + rule-based periodization",
        "ml_model_loaded": load_ml_bundle() is not None,
        "dataset_rows_after_exact_deduplication": int(len(df)),
        "candidate_exercise_count": int(len(catalogue)),
        "selection_note": selection_note,
        "limitations": [
            "Research/demo prototype, not a clinically validated prescription.",
            "Dataset intensity remains on its original 0-10 scale; it is not assumed to be RPE.",
            "The ML models learn dataset prescription patterns, not individual athlete outcomes.",
            "Progression guidance remains rule-based and does not calculate training loads.",
            "Exercise selection currently ranks by record frequency, not muscle-group balance."
        ],
        "weeks": weeks
    }

@app.get("/health")
def health(): return {"status": "ok", "dataset_path": str(DATA_PATH), "dataset_exists": DATA_PATH.exists()}

@app.post("/generate-plan")
def generate_plan(athlete: AthleteInput): return make_plan(athlete)

FEEDBACK_PATH = ROOT / "data" / "processed" / "workout_feedback.csv"


@app.post("/workout-feedback")
def record_workout_feedback(feedback: WorkoutFeedback):
    # Simple demonstration rules; these are not learned ML predictions.
    if feedback.completed_sets < feedback.planned_sets:
        action = "reduce"
        reason = "Not all planned sets were completed."

    elif feedback.average_completed_reps < feedback.planned_reps:
        action = "repeat"
        reason = "The planned repetition target was not achieved."

    elif feedback.effort >= 9:
        action = "repeat"
        reason = "The reported effort was very high."

    else:
        action = "progress"
        reason = "The planned work was completed without a very high effort rating."

    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **feedback.model_dump(),
        "recommended_action": action,
        "reason": reason,
    }

    file_exists = FEEDBACK_PATH.exists()

    with FEEDBACK_PATH.open(
        "a",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=record.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(record)

    return {
        "recommended_action": action,
        "reason": reason,
        "feedback_saved": True,
        "message": (
            "Feedback recorded. This recommendation uses explicit rules, "
            "not the Random Forest model."
        ),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
