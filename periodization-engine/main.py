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
from periodization import (
    choose_action,
    progress_load,
    make_week_schedule,
)
import json

ROOT = Path(__file__).resolve().parent
DATA_PATH = (  ROOT / "data" / "raw" / "programs_detailed_boostcamp_kaggle.csv")

MODEL_PATH = ROOT / "models" / "prescription_model.joblib"
FEEDBACK_PATH = ROOT / "data" / "processed" / "workout_feedback.csv"
STATE_PATH = ROOT / "data" / "processed" / "athlete_exercise_state.csv"
PROGRAM_STATE_PATH = (
    ROOT / "data" / "processed" / "athlete_program_state.json"
)

def load_adaptive_state(athlete_id: str, exercise_name: str):
    """Return the latest saved prescription for this athlete and exercise."""
    if not STATE_PATH.exists():
        return None

    try:
        state_df = pd.read_csv(STATE_PATH)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return None

    required = {
        "athlete_id", "exercise_name",
        "next_sets", "next_reps", "updated_at", "last_action"
    }
    if not required.issubset(state_df.columns):
        return None

    matches = state_df[
        state_df["athlete_id"].map(normalize).eq(normalize(athlete_id))
        & state_df["exercise_name"].map(normalize).eq(normalize(exercise_name))
    ]

    if matches.empty:
        return None

    row = matches.sort_values("updated_at").iloc[-1]

    return {
        "next_sets": int(row["next_sets"]),
        "next_reps": int(row["next_reps"]),
        "updated_at": str(row["updated_at"]),
        "last_action": str(row["last_action"]),
    }


def save_adaptive_state(feedback: WorkoutFeedback, action: str):
    """Save the next target prescription after receiving workout feedback."""
    next_sets = feedback.planned_sets
    next_reps = feedback.planned_reps

    if action == "progress":
        # Conservative demo policy: add one rep, up to the API's limit.
        next_reps = min(50, feedback.planned_reps + 1)

    elif action == "reduce":
        # If the athlete could not complete all sets, reduce the next set target.
        next_sets = max(1, feedback.planned_sets - 1)

    updated_at = datetime.now(timezone.utc).isoformat()

    new_row = {
        "athlete_id": feedback.athlete_id,
        "exercise_name": feedback.exercise_name,
        "next_sets": next_sets,
        "next_reps": next_reps,
        "last_action": action,
        "updated_at": updated_at,
    }

    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)

    if STATE_PATH.exists():
        try:
            state_df = pd.read_csv(STATE_PATH)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            state_df = pd.DataFrame()
    else:
        state_df = pd.DataFrame()

    if not state_df.empty and {
        "athlete_id", "exercise_name"
    }.issubset(state_df.columns):
        same_entry = (
            state_df["athlete_id"].map(normalize).eq(normalize(feedback.athlete_id))
            & state_df["exercise_name"].map(normalize).eq(normalize(feedback.exercise_name))
        )
        state_df = state_df.loc[~same_entry].copy()

    state_df = pd.concat(
        [state_df, pd.DataFrame([new_row])],
        ignore_index=True
    )

    state_df.to_csv(STATE_PATH, index=False)

    return new_row

def load_program_state(athlete_id: str) -> dict[str, Any]:
    """Load the saved programme state for one athlete."""
    default_state = {
        "athlete_id": athlete_id,
        "weeks": None,
        "current_week": 1,
        "completed_weeks": [],
        "current_loads_kg": {},
        "updated_at": None,
    }

    if not PROGRAM_STATE_PATH.exists():
        return default_state

    try:
        with PROGRAM_STATE_PATH.open("r", encoding="utf-8") as file:
            all_states = json.load(file)
    except (OSError, json.JSONDecodeError):
        return default_state

    if not isinstance(all_states, dict):
        return default_state

    saved = all_states.get(normalize(athlete_id))

    if not isinstance(saved, dict):
        return default_state

    state = {**default_state, **saved}

    if not isinstance(state.get("current_loads_kg"), dict):
        state["current_loads_kg"] = {}

    if not isinstance(state.get("completed_weeks"), list):
        state["completed_weeks"] = []

    return state


def save_program_state(athlete_id: str, state: dict[str, Any]) -> None:
    """Persist programme state without overwriting other athletes."""
    PROGRAM_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_states = {}

    if PROGRAM_STATE_PATH.exists():
        try:
            with PROGRAM_STATE_PATH.open("r", encoding="utf-8") as file:
                loaded = json.load(file)

            if isinstance(loaded, dict):
                all_states = loaded

        except (OSError, json.JSONDecodeError):
            all_states = {}

    state["athlete_id"] = athlete_id
    state["updated_at"] = datetime.now(timezone.utc).isoformat()

    all_states[normalize(athlete_id)] = state

    temp_path = PROGRAM_STATE_PATH.with_suffix(".tmp")

    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(all_states, file, indent=2, ensure_ascii=False)

    temp_path.replace(PROGRAM_STATE_PATH)

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
    athlete_id: str = Field(default="demo-athlete")
    goal: str = Field(default="strength")
    experience: str = Field(default="beginner")
    equipment: str = Field(default="Full Gym")

    days_per_week: int = Field(default=3, ge=2, le=6)
    weeks: int = Field(default=8, ge=4, le=12)
    week_number: int | None = Field(default=None, ge=1, le=12)
    exercises_per_session: int = Field(default=5, ge=3, le=8)

    # Enter working loads in kilograms, keyed by exercise name.
    # Example: {"Barbell Bench Press": 40}
    starting_loads_kg: dict[str, float] = Field(default_factory=dict)

    # Demonstration inputs for simulated athlete condition.
    fatigue: int = Field(default=3, ge=1, le=10)
    performance_declining: bool = False
    simulation_completed_all_sets: bool = True
    simulation_completed_target_reps: bool = True
    simulation_effort: int = Field(default=6, ge=1, le=10)

    # Exact exercise names to exclude from this demo plan.
    excluded_exercises: list[str] = Field(default_factory=list)

    # Displayed in the plan; not treated as medical advice.
    restriction_notes: str = ""

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
    try:
        df = load_dataset()
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Load the athlete's saved programme state.
    state = load_program_state(athlete.athlete_id)

    # Reset programme-week tracking if the requested programme duration changes.
    if state.get("weeks") != athlete.weeks:
        state["weeks"] = athlete.weeks
        state["current_week"] = 1
        state["completed_weeks"] = []

    saved_week = int(state.get("current_week", 1))
    week_no = athlete.week_number if athlete.week_number is not None else saved_week

    if week_no > athlete.weeks:
        raise HTTPException(
            status_code=422,
            detail=f"Week {week_no} exceeds the {athlete.weeks}-week programme.",
        )

    # Restore saved loads, then apply any manual starting-load overrides.
    current_loads = {
        normalize(name): float(load)
        for name, load in state.get("current_loads_kg", {}).items()
        if load is not None
        and math.isfinite(float(load))
        and float(load) > 0
    }

    for name, load in athlete.starting_loads_kg.items():
        if load is not None and math.isfinite(float(load)) and float(load) > 0:
            current_loads[normalize(name)] = float(load)

    state["current_loads_kg"] = current_loads

    catalogue = summarize_exercises(candidate_rows(df, athlete))

    if catalogue.empty:
        catalogue = summarize_exercises(df)
        selection_note = (
            "The requested metadata filters returned no usable records. "
            "The overall dataset catalogue was used instead."
        )
    else:
        selection_note = (
            "Exercise candidates were filtered using available program metadata."
        )

    if catalogue.empty:
        raise HTTPException(
            status_code=422,
            detail="No usable exercises found in the program dataset.",
        )

    catalogue = catalogue.reset_index(drop=True)

    excluded = {normalize(name) for name in athlete.excluded_exercises}

    if excluded:
        catalogue = catalogue[
            ~catalogue["exercise_name"].map(normalize).isin(excluded)
        ].reset_index(drop=True)

    if catalogue.empty:
        raise HTTPException(
            status_code=422,
            detail="All available exercises were excluded. Change the exclusions.",
        )

    # Keep the existing demo action visible, but do not use it to
    # automatically increase loads just because a plan was generated.
    simulated_action, simulated_reason = choose_action(
        completed_all_sets=athlete.simulation_completed_all_sets,
        completed_target_reps=athlete.simulation_completed_target_reps,
        effort=athlete.simulation_effort,
        fatigue=athlete.fatigue,
    )

    schedule = make_week_schedule(
        weeks=athlete.weeks,
        fatigue=athlete.fatigue,
        performance_declining=athlete.performance_declining,
    )

    week_info = schedule[week_no - 1]
    is_deload = week_info["is_deload"]
    set_multiplier = week_info["set_multiplier"]

    sessions = []

    for day_no in range(1, athlete.days_per_week + 1):
        day_catalogue = catalogue.iloc[
            (day_no - 1)::athlete.days_per_week
        ].head(athlete.exercises_per_session)

        exercises = []

        for item in day_catalogue.to_dict(orient="records"):
            exercise_name = item["exercise_name"]
            exercise_key = normalize(exercise_name)

            # Use the latest saved sets/reps when feedback exists.
            adaptive = load_adaptive_state(
                athlete.athlete_id,
                exercise_name,
            )

            base_sets = (
                adaptive["next_sets"]
                if adaptive is not None
                else int(item["sets"])
            )

            base_reps = (
                adaptive["next_reps"]
                if adaptive is not None
                else item["reps"]
            )

            final_sets = max(
                1,
                int(math.floor(base_sets * set_multiplier)),
            )

            last_action = (
                adaptive["last_action"]
                if adaptive is not None
                else "awaiting_feedback"
            )

            entry = {
                "exercise_name": exercise_name,
                "sets": final_sets,
                "prescription_type": item["prescription_type"],
                "dataset_intensity_0_10": item["dataset_intensity_0_10"],
                "prescription_source": (
                    "saved_adaptive_feedback"
                    if adaptive is not None
                    else "dataset_median_plus_rule_based_periodization"
                ),
                "phase": week_info["phase"],
                "is_deload": is_deload,
                "progression_action": (
                        "maintain_during_deload"
                        if is_deload
                        else "awaiting_workout_feedback"
                    ),
                "last_recorded_action": last_action,
                "progression_reason": (
                        "Deload week: avoid load increases."
                        if is_deload
                        else (
                            "Previous feedback is available, but this week's workout "
                            "must still be recorded before progression is applied."
                        )
                    ),
            }

            if item["prescription_type"] == "time":
                entry["duration_seconds"] = item["duration_seconds"]
            else:
                entry["reps"] = int(base_reps)

            current_load = current_loads.get(exercise_key)

            if current_load is not None:
                entry["load_kg"] = round(current_load, 2)
                entry["load_source"] = "saved_or_athlete_supplied_working_load"

                # A proposed load increase is not applied until actual
                # workout feedback and week completion are recorded.
                entry["next_exposure_load_kg"] = round(current_load, 2)
                entry["load_progression_status"] = "pending_workout_feedback"
            else:
                entry["load_kg"] = None
                entry["load_source"] = "not_provided"
                entry["next_exposure_load_kg"] = None
                entry["load_progression_status"] = "starting_load_required"

            exercises.append(entry)

        sessions.append({
            "day": day_no,
            "exercises": exercises,
        })

    # Persist known loads and programme duration. Generating a plan
    # deliberately does NOT mark the week complete or advance the week.
    state["current_loads_kg"] = current_loads
    save_program_state(athlete.athlete_id, state)

    return {
        "athlete": athlete.model_dump(),
        "programme": {
            "total_weeks": athlete.weeks,
            "requested_week": week_no,
            "saved_current_week": state["current_week"],
            "completed_weeks": state["completed_weeks"],
            "week_completed": week_no in state["completed_weeks"],
            "advance_requires_completion_confirmation": True,
        },
        "engine_type": (
            "Rule-based periodization with dataset-informed prescriptions "
            "and saved athlete state"
        ),
        "ml_model_loaded": load_ml_bundle() is not None,
        "dataset_rows_after_exact_deduplication": int(len(df)),
        "candidate_exercise_count": int(len(catalogue)),
        "selection_note": selection_note,
        "simulated_condition": {
            "fatigue": athlete.fatigue,
            "performance_declining": athlete.performance_declining,
            "completed_all_sets": athlete.simulation_completed_all_sets,
            "completed_target_reps": athlete.simulation_completed_target_reps,
            "effort": athlete.simulation_effort,
            "demo_recommended_action": simulated_action,
            "demo_reason": simulated_reason,
            "note": (
                "These simulation inputs are for demonstration only. "
                "They do not confirm that the workout was actually completed."
            ),
        },
        "restrictions": {
            "excluded_exercises": athlete.excluded_exercises,
            "notes": athlete.restriction_notes,
            "warning": (
                "Exercise exclusions are user-selected demo filters, not "
                "medical clearance or a determination that other exercises "
                "are safe for an injury."
            ),
        },
        "limitations": [
            "Research/demo prototype, not clinically validated.",
            "Dataset intensity remains on its original 0-10 scale; it is not assumed to be RPE or a percentage of 1RM.",
            "The Random Forest does not control this plan.",
            "Load progression is not applied automatically; actual workout feedback and completion confirmation are required.",
            "Exercises without a supplied or saved working load retain a null load.",
            "Exercise selection does not yet optimize muscle-group balance.",
        ],
        "weeks": [{
            "week": week_no,
            "phase": week_info["phase"],
            "is_deload": is_deload,
            "phase_note": week_info["phase_note"],
            "sessions": sessions,
        }],
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
        next_prescription = save_adaptive_state(feedback, action)

        return {
        "recommended_action": action,
        "reason": reason,
        "feedback_saved": True,
        "adaptive_state_saved": True,
        "next_prescription": {
            "sets": next_prescription["next_sets"],
            "reps": next_prescription["next_reps"],
        },
        "message": (
            "Feedback recorded. The saved next prescription can be used "
            "when generating another plan for the same athlete and exercise. "
            "The recommendation uses explicit rules, not the Random Forest model."
        ),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
