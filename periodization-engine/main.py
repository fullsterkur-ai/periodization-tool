from __future__ import annotations
import ast, math, re
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
DATA_PATH = (  ROOT / "data" / "raw" / "programs_detailed_boostcamp_kaggle.csv")
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
    chosen = catalogue.head(athlete.exercises_per_session).to_dict(orient="records")
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
            for item in chosen:
                entry = {"exercise_name": item["exercise_name"], "sets": max(1, int(math.floor(int(item["sets"]) * set_multiplier))),
                         "prescription_type": item["prescription_type"], "dataset_intensity_0_10": item["dataset_intensity_0_10"]}
                if item["prescription_type"] == "time": entry["duration_seconds"] = item["duration_seconds"]
                else: entry["reps"] = item["reps"]
                entry["progression_rule"] = ("Deload: do not increase load; prioritize recovery and technique." if is_deload else
                    "If all work is completed with good technique and manageable effort, increase the next exposure conservatively; otherwise repeat or reduce.")
                exercises.append(entry)
            sessions.append({"day": day_no, "exercises": exercises})
        weeks.append({"week": week_no, "phase": phase, "is_deload": is_deload, "phase_note": phase_note, "sessions": sessions})
    return {"athlete": athlete.model_dump(), "engine_type": "Data-driven exercise retrieval + rule-based periodization",
            "dataset_rows_after_exact_deduplication": int(len(df)), "candidate_exercise_count": int(len(catalogue)),
            "selection_note": selection_note,
            "limitations": ["Research/demo prototype, not a clinically validated prescription.",
                "Dataset intensity remains on its original 0-10 scale; it is not assumed to be RPE.",
                "No individual athlete outcome data is used to train a progression predictor.",
                "Progression guidance is rule-based and does not calculate loads from a generic 1RM.",
                "Exercise selection currently ranks by record frequency, not muscle-group balance."], "weeks": weeks}

@app.get("/health")
def health(): return {"status": "ok", "dataset_path": str(DATA_PATH), "dataset_exists": DATA_PATH.exists()}

@app.post("/generate-plan")
def generate_plan(athlete: AthleteInput): return make_plan(athlete)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
