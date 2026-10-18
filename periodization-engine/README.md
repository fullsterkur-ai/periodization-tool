# Periodization Engine Starter

1. Copy `fitness_programs.csv` to `data/raw/fitness_programs.csv`.
2. In VS Code terminal (Windows PowerShell):

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn main:app --reload
```

If activation is blocked, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and activate again.

Open `http://127.0.0.1:8000/health`, then `http://127.0.0.1:8000/docs`.

In `POST /generate-plan`, click Try it out and send:

```json
{
  "goal": "strength",
  "experience": "beginner",
  "equipment": "Full Gym",
  "days_per_week": 3,
  "weeks": 8,
  "exercises_per_session": 5
}
```

This is a research/demo prototype. It uses dataset-based exercise retrieval and transparent rules; it does not claim to train a progression predictor from individual athlete outcome data. Negative source `reps` values are treated as timed prescriptions in seconds. Review programming rules with a qualified strength coach before real-world use.
