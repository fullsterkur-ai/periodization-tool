import { useState } from "react";
import "./App.css";
import {
  generatePlan,
  submitWorkoutFeedback,
  completeWeek,
  type AthleteSettings,
  type WorkoutFeedback,
} from "./api";

type Exercise = {
  exercise_name: string;
  sets: number;
  reps: number;
  dataset_intensity_0_10?: number;
  load_kg: number | null;
  progression_action?: string;
  progression_reason?: string;
  load_progression_status?: string;
  prescription_source?: string;
};

type Session = {
  day: number;
  exercises: Exercise[];
};

type PlanResponse = {
  athlete: Record<string, any>;
  programme: {
    total_weeks: number;
    requested_week: number;
    saved_current_week: number;
    completed_weeks: number[];
    week_completed: boolean;
  };
  weeks: {
    week: number;
    phase: string;
    is_deload: boolean;
    phase_note: string;
    sessions: Session[];
  }[];
  limitations?: string[];
};

const defaultAthlete: AthleteSettings = {
  athlete_id: "demo-athlete",
  goal: "strength",
  experience: "beginner",
  equipment: "Full Gym",
  days_per_week: 3,
  weeks: 8,
  week_number: 2,
  exercises_per_session: 5,
  starting_loads_kg: {
    "Bench Press (Barbell)": 40,
  },
  fatigue: 3,
  performance_declining: false,
  simulation_completed_all_sets: true,
  simulation_completed_target_reps: true,
  simulation_effort: 6,
  excluded_exercises: [],
  restriction_notes: "",
};

type FeedbackValues = {
  completed_sets: number;
  average_completed_reps: number;
  effort: number;
};

function App() {
  const [activePage, setActivePage] = useState("Training Plan");
  const [athlete, setAthlete] =
    useState<AthleteSettings>(defaultAthlete);
  const [plan, setPlan] = useState<PlanResponse | null>(null);
  const [selectedDay, setSelectedDay] = useState(0);
  const [feedbackValues, setFeedbackValues] =
    useState<Record<string, FeedbackValues>>({});
  const [submittedFeedback, setSubmittedFeedback] =
    useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");

  const weekData = plan?.weeks?.[0];
  const sessions = weekData?.sessions ?? [];
  const session = sessions[selectedDay];
  const currentWeek = weekData?.week ?? athlete.week_number;
  const totalWeeks = plan?.programme?.total_weeks ?? athlete.weeks;
  const completedWeeks = plan?.programme?.completed_weeks ?? [];

  function exerciseKey(exercise: Exercise) {
    return `${currentWeek}:${exercise.exercise_name.toLowerCase()}`;
  }

  function getFeedback(exercise: Exercise): FeedbackValues {
    const key = exerciseKey(exercise);

    return feedbackValues[key] ?? {
      completed_sets: exercise.sets,
      average_completed_reps: exercise.reps,
      effort: 6,
    };
  }

  async function loadPlan(weekNumber = athlete.week_number) {
    setLoading(true);
    setError("");
    setMessage("");

    try {
      const settings = {
        ...athlete,
        week_number: weekNumber,
      };

      const result = (await generatePlan(settings)) as PlanResponse;

      setPlan(result);
      setAthlete(settings);
      setSelectedDay(0);
      setFeedbackValues({});
      setSubmittedFeedback([]);
      setMessage(`Week ${result.weeks[0]?.week ?? weekNumber} loaded successfully.`);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Could not generate the plan."
      );
    } finally {
      setLoading(false);
    }
  }

  function updateFeedback(
    exercise: Exercise,
    field: keyof FeedbackValues,
    value: number
  ) {
    const key = exerciseKey(exercise);
    const current = getFeedback(exercise);

    setFeedbackValues((previous) => ({
      ...previous,
      [key]: {
        ...current,
        [field]: value,
      },
    }));
  }

  async function recordFeedback(exercise: Exercise) {
    const key = exerciseKey(exercise);

    if (submittedFeedback.includes(key)) {
      setMessage("Feedback has already been submitted for this exercise.");
      return;
    }

    const values = getFeedback(exercise);

    const payload: WorkoutFeedback = {
      athlete_id: athlete.athlete_id,
      exercise_name: exercise.exercise_name,
      goal: athlete.goal,
      experience: athlete.experience,
      planned_sets: exercise.sets,
      planned_reps: exercise.reps,
      completed_sets: values.completed_sets,
      average_completed_reps: values.average_completed_reps,
      effort: values.effort,
      week_number: currentWeek,
    };

    setLoading(true);
    setError("");
    setMessage("");

    try {
      const result = await submitWorkoutFeedback(payload);

      setSubmittedFeedback((previous) => [...previous, key]);
      setMessage(
        `${exercise.exercise_name}: feedback saved. Recommended action: ${
          result.recommended_action ?? "recorded"
        }.`
      );
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Could not submit feedback."
      );
    } finally {
      setLoading(false);
    }
  }

  async function finishWeek() {
    if (!plan) return;

    setLoading(true);
    setError("");
    setMessage("");

    try {
      const result = await completeWeek(
        athlete.athlete_id,
        currentWeek
      );

      setMessage(result.message ?? `Week ${currentWeek} completed.`);

      const nextWeek = result.current_week;

      if (nextWeek && nextWeek <= totalWeeks) {
        await loadPlan(nextWeek);
      } else {
        setMessage("Programme completed!");
      }
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Could not complete this week."
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">P</div>
          <div>
            <strong>PERIODIZE</strong>
            <span>TRAINING SYSTEM</span>
          </div>
        </div>

        <p className="nav-label">WORKSPACE</p>

        {["Overview", "Training Plan", "Periodization", "Progress"].map(
          (item) => (
            <button
              key={item}
              className={`nav-item ${activePage === item ? "active" : ""}`}
              onClick={() => setActivePage(item)}
            >
              {item}
            </button>
          )
        )}

        <div className="sidebar-bottom">
          <div className="demo-tag">API CONNECTED</div>
          <p>{athlete.athlete_id}</p>
          <small>FastAPI periodization engine</small>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <span className="breadcrumb">Workspace / </span>
            <strong>{activePage}</strong>
          </div>
          <div className="profile">
            <div className="avatar">A</div>
            <div>
              <strong>{athlete.athlete_id}</strong>
              <small>{athlete.experience} athlete</small>
            </div>
          </div>
        </header>

        <section className="page">
          <div className="page-heading">
            <div>
              <p className="eyebrow">PERSONALIZED TRAINING</p>
              <h1>{activePage}</h1>
              <p className="muted">
                Training prescriptions and progression from your backend.
              </p>
            </div>

            <button
              className="outline-button"
              disabled={loading}
              onClick={() => loadPlan(athlete.week_number)}
            >
              {loading ? "Please wait..." : "Generate / Refresh Plan"}
            </button>
          </div>

          <div className="panel">
            <h3>Athlete and programme settings</h3>

            <div className="stats-grid">
              <label>
                Athlete ID
                <input
                  value={athlete.athlete_id}
                  onChange={(e) =>
                    setAthlete({ ...athlete, athlete_id: e.target.value })
                  }
                />
              </label>

              <label>
                Goal
                <select
                  value={athlete.goal}
                  onChange={(e) =>
                    setAthlete({ ...athlete, goal: e.target.value })
                  }
                >
                  <option value="strength">Strength</option>
                  <option value="hypertrophy">Hypertrophy</option>
                  <option value="general_fitness">General fitness</option>
                </select>
              </label>

              <label>
                Experience
                <select
                  value={athlete.experience}
                  onChange={(e) =>
                    setAthlete({ ...athlete, experience: e.target.value })
                  }
                >
                  <option value="beginner">Beginner</option>
                  <option value="intermediate">Intermediate</option>
                  <option value="advanced">Advanced</option>
                </select>
              </label>

              <label>
                Training days per week
                <input
                  type="number"
                  min={2}
                  max={6}
                  value={athlete.days_per_week}
                  onChange={(e) =>
                    setAthlete({
                      ...athlete,
                      days_per_week: Number(e.target.value),
                    })
                  }
                />
              </label>

              <label>
                Programme duration (weeks)
                <input
                  type="number"
                  min={4}
                  max={12}
                  value={athlete.weeks}
                  onChange={(e) =>
                    setAthlete({
                      ...athlete,
                      weeks: Number(e.target.value),
                    })
                  }
                />
              </label>

              <label>
                Week to generate
                <input
                  type="number"
                  min={1}
                  max={athlete.weeks}
                  value={athlete.week_number}
                  onChange={(e) =>
                    setAthlete({
                      ...athlete,
                      week_number: Number(e.target.value),
                    })
                  }
                />
              </label>
            </div>

            <button
              className="outline-button"
              disabled={loading}
              onClick={() => loadPlan(athlete.week_number)}
            >
              Generate plan for selected week
            </button>
          </div>

          {error && (
            <div className="panel" role="alert">
              <strong>Request failed</strong>
              <p>{error}</p>
            </div>
          )}

          {message && (
            <div className="panel" role="status">
              {message}
            </div>
          )}

          {!plan && (
            <div className="panel">
              <h3>Your training plan will appear here</h3>
              <p className="muted">
                Generate a plan to load real exercises from FastAPI.
              </p>
            </div>
          )}

          {plan && (
            <>
              <div className="welcome-banner">
                <div>
                  <span className="banner-label">CURRENT TRAINING BLOCK</span>
                  <h2>Week {currentWeek} — {weekData?.phase}</h2>
                  <p>{weekData?.phase_note}</p>
                  <div className="banner-meta">
                    <span>{totalWeeks} total weeks</span>
                    <span>{sessions.length} sessions</span>
                    <span>{athlete.goal}</span>
                  </div>
                </div>
              </div>

              <div className="stats-grid">
                <Stat
                  label="CURRENT WEEK"
                  value={`${currentWeek} / ${totalWeeks}`}
                  change="Loaded from the API"
                />
                <Stat
                  label="COMPLETED WEEKS"
                  value={String(completedWeeks.length)}
                  change="Saved programme state"
                />
                <Stat
                  label="TRAINING SESSIONS"
                  value={String(sessions.length)}
                  change="For this generated week"
                />
                <Stat
                  label="PHASE"
                  value={weekData?.phase ?? "—"}
                  change={weekData?.is_deload ? "Deload week" : "Training phase"}
                />
              </div>

              <div className="section-heading">
                <div>
                  <h2>Weekly training schedule</h2>
                  <p className="muted">
                    Select a session to view its prescribed exercises.
                  </p>
                </div>
                <span className="small-chip">WEEK {currentWeek}</span>
              </div>

              <div className="schedule-layout">
                <div className="day-list">
                  {sessions.map((item, index) => (
                    <button
                      key={item.day}
                      className={`day-card ${
                        selectedDay === index ? "selected" : ""
                      }`}
                      onClick={() => setSelectedDay(index)}
                    >
                      <div className="day-date">DAY {item.day}</div>
                      <div className="day-info">
                        <strong>Session {item.day}</strong>
                        <small>{item.exercises.length} exercises</small>
                      </div>
                      <span className="day-arrow">›</span>
                    </button>
                  ))}
                </div>

                {session && (
                  <div className="workout-panel">
                    <div className="workout-heading">
                      <div>
                        <p className="eyebrow">
                          DAY {session.day} / WEEK {currentWeek}
                        </p>
                        <h3>Training session {session.day}</h3>
                        <p className="muted">{weekData?.phase}</p>
                      </div>
                      <span className="workout-symbol">✦</span>
                    </div>

                    <div className="exercise-table">
                      <div className="table-head">
                        <span>EXERCISE</span>
                        <span>SETS</span>
                        <span>REPS</span>
                        <span>LOAD</span>
                      </div>

                      {session.exercises.map((exercise, index) => {
                        const key = exerciseKey(exercise);
                        const values = getFeedback(exercise);
                        const submitted = submittedFeedback.includes(key);

                        return (
                          <div className="exercise-row" key={exercise.exercise_name}>
                            <div className="exercise-name">
                              <span className="exercise-number">
                                {String(index + 1).padStart(2, "0")}
                              </span>
                              <strong>{exercise.exercise_name}</strong>
                            </div>

                            <span>{exercise.sets}</span>
                            <span>{exercise.reps}</span>
                            <span className="intensity">
                              {exercise.load_kg == null
                                ? "Not set"
                                : `${exercise.load_kg} kg`}
                            </span>

                            <div className="feedback-controls">
                              <label>
                                Completed sets
                                <input
                                  type="number"
                                  min={0}
                                  max={exercise.sets}
                                  value={values.completed_sets}
                                  disabled={submitted}
                                  onChange={(e) =>
                                    updateFeedback(
                                      exercise,
                                      "completed_sets",
                                      Number(e.target.value)
                                    )
                                  }
                                />
                              </label>

                              <label>
                                Average reps completed
                                <input
                                  type="number"
                                  min={0}
                                  max={50}
                                  value={values.average_completed_reps}
                                  disabled={submitted}
                                  onChange={(e) =>
                                    updateFeedback(
                                      exercise,
                                      "average_completed_reps",
                                      Number(e.target.value)
                                    )
                                  }
                                />
                              </label>

                              <label>
                                Effort (1–10)
                                <input
                                  type="number"
                                  min={1}
                                  max={10}
                                  value={values.effort}
                                  disabled={submitted}
                                  onChange={(e) =>
                                    updateFeedback(
                                      exercise,
                                      "effort",
                                      Number(e.target.value)
                                    )
                                  }
                                />
                              </label>

                              <button
                                className="outline-button"
                                disabled={loading || submitted}
                                onClick={() => recordFeedback(exercise)}
                              >
                                {submitted ? "Feedback saved" : "Submit feedback"}
                              </button>

                              <small className="muted">
                                Intensity scale:{" "}
                                {exercise.dataset_intensity_0_10 ?? "Not provided"}
                                /10
                              </small>

                              {exercise.load_progression_status && (
                                <small className="muted">
                                  Load status: {exercise.load_progression_status}
                                </small>
                              )}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>

              <div className="panel">
                <h3>Complete week {currentWeek}</h3>
                <p className="muted">
                  Submit feedback for every exercise in the planned week first.
                  The backend will reject completion if feedback is missing.
                  Only confirm completion after the week's workouts have
                  actually been performed and recorded.
                </p>
                <button
                  className="outline-button"
                  disabled={loading}
                  onClick={finishWeek}
                >
                  {loading ? "Processing..." : `Confirm Week ${currentWeek} Complete`}
                </button>
              </div>

              {plan.limitations && (
                <div className="panel">
                  <h3>Research prototype limitations</h3>
                  <ul>
                    {plan.limitations.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                </div>
              )}
            </>
          )}
        </section>
      </main>
    </div>
  );
}

function Stat({
  label,
  value,
  change,
}: {
  label: string;
  value: string;
  change: string;
}) {
  return (
    <div className="stat-card">
      <span className="stat-label">{label}</span>
      <strong className="stat-value">{value}</strong>
      <span className="stat-change">{change}</span>
    </div>
  );
}

export default App;
