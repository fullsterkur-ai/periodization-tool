
import { useState } from "react";
import "./App.css";

type Exercise = {
  name: string;
  sets: number;
  reps: string;
  intensity: string;
};

type Workout = {
  day: string;
  title: string;
  focus: string;
  duration: string;
  exercises: Exercise[];
};

const workouts: Workout[] = [
  {
    day: "MON",
    title: "Lower Body A",
    focus: "Strength",
    duration: "55 min",
    exercises: [
      { name: "Barbell Back Squat", sets: 4, reps: "6", intensity: "75%" },
      { name: "Romanian Deadlift", sets: 3, reps: "8", intensity: "70%" },
      { name: "Leg Press", sets: 3, reps: "10", intensity: "65%" },
      { name: "Plank", sets: 3, reps: "45 sec", intensity: "Bodyweight" },
    ],
  },
  {
    day: "TUE",
    title: "Upper Body A",
    focus: "Strength",
    duration: "50 min",
    exercises: [
      { name: "Bench Press", sets: 4, reps: "6", intensity: "75%" },
      { name: "Barbell Row", sets: 4, reps: "8", intensity: "70%" },
      { name: "Overhead Press", sets: 3, reps: "8", intensity: "65%" },
    ],
  },
  {
    day: "WED",
    title: "Recovery",
    focus: "Active recovery",
    duration: "25 min",
    exercises: [
      { name: "Light Cycling", sets: 1, reps: "15 min", intensity: "Easy" },
      { name: "Mobility Work", sets: 1, reps: "10 min", intensity: "Easy" },
    ],
  },
  {
    day: "THU",
    title: "Lower Body B",
    focus: "Hypertrophy",
    duration: "55 min",
    exercises: [
      { name: "Deadlift", sets: 3, reps: "5", intensity: "80%" },
      { name: "Bulgarian Split Squat", sets: 3, reps: "10/leg", intensity: "65%" },
      { name: "Leg Curl", sets: 3, reps: "12", intensity: "60%" },
    ],
  },
  {
    day: "FRI",
    title: "Upper Body B",
    focus: "Hypertrophy",
    duration: "50 min",
    exercises: [
      { name: "Incline Dumbbell Press", sets: 3, reps: "10", intensity: "65%" },
      { name: "Lat Pulldown", sets: 3, reps: "10", intensity: "65%" },
      { name: "Lateral Raise", sets: 3, reps: "12", intensity: "Light" },
    ],
  },
  {
    day: "SAT",
    title: "Optional Activity",
    focus: "Conditioning",
    duration: "30 min",
    exercises: [
      { name: "Walking or Cycling", sets: 1, reps: "30 min", intensity: "Easy" },
    ],
  },
  {
    day: "SUN",
    title: "Rest Day",
    focus: "Recovery",
    duration: "Rest",
    exercises: [
      { name: "Rest and Recovery", sets: 0, reps: "—", intensity: "—" },
    ],
  },
];

const phases = [
  { week: "W1", name: "Foundation", type: "Accumulation", status: "complete" },
  { week: "W2", name: "Volume Build", type: "Accumulation", status: "complete" },
  { week: "W3", name: "Progression", type: "Intensification", status: "current" },
  { week: "W4", name: "Deload", type: "Recovery", status: "upcoming" },
  { week: "W5", name: "Strength Build", type: "Intensification", status: "upcoming" },
  { week: "W6", name: "Peak Week", type: "Realization", status: "upcoming" },
];

function App() {
  const [selectedDay, setSelectedDay] = useState(0);
  const [activePage, setActivePage] = useState("Overview");
  const workout = workouts[selectedDay];

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

        {["Overview", "Training Plan", "Periodization", "Exercise Library", "Progress"].map(
          (item) => (
            <button
              key={item}
              className={`nav-item ${activePage === item ? "active" : ""}`}
              onClick={() => setActivePage(item)}
            >
              <span>{{
                Overview: "◈",
                "Training Plan": "▤",
                Periodization: "◷",
                "Exercise Library": "▦",
                Progress: "↗",
              }[item]}</span>
              {item}
            </button>
          )
        )}

        <div className="sidebar-bottom">
          <div className="demo-tag">DEMO MODE</div>
          <p>Sample athlete data</p>
          <small>Periodization Module · v1.0</small>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <span className="breadcrumb">Workspace / </span>
            <strong>{activePage}</strong>
          </div>
          <div className="profile">
            <div className="avatar">AJ</div>
            <div>
              <strong>Alex Johnson</strong>
              <small>Intermediate athlete</small>
            </div>
          </div>
        </header>

        {activePage === "Exercise Library" ? (
          <section className="page">
            <p className="eyebrow">EXERCISE LIBRARY</p>
            <h1>Exercise collection</h1>
            <p className="muted">Exercises included in the demonstration program.</p>
            <div className="exercise-library">
              {[...new Set(workouts.flatMap(w => w.exercises.map(e => e.name)))].map(name => (
                <div className="library-item" key={name}>
                  <span className="exercise-icon">✦</span>
                  <strong>{name}</strong>
                  <span className="muted">Strength training</span>
                </div>
              ))}
            </div>
          </section>
        ) : activePage === "Progress" ? (
          <section className="page">
            <p className="eyebrow">TRAINING ANALYTICS</p>
            <h1>Progress overview</h1>
            <p className="muted">Illustrative progress indicators for the demo athlete.</p>
            <div className="stats-grid">
              <Stat label="Sessions completed" value="12 / 18" change="67% of planned sessions" />
              <Stat label="Program adherence" value="89%" change="Sample metric" />
              <Stat label="Training block" value="Week 3" change="Of a 6-week block" />
            </div>
            <div className="panel">
              <h3>Program progression</h3>
              <div className="progress-track"><div style={{ width: "50%" }} /></div>
              <p className="muted">3 of 6 weeks represented in the current demo.</p>
            </div>
          </section>
        ) : activePage === "Periodization" ? (
          <section className="page">
            <p className="eyebrow">TRAINING STRUCTURE</p>
            <h1>Periodization timeline</h1>
            <p className="muted">Illustrative six-week training block.</p>
            <div className="phase-list">
              {phases.map((phase) => (
                <div className={`phase-row ${phase.status}`} key={phase.week}>
                  <div className="phase-week">{phase.week}</div>
                  <div className="phase-info">
                    <strong>{phase.name}</strong>
                    <small>{phase.type}</small>
                  </div>
                  <span className="status-pill">{phase.status}</span>
                </div>
              ))}
            </div>
          </section>
        ) : (
          <section className="page">
            <div className="page-heading">
              <div>
                <p className="eyebrow">ATHLETE WORKSPACE / 2026</p>
                <h1>{activePage === "Training Plan" ? "Your training plan" : "Training overview"}</h1>
                <p className="muted">
                  Your training schedule, phase progression and workout prescriptions.
                </p>
              </div>
              <button className="outline-button" onClick={() => window.print()}>
                Export / Print ↗
              </button>
            </div>

            <div className="welcome-banner">
              <div>
                <span className="banner-label">CURRENT TRAINING BLOCK</span>
                <h2>Build strength. Train smarter.</h2>
                <p>A structured six-week program focused on strength development.</p>
                <div className="banner-meta">
                  <span>◷ 6 weeks</span>
                  <span>♧ 5 training days</span>
                  <span>✦ Intermediate</span>
                </div>
              </div>
              <div className="banner-art">↗</div>
            </div>

            <div className="stats-grid">
              <Stat label="CURRENT WEEK" value="03 / 06" change="Progression phase" />
              <Stat label="CURRENT PHASE" value="Build" change="Intensification" />
              <Stat label="PLANNED SESSIONS" value="5 days" change="Per training week" />
              <Stat label="NEXT DELOAD" value="Week 04" change="Reduced training stress" />
            </div>

            <div className="section-heading">
              <div>
                <h2>Periodization timeline</h2>
                <p className="muted">Your training focus changes across the block.</p>
              </div>
              <span className="small-chip">6-WEEK PROGRAM</span>
            </div>

            <div className="phase-grid">
              {phases.map((phase) => (
                <div className={`phase-card ${phase.status}`} key={phase.week}>
                  <div className="phase-card-top">
                    <span>{phase.week}</span>
                    {phase.status === "current" && <span className="live-dot">● CURRENT</span>}
                    {phase.status === "complete" && <span className="check">✓</span>}
                  </div>
                  <strong>{phase.name}</strong>
                  <p>{phase.type}</p>
                  <div className="phase-bar"><div /></div>
                </div>
              ))}
            </div>

            <div className="section-heading">
              <div>
                <h2>Weekly training schedule</h2>
                <p className="muted">Select a day to view its exercise prescriptions.</p>
              </div>
              <span className="small-chip">WEEK 03</span>
            </div>

            <div className="schedule-layout">
              <div className="day-list">
                {workouts.map((item, index) => (
                  <button
                    key={item.day}
                    className={`day-card ${selectedDay === index ? "selected" : ""}`}
                    onClick={() => setSelectedDay(index)}
                  >
                    <div className="day-date">{item.day}</div>
                    <div className="day-info">
                      <strong>{item.title}</strong>
                      <small>{item.focus}</small>
                    </div>
                    <span className="day-arrow">›</span>
                  </button>
                ))}
              </div>

              <div className="workout-panel">
                <div className="workout-heading">
                  <div>
                    <p className="eyebrow">{workout.day} / WEEK 03</p>
                    <h3>{workout.title}</h3>
                    <p className="muted">{workout.focus} · {workout.duration}</p>
                  </div>
                  <span className="workout-symbol">✦</span>
                </div>

                <div className="exercise-table">
                  <div className="table-head">
                    <span>EXERCISE</span>
                    <span>SETS</span>
                    <span>REPS</span>
                    <span>INTENSITY</span>
                  </div>
                  {workout.exercises.map((exercise, index) => (
                    <div className="exercise-row" key={exercise.name}>
                      <div className="exercise-name">
                        <span className="exercise-number">{String(index + 1).padStart(2, "0")}</span>
                        <strong>{exercise.name}</strong>
                      </div>
                      <span>{exercise.sets || "—"}</span>
                      <span>{exercise.reps}</span>
                      <span className="intensity">{exercise.intensity}</span>
                    </div>
                  ))}
                </div>

                <div className="workout-note">
                  <span>ⓘ</span>
                  <p>
                    <strong>Prescription note</strong>
                    <br />
                    Percentages are illustrative demo targets, not personalized
                    prescriptions. Adjust based on the athlete's assessed capacity.
                  </p>
                </div>
              </div>
            </div>
          </section>
        )}
      </main>
    </div>
  );
}

function Stat({ label, value, change }: {
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
