
const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

export type AthleteSettings = {
  athlete_id: string;
  goal: string;
  experience: string;
  equipment: string;
  days_per_week: number;
  weeks: number;
  week_number: number;
  exercises_per_session: number;
  starting_loads_kg: Record<string, number>;
  fatigue: number;
  performance_declining: boolean;
  simulation_completed_all_sets: boolean;
  simulation_completed_target_reps: boolean;
  simulation_effort: number;
  excluded_exercises: string[];
  restriction_notes: string;
};

export type WorkoutFeedback = {
  athlete_id: string;
  exercise_name: string;
  goal: string;
  experience: string;
  planned_sets: number;
  planned_reps: number;
  completed_sets: number;
  average_completed_reps: number;
  effort: number;
  week_number: number;
};

async function apiRequest<T>(
  endpoint: string,
  body: unknown
): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    body: JSON.stringify(body),
  });

  const data = await response.json().catch(() => null);

  if (!response.ok) {
    const detail = data?.detail;
    const errorMessage =
      typeof detail === "string"
        ? detail
        : detail
          ? JSON.stringify(detail)
          : `Request failed: ${response.status}`;

    throw new Error(errorMessage);
  }

  return data as T;
}

export function generatePlan(settings: AthleteSettings) {
  return apiRequest<Record<string, any>>("/generate-plan", {
    athlete: settings,
  });
}

export function submitWorkoutFeedback(feedback: WorkoutFeedback) {
  return apiRequest<Record<string, any>>(
    "/workout-feedback",
    feedback
  );
}

export function completeWeek(
  athlete_id: string,
  week_number: number
) {
  return apiRequest<Record<string, any>>("/complete-week", {
    athlete_id,
    week_number,
  });
}
