import type {
  AthleteSettings,
  WorkoutFeedback,
  FeedbackResponse,
  CompleteWeekResponse,
  PlanResponse,
} from "./types";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

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

export function generatePlan(
  settings: AthleteSettings
): Promise<PlanResponse> {
  // Send the athlete settings directly.
  return apiRequest<PlanResponse>("/generate-plan", settings);
}

export function submitWorkoutFeedback(
  feedback: WorkoutFeedback
): Promise<FeedbackResponse> {
  return apiRequest<FeedbackResponse>(
    "/workout-feedback",
    feedback
  );
}

export function completeWeek(
  athlete_id: string,
  week_number: number
): Promise<CompleteWeekResponse> {
  return apiRequest<CompleteWeekResponse>("/complete-week", {
    athlete_id,
    week_number,
  });
}
