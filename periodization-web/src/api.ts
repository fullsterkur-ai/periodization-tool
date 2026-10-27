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

export interface TrainingDashboardData {
  model_type: string;
  random_state: number;
  max_train_rows: number;
  evaluation_method: string;
  features: string[];
  targets: {
    sets: TrainingTargetData;
    reps: TrainingTargetData;
  };
  limitations: string[];
}

export interface TrainingTargetData {
  metrics: {
    target: string;
    train_rows: number;
    test_rows: number;
    mae: number;
    rmse: number;
  };
  diagnostics: {
    actual_vs_predicted: {
      actual: number;
      predicted: number;
    }[];
    residual_histogram: {
      counts: number[];
      bin_edges: number[];
    };
    feature_importance: {
      feature: string;
      importance: number;
    }[];
    learning_curve: {
  training_rows: number[];
  train_mae: number[];
  validation_mae: number[];
  validation_method: string;
  diagnostic_estimators: number;
  max_rows: number;
};
  };
}

export async function fetchTrainingDashboard(): Promise<TrainingDashboardData> {
  const response = await fetch(
    `${API_BASE_URL}/training-dashboard`,
    {
      headers: { Accept: "application/json" },
    }
  );

  const data = await response.json().catch(() => null);

  if (!response.ok) {
    throw new Error(
      typeof data?.detail === "string"
        ? data.detail
        : `Could not load training results (${response.status}).`
    );
  }

  return data as TrainingDashboardData;
}
