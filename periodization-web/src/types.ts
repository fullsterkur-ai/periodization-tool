
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

export type Exercise = {
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

export type Session = {
  day: number;
  exercises: Exercise[];
};

export type PlanWeek = {
  week: number;
  phase: string;
  is_deload: boolean;
  phase_note: string;
  sessions: Session[];
};

export type PlanResponse = {
  athlete: AthleteSettings;
  programme: {
    total_weeks: number;
    requested_week: number;
    saved_current_week: number;
    completed_weeks: number[];
    week_completed: boolean;
  };
  weeks: PlanWeek[];
  limitations?: string[];
};

export type FeedbackValues = {
  completed_sets: number;
  average_completed_reps: number;
  effort: number;
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

export type FeedbackResponse = {
  recommended_action?: string;
  message?: string;
  [key: string]: unknown;
};

export type CompleteWeekResponse = {
  message?: string;
  current_week?: number;
  [key: string]: unknown;
};
