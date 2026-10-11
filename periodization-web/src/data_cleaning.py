
from pathlib import Path
import ast
import pandas as pd

# --------------------------------------------------
# 1. SET UP PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "fitness_programs.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. LOAD THE DATA
# --------------------------------------------------

df = pd.read_csv(RAW_PATH)

print("Original dataset shape:", df.shape)

# --------------------------------------------------
# 3. PARSE LIST-LIKE FIELDS
# --------------------------------------------------

def parse_list(value):
    """Convert text representations of lists into Python lists."""

    if pd.isna(value):
        return []

    if isinstance(value, list):
        return value

    if not isinstance(value, str):
        return [value]

    value = value.strip()

    if not value:
        return []

    try:
        parsed = ast.literal_eval(value)

        if isinstance(parsed, list):
            return parsed

        return [parsed]

    except (ValueError, SyntaxError):
        return [value]


df["goal_list"] = df["goal"].apply(parse_list)
df["level_list"] = df["level"].apply(parse_list)

# --------------------------------------------------
# 4. CLEAN EXERCISE NAMES
# --------------------------------------------------

df["exercise_name"] = (
    df["exercise_name"]
    .astype("string")
    .str.strip()
)

# A normalized key helps identify spelling/case duplicates.
# We retain the original exercise_name for display.
df["exercise_key"] = (
    df["exercise_name"]
    .str.casefold()
    .str.replace(r"\s+", " ", regex=True)
)

# --------------------------------------------------
# 5. VALIDATE REQUIRED FIELDS
# --------------------------------------------------

before = len(df)

df = df.dropna(subset=["title", "exercise_name"]).copy()

df = df[
    df["title"].astype("string").str.strip().ne("")
    & df["exercise_name"].str.len().gt(0)
].copy()

print("Rows removed for missing required fields:", before - len(df))

# --------------------------------------------------
# 6. PRESERVE REPETITION INFORMATION
# --------------------------------------------------

# Do not convert negative values to positive values.
# Their exact interpretation still needs source verification.

df["prescription_type"] = "unknown"

df.loc[df["reps"] > 0, "prescription_type"] = "repetitions"
df.loc[df["reps"] < 0, "prescription_type"] = "negative_encoded"

# Store a candidate duration only as an investigation aid.
# This is NOT yet a verified duration field.
df["candidate_duration_seconds"] = pd.NA

negative_mask = df["reps"] < 0

df.loc[
    negative_mask,
    "candidate_duration_seconds"
] = df.loc[negative_mask, "reps"].abs()

# Keep the original reps column unchanged.

# --------------------------------------------------
# 7. SAVE CLEANED RECORDS
# --------------------------------------------------

clean_path = OUTPUT_DIR / "fitness_programs_clean.csv"

df.to_csv(clean_path, index=False)

print("Saved cleaned records:", clean_path)
print("Cleaned dataset shape:", df.shape)

# --------------------------------------------------
# 8. BUILD THE EXERCISE CATALOGUE
# --------------------------------------------------

def join_unique(values):
    """Combine distinct non-empty values into a readable string."""

    unique_values = set()

    for value in values:
        if pd.isna(value):
            continue

        value = str(value).strip()

        if value:
            unique_values.add(value)

    return " | ".join(sorted(unique_values))


catalogue = (
    df.groupby("exercise_key", dropna=False)
    .agg(
        exercise_name=("exercise_name", "first"),
        record_count=("exercise_name", "size"),
        program_count=("title", "nunique"),
        positive_rep_records=(
            "prescription_type",
            lambda s: (s == "repetitions").sum()
        ),
        negative_encoded_records=(
            "prescription_type",
            lambda s: (s == "negative_encoded").sum()
        ),
        goals=("goal_list", lambda s: join_unique(
            item for values in s for item in values
        )),
        levels=("level_list", lambda s: join_unique(
            item for values in s for item in values
        )),
        equipment=("equipment", join_unique),
        median_sets=("sets", "median"),
        median_positive_reps=(
            "reps",
            lambda s: s[s > 0].median()
        ),
        median_intensity=("intensity", "median"),
    )
    .reset_index()
)

# --------------------------------------------------
# 9. SAVE CATALOGUE
# --------------------------------------------------

catalogue_path = OUTPUT_DIR / "exercise_catalogue.csv"

catalogue.to_csv(catalogue_path, index=False)

print("Saved exercise catalogue:", catalogue_path)
print("Catalogue shape:", catalogue.shape)

# --------------------------------------------------
# 10. BASIC VALIDATION
# --------------------------------------------------

print("\n--- CATALOGUE SUMMARY ---")
print("Unique normalized exercise names:", len(catalogue))

print("\nMost frequently recorded exercises:")
print(
    catalogue[
        ["exercise_name", "record_count", "program_count"]
    ]
    .sort_values("record_count", ascending=False)
    .head(15)
    .to_string(index=False)
)

print("\nPrescription types:")
print(df["prescription_type"].value_counts())

print("\nCatalogue preview:")
print(catalogue.head(10).to_string(index=False))
