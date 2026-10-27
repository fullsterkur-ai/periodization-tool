
import pandas as pd

from train_model import prepare_data

df = prepare_data()

for target in ["sets", "reps"]:
    values = df[target].dropna()

    print(f"\n{'=' * 45}")
    print(f"TARGET: {target}")
    print(f"{'=' * 45}")

    print("Valid rows:", len(values))
    print("\nDistribution:")
    print(
        values.describe(
            percentiles=[0.10, 0.25, 0.50, 0.75, 0.90]
        ).round(2)
    )

    print("\nMost common values:")
    print(values.value_counts().head(15).sort_index())

    print("\nUnique values:", values.nunique())
    print("Missing target values:", df[target].isna().sum())

print("\nUnique program titles:", df["title"].nunique())
print("Unique exercises:", df["exercise_name"].nunique())

