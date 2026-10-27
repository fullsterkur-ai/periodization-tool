
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

from train_model import prepare_data, RANDOM_STATE


# Load the same cleaned training data used by the ML pipeline.
df = prepare_data()

# Keep entire program titles in either training or testing,
# so rows from the same program do not leak across the split.
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=RANDOM_STATE,
)

train_pos, test_pos = next(
    splitter.split(df, groups=df["title"])
)

train_df = df.iloc[train_pos].copy()
test_df = df.iloc[test_pos].copy()

print("Training rows:", len(train_df))
print("Testing rows:", len(test_df))
print("Training program titles:", train_df["title"].nunique())
print("Testing program titles:", test_df["title"].nunique())


def evaluate_baselines(target):
    # Missing targets are excluded separately for each prediction task.
    train = train_df.dropna(subset=[target]).copy()
    test = test_df.dropna(subset=[target]).copy()

    if train.empty or test.empty:
        print(f"\nCannot evaluate {target}: no valid training/test rows.")
        return

    # Baseline A: predict the overall training median.
    overall_median = train[target].median()

    overall_predictions = np.full(
        len(test),
        overall_median,
        dtype=float,
    )

    # Baseline B: predict the median for the same exercise and goal.
    # If the combination was not seen during training, use the
    # overall training median instead.
    group_medians = train.groupby(
        ["exercise_name", "goal"]
    )[target].median()

    test_groups = pd.MultiIndex.from_frame(
        test[["exercise_name", "goal"]]
    )

    group_predictions = group_medians.reindex(
        test_groups
    ).to_numpy(dtype=float)

    group_predictions = np.where(
        np.isnan(group_predictions),
        overall_median,
        group_predictions,
    )

    actual = test[target].to_numpy(dtype=float)

    print(f"\n{'=' * 55}")
    print(f"TARGET: {target.upper()}")
    print(f"{'=' * 55}")
    print("Training target rows:", len(train))
    print("Testing target rows:", len(test))
    print("Training median:", round(overall_median, 3))

    for name, predictions in [
        ("Overall median baseline", overall_predictions),
        ("Exercise + goal median baseline", group_predictions),
    ]:
        mae = mean_absolute_error(actual, predictions)
        rmse = np.sqrt(mean_squared_error(actual, predictions))

        print(f"\n{name}")
        print(f"MAE:  {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")


for target_name in ["sets", "reps"]:
    evaluate_baselines(target_name)
