
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

from train_model import (
    prepare_data,
    build_pipeline,
    FEATURES,
    RANDOM_STATE,
)

MAX_TRAIN_ROWS = 30_000

df = prepare_data()

# Keep program titles separate between training and testing.
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

print("Training partition:", len(train_df))
print("Held-out partition:", len(test_df))


def evaluate_target(target):
    train = train_df.dropna(subset=[target]).copy()
    test = test_df.dropna(subset=[target]).copy()

    # Match the sample size used in the earlier learning-curve experiment.
    if len(train) > MAX_TRAIN_ROWS:
        train = train.sample(
            n=MAX_TRAIN_ROWS,
            random_state=RANDOM_STATE,
        )

    X_train = train[FEATURES]
    y_train = train[target]
    X_test = test[FEATURES]
    y_test = test[target].to_numpy(dtype=float)

    # Model: Random Forest pipeline from training_data.py.
    model = build_pipeline()
    model.fit(X_train, y_train)
    rf_predictions = model.predict(X_test)

    # Baseline A: overall median from the SAME training sample.
    overall_median = y_train.median()
    overall_predictions = np.full(
        len(test),
        overall_median,
        dtype=float,
    )

    # Baseline B: exercise + goal median from the SAME sample.
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

    methods = [
        ("Overall median", overall_predictions),
        ("Exercise + goal median", group_predictions),
        ("Random Forest", rf_predictions),
    ]

    print(f"\n{'=' * 55}")
    print(f"TARGET: {target.upper()}")
    print(f"Training rows used: {len(train)}")
    print(f"Testing rows used: {len(test)}")
    print(f"{'=' * 55}")

    for name, predictions in methods:
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(
            mean_squared_error(y_test, predictions)
        )

        print(
            f"{name:25s} "
            f"MAE={mae:.4f}  RMSE={rmse:.4f}"
        )


for target in ["sets", "reps"]:
    evaluate_target(target)
