
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, learning_curve

from train_model import (
    FEATURES,
    MAX_TRAIN_ROWS,
    RANDOM_STATE,
    build_pipeline,
    prepare_data,
)

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "models" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Keep the experiment manageable on a laptop.
MAX_CURVE_ROWS = 30_000


def evaluate_target(df, target):
    usable = df.dropna(subset=[target]).copy()

    X = usable[FEATURES]
    y = usable[target]
    groups = usable["title"]

    # Hold out entire program titles for final evaluation.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups=groups)
    )

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    groups_train = groups.iloc[train_idx]

    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    # Limit the curve experiment's training pool.
    if len(X_train) > MAX_CURVE_ROWS:
        rng = np.random.default_rng(RANDOM_STATE)
        selected = np.sort(
            rng.choice(
                len(X_train),
                size=MAX_CURVE_ROWS,
                replace=False,
            )
        )

        X_train = X_train.iloc[selected]
        y_train = y_train.iloc[selected]
        groups_train = groups_train.iloc[selected]

    # Three group-based validation folds.
    cv = GroupKFold(n_splits=3)

    sizes, train_scores, val_scores = learning_curve(
        estimator=build_pipeline(),
        X=X_train,
        y=y_train,
        groups=groups_train,
        train_sizes=np.linspace(0.2, 1.0, 5),
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        shuffle=True,
        random_state=RANDOM_STATE,
        error_score="raise",
    )

    train_mae = -train_scores.mean(axis=1)
    val_mae = -val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    # Fit once on the curve's training pool and evaluate
    # on the untouched held-out program titles.
    final_model = build_pipeline()
    final_model.fit(X_train, y_train)
    predictions = final_model.predict(X_test)

    final_metrics = {
        "target": target,
        "curve_training_rows": int(len(X_train)),
        "held_out_test_rows": int(len(X_test)),
        "held_out_test_program_titles": int(
            groups.iloc[test_idx].nunique()
        ),
        "held_out_mae": float(
            mean_absolute_error(y_test, predictions)
        ),
        "held_out_rmse": float(
            np.sqrt(mean_squared_error(y_test, predictions))
        ),
    }

    # Save the curve's numerical data.
    curve_df = pd.DataFrame({
        "training_samples": sizes,
        "training_mae": train_mae,
        "validation_mae": val_mae,
        "validation_score_std": val_std,
    })

    curve_df.to_csv(
        OUTPUT_DIR / f"{target}_learning_curve.csv",
        index=False,
    )

    # Plot MAE: lower is better.
    plt.figure(figsize=(9, 5))
    plt.plot(
        sizes, train_mae, marker="o",
        label="Training MAE",
    )
    plt.plot(
        sizes, val_mae, marker="o",
        label="Cross-validation MAE",
    )

    plt.xlabel("Training samples")
    plt.ylabel("Mean Absolute Error")
    plt.title(f"Random Forest Learning Curve — {target.title()}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"{target}_learning_curve.png",
        dpi=160,
    )
    plt.close()

    return final_metrics, curve_df


def main():
    df = prepare_data()

    results = []

    for target in ["sets", "reps"]:
        print(f"\nEvaluating target: {target}")

        metrics, curve = evaluate_target(df, target)
        results.append(metrics)

        print("\nLearning curve:")
        print(curve.round(3).to_string(index=False))

        print("\nHeld-out test results:")
        print(pd.Series(metrics).to_string())

    pd.DataFrame(results).to_csv(
        OUTPUT_DIR / "held_out_metrics.csv",
        index=False,
    )

    print(f"\nEvaluation files saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
