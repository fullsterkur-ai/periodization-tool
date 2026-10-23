
import { useEffect, useState } from "react";
import {
  fetchTrainingDashboard,
} from "./api";
import type {
  TrainingDashboardData,
  TrainingTargetData,
} from "./api";
import "./ModelTraining.css";

function ScatterPlot({
  points,
}: {
  points: TrainingTargetData["diagnostics"]["actual_vs_predicted"];
}) {
  if (!points.length) return <p>No prediction samples available.</p>;

  const values = points.flatMap((p) => [p.actual, p.predicted]);
  const min = Math.min(0, ...values);
  const max = Math.max(1, ...values);
  const range = max - min || 1;
  const x = (v: number) => 42 + ((v - min) / range) * 350;
  const y = (v: number) => 260 - ((v - min) / range) * 230;

  return (
    <svg viewBox="0 0 420 310" className="training-chart" role="img"
      aria-label="Actual versus predicted values">
      <line x1="42" y1="260" x2="392" y2="260" stroke="currentColor" />
      <line x1="42" y1="30" x2="42" y2="260" stroke="currentColor" />
      <line x1={x(min)} y1={y(min)} x2={x(max)} y2={y(max)}
        stroke="currentColor" strokeDasharray="5 5" opacity=".65" />
      {points.map((p, i) => (
        <circle
          key={i}
          cx={x(p.actual)}
          cy={y(p.predicted)}
          r="3"
          fill="#3977d5"
          opacity=".6"
        >
          <title>{`Actual: ${p.actual}; predicted: ${p.predicted}`}</title>
        </circle>
      ))}
      <text x="217" y="300" textAnchor="middle">Actual value</text>
      <text x="13" y="145" transform="rotate(-90 13 145)"
        textAnchor="middle">Predicted value</text>
    </svg>
  );
}

function ImportanceChart({
  items,
}: {
  items: TrainingTargetData["diagnostics"]["feature_importance"];
}) {
  const max = Math.max(0.000001, ...items.map((item) => item.importance));

  return (
    <div className="importance-list">
      {items.map((item) => (
        <div className="importance-row" key={item.feature}>
          <span title={item.feature}>{item.feature.replace(/^(categorical|numeric)__/, "")}</span>
          <div className="importance-track">
            <div
              className="importance-fill"
              style={{ width: `${(item.importance / max) * 100}%` }}
            />
          </div>
          <strong>{item.importance.toFixed(4)}</strong>
        </div>
      ))}
    </div>
  );
}

function ResidualChart({
  counts,
  binEdges,
}: {
  counts: number[];
  binEdges: number[];
}) {
  const maxCount = Math.max(1, ...counts);
  const barWidth = 100 / Math.max(1, counts.length);

  return (
    <div className="residual-chart">
      {counts.map((count, i) => (
        <div className="residual-bin" key={i}>
          <div
            className="residual-bar"
            style={{ height: `${(count / maxCount) * 100}%` }}
            title={`${count} errors: ${binEdges[i]} to ${binEdges[i + 1]}`}
          />
          <small>{Number(binEdges[i]).toFixed(1)}</small>
          <span style={{ width: `${barWidth}%` }} />
        </div>
      ))}
    </div>
  );
}

function LearningCurveChart({
  rows,
  trainMae,
  validationMae,
}: {
  rows: number[];
  trainMae: number[];
  validationMae: number[];
}) {
  if (
    rows.length === 0 ||
    rows.length !== trainMae.length ||
    rows.length !== validationMae.length
  ) {
    return <p>Learning-curve data is unavailable.</p>;
  }

  const left = 55;
  const right = 400;
  const top = 25;
  const bottom = 255;

  const minRows = Math.min(...rows);
  const maxRows = Math.max(...rows);
  const maxError = Math.max(
    0.001,
    ...trainMae,
    ...validationMae,
  );

  const x = (value: number) =>
    left +
    ((value - minRows) / (maxRows - minRows || 1)) *
      (right - left);

  const y = (value: number) =>
    bottom - (value / maxError) * (bottom - top);

  const trainPoints = rows
    .map((row, i) => `${x(row)},${y(trainMae[i])}`)
    .join(" ");

  const validationPoints = rows
    .map((row, i) => `${x(row)},${y(validationMae[i])}`)
    .join(" ");

  return (
    <div>
      <svg
        viewBox="0 0 430 310"
        className="training-chart"
        role="img"
        aria-label="Training and validation MAE versus training rows"
      >
        <line
          x1={left}
          y1={bottom}
          x2={right}
          y2={bottom}
          stroke="currentColor"
        />
        <line
          x1={left}
          y1={top}
          x2={left}
          y2={bottom}
          stroke="currentColor"
        />

        <polyline
          points={trainPoints}
          fill="none"
          stroke="#3977d5"
          strokeWidth="3"
        />

        <polyline
          points={validationPoints}
          fill="none"
          stroke="#e08b36"
          strokeWidth="3"
        />

        {rows.map((row, i) => (
          <g key={row}>
            <circle
              cx={x(row)}
              cy={y(trainMae[i])}
              r="4"
              fill="#3977d5"
            >
              <title>
                {`Training: ${row} rows, MAE ${trainMae[i]}`}
              </title>
            </circle>
            <circle
              cx={x(row)}
              cy={y(validationMae[i])}
              r="4"
              fill="#e08b36"
            >
              <title>
                {`Validation: ${row} rows, MAE ${validationMae[i]}`}
              </title>
            </circle>
          </g>
        ))}

        <text x="220" y="295" textAnchor="middle">
          Training rows
        </text>
        <text
          x="15"
          y="145"
          textAnchor="middle"
          transform="rotate(-90 15 145)"
        >
          Mean absolute error (MAE)
        </text>
      </svg>

      <div className="learning-legend">
        <span><i className="legend-training" /> Training MAE</span>
        <span><i className="legend-validation" /> Validation MAE</span>
      </div>
    </div>
  );
}

function TargetPanel({
  title,
  target,
}: {
  title: string;
  target: TrainingTargetData;
}) {
  const { metrics, diagnostics } = target;
const curve = diagnostics.learning_curve;
  return (
    <section className="training-section">
      <h2>{title} prediction model</h2>

      <div className="training-metrics">
        <article>
          <span>MAE</span>
          <strong>{metrics.mae.toFixed(3)}</strong>
        </article>
        <article>
          <span>RMSE</span>
          <strong>{metrics.rmse.toFixed(3)}</strong>
        </article>
        <article>
          <span>Training rows</span>
          <strong>{metrics.train_rows.toLocaleString()}</strong>
        </article>
        <article>
          <span>Test rows</span>
          <strong>{metrics.test_rows.toLocaleString()}</strong>
        </article>
      </div>
      <article className="training-card">
  <h3>Learning curve</h3>
  <p>
    Mean absolute error at increasing training-set sizes.
    Lower values indicate smaller prediction errors.
  </p>

  <LearningCurveChart
    rows={curve.training_rows}
    trainMae={curve.train_mae}
    validationMae={curve.validation_mae}
  />

  <small>
    Diagnostic model: {curve.diagnostic_estimators} trees.
    Validation uses held-out programme titles within the
    outer training partition.
  </small>
</article>

      <div className="training-grid">
        <article className="training-card">
          <h3>Actual vs. predicted</h3>
          <p>Each point represents a held-out evaluation sample.</p>
          <ScatterPlot points={diagnostics.actual_vs_predicted} />
          <small>
            Dashed line = perfect prediction. Points are sampled from
            the evaluation set.
          </small>
        </article>

        <article className="training-card">
          <h3>Feature importance</h3>
          <p>Relative importance reported by the fitted Random Forest.</p>
          <ImportanceChart items={diagnostics.feature_importance} />
          <small>
            Importance indicates model reliance, not causation.
          </small>
        </article>

        <article className="training-card">
          <h3>Prediction residuals</h3>
          <p>Residual = actual value minus predicted value.</p>
          <ResidualChart
            counts={diagnostics.residual_histogram.counts}
            binEdges={diagnostics.residual_histogram.bin_edges}
          />
          <small>
            Values closer to zero indicate smaller prediction errors.
          </small>
        </article>
      </div>
    </section>
  );
}

export default function ModelTraining() {
  const [data, setData] = useState<TrainingDashboardData | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    fetchTrainingDashboard()
      .then((result) => {
        if (!cancelled) setData(result);
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(
            err instanceof Error
              ? err.message
              : "Could not load model training results."
          );
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, []);

  if (loading) return <div className="panel">Loading model results…</div>;

  if (error) {
    return (
      <div className="panel" role="alert">
        <h2>Training results unavailable</h2>
        <p>{error}</p>
        <p>
          Run the training script successfully and confirm that
          models/training_dashboard.json exists.
        </p>
      </div>
    );
  }

  if (!data) return null;

  return (
    <div className="model-training-page">
      <div className="panel">
        <p className="eyebrow">MODEL EVALUATION</p>
        <h2>{data.model_type}</h2>
        <p>{data.evaluation_method}</p>
        <p>
          Training row cap: {data.max_train_rows.toLocaleString()} ·
          Random seed: {data.random_state}
        </p>
        <p>
          Input features: {data.features.join(", ")}
        </p>
      </div>

      <TargetPanel title="Sets" target={data.targets.sets} />
      <TargetPanel title="Repetitions" target={data.targets.reps} />

      <div className="panel">
        <h2>Research limitations</h2>
        <ul>
          {data.limitations.map((item) => <li key={item}>{item}</li>)}
        </ul>
      </div>
    </div>
  );
}
