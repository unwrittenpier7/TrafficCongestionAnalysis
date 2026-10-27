

function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return '\u2014'
  return Number(value).toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })
}

/**
 * Displays the real evaluation metrics saved by the training script, including
 * the baseline-versus-model comparison and the confusion matrix.
 */
export default function MetricsPanel({ metrics }) {
  if (!metrics) return null

  if (!metrics.available) {
    return (
      <div className="card">
        <div className="card__header">
          <h2 className="card__title">Model evaluation</h2>
        </div>
        <p className="alert alert--warn" role="status">
          {metrics.detail}
        </p>
      </div>
    )
  }

  const data = metrics.metrics || {}
  const test = data.test || {}
  const baseline = data.baseline_test || {}
  const classification = data.classification_test || {}
  const metadata = metrics.metadata || {}
  const periods = metadata.split?.periods || {}

  const maeImprovement =
    baseline.mae && test.mae ? ((baseline.mae - test.mae) / baseline.mae) * 100 : null

  const labels = classification.labels || []
  const matrix = classification.confusion_matrix || []

  return (
    <div className="card">
      <div className="card__header">
        <h2 className="card__title">Model evaluation</h2>
        <p className="card__subtitle">
          Chronological holdout test set &middot; {metadata.selected_model}
          {periods.test ? ` \u00B7 ${periods.test.rows?.toLocaleString()} test rows` : ''}
        </p>
      </div>

      <div className="metric-grid">
        <div className="metric">
          <span className="metric__label">MAE (veh/h)</span>
          <span className="metric__value">{formatNumber(test.mae, 1)}</span>
        </div>
        <div className="metric">
          <span className="metric__label">RMSE (veh/h)</span>
          <span className="metric__value">{formatNumber(test.rmse, 1)}</span>
        </div>
        <div className="metric">
          <span className="metric__label">R&sup2;</span>
          <span className="metric__value">{formatNumber(test.r2, 3)}</span>
        </div>
        <div className="metric">
          <span className="metric__label">Macro F1 (categories)</span>
          <span className="metric__value">{formatNumber(classification.macro?.f1, 3)}</span>
        </div>
      </div>

      <div className="comparison">
        <h3 className="section-title">Baseline vs model (same test rows)</h3>
        <table className="table">
          <thead>
            <tr>
              <th scope="col">Model</th>
              <th scope="col">MAE</th>
              <th scope="col">RMSE</th>
              <th scope="col">R&sup2;</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Training-mean baseline</td>
              <td>{formatNumber(baseline.mae, 1)}</td>
              <td>{formatNumber(baseline.rmse, 1)}</td>
              <td>{formatNumber(baseline.r2, 3)}</td>
            </tr>
            <tr>
              <td>
                <strong>{metadata.selected_model}</strong>
              </td>
              <td>
                <strong>{formatNumber(test.mae, 1)}</strong>
              </td>
              <td>
                <strong>{formatNumber(test.rmse, 1)}</strong>
              </td>
              <td>
                <strong>{formatNumber(test.r2, 3)}</strong>
              </td>
            </tr>
          </tbody>
        </table>
        {maeImprovement !== null ? (
          <p className="muted">
            The selected model reduces MAE by {formatNumber(maeImprovement, 1)}% compared
            with the training-mean baseline.
          </p>
        ) : null}
      </div>

      {labels.length && matrix.length ? (
        <div className="comparison">
          <h3 className="section-title">Confusion matrix (congestion categories)</h3>
          <table className="table table--matrix">
            <thead>
              <tr>
                <th scope="col">Actual \ Predicted</th>
                {labels.map((label) => (
                  <th key={label} scope="col">
                    {label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {matrix.map((row, rowIndex) => (
                <tr key={labels[rowIndex]}>
                  <th scope="row">{labels[rowIndex]}</th>
                  {row.map((value, columnIndex) => (
                    <td
                      key={`${rowIndex}-${columnIndex}`}
                      className={rowIndex === columnIndex ? 'cell--diagonal' : undefined}
                    >
                      {value}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </div>
  )
}
