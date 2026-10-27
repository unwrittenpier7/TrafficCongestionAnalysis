

/**
 * Dashboard header: branding, subtitle and live backend/model status.
 */
export default function Header({ health, healthError, loading }) {
  let statusLabel = 'Checking backend...'
  let statusClass = 'status status--pending'

  if (healthError) {
    statusLabel = 'Backend unreachable'
    statusClass = 'status status--error'
  } else if (health) {
    if (health.model_available) {
      statusLabel = `Model ready (${health.model_name})`
      statusClass = 'status status--ok'
    } else {
      statusLabel = 'Model not trained'
      statusClass = 'status status--warn'
    }
  }

  return (
    <header className="app-header">
      <div className="app-header__brand">
        <div className="app-header__logo" aria-hidden="true">
          TS
        </div>
        <div>
          <h1 className="app-header__title">TrafficSense</h1>
          <p className="app-header__subtitle">
            Traffic Congestion Prediction and Route Analysis Using Machine Learning
          </p>
        </div>
      </div>

      <div className="app-header__status">
        <span className={statusClass} role="status" aria-live="polite">
          <span className="status__dot" aria-hidden="true" />
          {loading ? 'Checking backend...' : statusLabel}
        </span>
        {health && health.dataset_rows ? (
          <span className="app-header__meta">
            Trained on {health.dataset_rows.toLocaleString()} hourly records
          </span>
        ) : null}
      </div>
    </header>
  )
}
