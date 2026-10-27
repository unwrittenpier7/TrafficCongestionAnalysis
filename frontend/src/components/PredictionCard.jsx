

const CATEGORY_CLASS = {
  Low: 'badge badge--low',
  Moderate: 'badge badge--moderate',
  High: 'badge badge--high',
  Severe: 'badge badge--severe',
}

/**
 * Renders a real prediction returned by the backend, plus its congestion
 * category, the training-derived thresholds and the historical-estimate
 * disclaimer. Nothing is shown unless a successful response exists.
 */
export default function PredictionCard({ prediction }) {
  if (!prediction) {
    return (
      <div className="card prediction-card prediction-card--empty">
        <div className="card__header">
          <h2 className="card__title">Prediction result</h2>
        </div>
        <p className="muted">
          Submit the form to see a model-generated traffic volume estimate. The
          result will appear here.
        </p>
      </div>
    )
  }

  const thresholds = prediction.congestion_thresholds
  const categoryClass = CATEGORY_CLASS[prediction.congestion_category] || 'badge'

  return (
    <div className="card prediction-card">
      <div className="card__header">
        <h2 className="card__title">Prediction result</h2>
        <p className="card__subtitle">
          Model: {prediction.model_name}
          {prediction.model_created_utc
            ? ` \u00B7 trained ${new Date(prediction.model_created_utc).toLocaleString()}`
            : ''}
        </p>
      </div>

      <div className="prediction-card__value">
        <div>
          <span className="prediction-card__number">
            {Math.round(prediction.predicted_traffic_volume).toLocaleString()}
          </span>
          <span className="prediction-card__unit">vehicles / hour</span>
        </div>
        <span className={categoryClass}>{prediction.congestion_category}</span>
      </div>

      <div className="threshold-list">
        <h3 className="threshold-list__title">Congestion thresholds (training-derived)</h3>
        <ul>
          <li>
            <span className="badge badge--low">Low</span>
            <span>0 &ndash; {Math.round(thresholds.low_max).toLocaleString()} veh/h</span>
          </li>
          <li>
            <span className="badge badge--moderate">Moderate</span>
            <span>
              {Math.round(thresholds.low_max).toLocaleString()} &ndash;{' '}
              {Math.round(thresholds.moderate_max).toLocaleString()} veh/h
            </span>
          </li>
          <li>
            <span className="badge badge--high">High</span>
            <span>
              {Math.round(thresholds.moderate_max).toLocaleString()} &ndash;{' '}
              {Math.round(thresholds.high_max).toLocaleString()} veh/h
            </span>
          </li>
          <li>
            <span className="badge badge--severe">Severe</span>
            <span>above {Math.round(thresholds.high_max).toLocaleString()} veh/h</span>
          </li>
        </ul>
      </div>

      <p className="disclaimer" role="note">
        {prediction.disclaimer}
      </p>
    </div>
  )
}
