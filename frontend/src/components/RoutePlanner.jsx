import { useMemo, useState } from 'react'
import { planRoutes } from '../api'
import RouteMap, { ROUTE_COLORS } from './RouteMap'

const DEFAULT_STUDY_AREA = {
  latitude: 17.44,
  longitude: 78.35,
  distance_m: 600,
}

const DEFAULT_FORM = {
  originLat: 17.4388,
  originLon: 78.3504,
  destLat: 17.4442,
  destLon: 78.3537,
  scenario: 'simulated_peak',
  severity: 1.0,
}

/**
 * Route planner: coordinate inputs plus map-click selection, alternative route
 * comparison and an interactive OpenStreetMap view.
 */
export default function RoutePlanner() {
  const [form, setForm] = useState(DEFAULT_FORM)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)
  const [pickMode, setPickMode] = useState('origin')

  const update = (field) => (event) => {
    const raw = event.target.value
    setForm((previous) => ({
      ...previous,
      [field]: field === 'scenario' ? raw : raw === '' ? '' : Number(raw),
    }))
  }

  const handleMapPick = ([lat, lon]) => {
    setForm((previous) =>
      pickMode === 'origin'
        ? { ...previous, originLat: Number(lat.toFixed(6)), originLon: Number(lon.toFixed(6)) }
        : { ...previous, destLat: Number(lat.toFixed(6)), destLon: Number(lon.toFixed(6)) },
    )
  }

  const validate = () => {
    const checks = [
      ['originLat', -90, 90, 'Origin latitude must be between -90 and 90.'],
      ['originLon', -180, 180, 'Origin longitude must be between -180 and 180.'],
      ['destLat', -90, 90, 'Destination latitude must be between -90 and 90.'],
      ['destLon', -180, 180, 'Destination longitude must be between -180 and 180.'],
    ]
    for (const [field, min, max, message] of checks) {
      const value = form[field]
      if (value === '' || Number.isNaN(value) || value < min || value > max) return message
    }
    if (
      form.originLat === form.destLat &&
      form.originLon === form.destLon
    ) {
      return 'Origin and destination must be different points.'
    }
    return null
  }

  const handleSubmit = async (event) => {
    event.preventDefault()
    const problem = validate()
    setError(problem)
    if (problem) return

    setLoading(true)
    setResult(null)
    try {
      const payload = await planRoutes({
        origin: { latitude: Number(form.originLat), longitude: Number(form.originLon) },
        destination: { latitude: Number(form.destLat), longitude: Number(form.destLon) },
        scenario: form.scenario,
        severity: Number(form.severity),
        max_alternatives: 3,
      })
      setResult(payload)
    } catch (requestError) {
      setError(requestError.message)
    } finally {
      setLoading(false)
    }
  }

  const routes = useMemo(() => result?.routes || [], [result])

  const mapCenter = useMemo(() => {
    if (!routes.length) {
      return [
        (Number(form.originLat) + Number(form.destLat)) / 2 || DEFAULT_STUDY_AREA.latitude,
        (Number(form.originLon) + Number(form.destLon)) / 2 || DEFAULT_STUDY_AREA.longitude,
      ]
    }
    const points = routes[0].geometry
    return points[Math.floor(points.length / 2)]
  }, [routes, form.originLat, form.originLon, form.destLat, form.destLon])

  const scenarioProvenance = result?.scenario?.provenance
  const provenanceLabel = {
    simulated: 'SIMULATED congestion scenario (not measured traffic)',
    free_flow: 'No congestion model (free-flow travel times only)',
    custom: 'Caller-supplied congestion multipliers',
  }[scenarioProvenance]

  return (
    <div className="card">
      <div className="card__header">
        <h2 className="card__title">Route planner and congestion-aware analysis</h2>
        <p className="card__subtitle">
          Real OpenStreetMap road network routed with NetworkX. Click the map to set the
          origin or destination.
        </p>
      </div>

      <form className="route-form" onSubmit={handleSubmit} noValidate>
        <div className="form-grid">
          <label className="field">
            <span className="field__label">Origin latitude</span>
            <input
              className="field__input"
              type="number"
              step="0.0001"
              value={form.originLat}
              onChange={update('originLat')}
              disabled={loading}
            />
          </label>
          <label className="field">
            <span className="field__label">Origin longitude</span>
            <input
              className="field__input"
              type="number"
              step="0.0001"
              value={form.originLon}
              onChange={update('originLon')}
              disabled={loading}
            />
          </label>
          <label className="field">
            <span className="field__label">Destination latitude</span>
            <input
              className="field__input"
              type="number"
              step="0.0001"
              value={form.destLat}
              onChange={update('destLat')}
              disabled={loading}
            />
          </label>
          <label className="field">
            <span className="field__label">Destination longitude</span>
            <input
              className="field__input"
              type="number"
              step="0.0001"
              value={form.destLon}
              onChange={update('destLon')}
              disabled={loading}
            />
          </label>

          <label className="field">
            <span className="field__label">Route cost model</span>
            <select
              className="field__input"
              value={form.scenario}
              onChange={update('scenario')}
              disabled={loading}
            >
              <option value="simulated_peak">Simulated peak congestion (edge-specific)</option>
              <option value="free_flow">Free flow (no congestion)</option>
            </select>
          </label>

          <label className="field">
            <span className="field__label">Simulated severity ({form.severity})</span>
            <input
              className="field__input"
              type="range"
              min="0"
              max="3"
              step="0.1"
              value={form.severity}
              onChange={update('severity')}
              disabled={loading || form.scenario !== 'simulated_peak'}
            />
          </label>
        </div>

        <div className="form-actions">
          <button className="button" type="submit" disabled={loading}>
            {loading ? 'Calculating routes...' : 'Find routes'}
          </button>
          <div className="pick-toggle" role="group" aria-label="Map click target">
            <button
              type="button"
              className={`button button--ghost ${pickMode === 'origin' ? 'button--active' : ''}`}
              onClick={() => setPickMode('origin')}
            >
              Set origin by click
            </button>
            <button
              type="button"
              className={`button button--ghost ${pickMode === 'destination' ? 'button--active' : ''}`}
              onClick={() => setPickMode('destination')}
            >
              Set destination by click
            </button>
          </div>
        </div>
      </form>

      {error ? (
        <p className="alert alert--error" role="alert">
          {error}
        </p>
      ) : null}

      {provenanceLabel ? (
        <p
          className={`alert ${
            scenarioProvenance === 'simulated' ? 'alert--warn' : 'alert--info'
          }`}
          role="note"
        >
          Congestion data provenance: {provenanceLabel}. {result?.scenario?.note}
        </p>
      ) : null}

      <RouteMap
        center={mapCenter}
        routes={routes}
        origin={[Number(form.originLat), Number(form.originLon)]}
        destination={[Number(form.destLat), Number(form.destLon)]}
        onPick={handleMapPick}
      />

      {routes.length ? (
        <>
          <div className="route-legend">
            {routes.map((route, index) => (
              <span key={`${route.label}-legend`} className="route-legend__item">
                <span
                  className="route-legend__swatch"
                  style={{ background: ROUTE_COLORS[index % ROUTE_COLORS.length] }}
                  aria-hidden="true"
                />
                #{route.rank} {route.label.replaceAll('_', ' ')}
              </span>
            ))}
          </div>

          <table className="table">
            <caption className="muted">
              Route comparison. Times are estimates from OpenStreetMap speed limits.
            </caption>
            <thead>
              <tr>
                <th scope="col">#</th>
                <th scope="col">Route</th>
                <th scope="col">Distance (km)</th>
                <th scope="col">Free-flow (min)</th>
                <th scope="col">Congestion-aware (min)</th>
                <th scope="col">Mean multiplier</th>
              </tr>
            </thead>
            <tbody>
              {routes.map((route) => (
                <tr key={`row-${route.rank}`}>
                  <td>{route.rank}</td>
                  <td>{route.label.replaceAll('_', ' ')}</td>
                  <td>{route.distance_km.toFixed(3)}</td>
                  <td>{route.free_flow_time_minutes.toFixed(2)}</td>
                  <td>{route.congestion_aware_time_minutes.toFixed(2)}</td>
                  <td>{route.mean_congestion_multiplier.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <p className="muted">
            Road network: {result.graph.nodes} nodes / {result.graph.edges} edges.{' '}
            {result.attribution}
          </p>
          <ul className="limitations">
            {result.limitations.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </>
      ) : (
        <p className="muted">
          No routes calculated yet. Submit the form to compare alternative routes.
        </p>
      )}
    </div>
  )
}
