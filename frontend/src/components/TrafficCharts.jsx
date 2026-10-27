import { useMemo, useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

const BLUE = '#2563eb'
const AMBER = '#d97706'
const TEAL = '#0f766e'

const CHART_TABS = [
  { id: 'time', label: 'Actual vs predicted' },
  { id: 'hour', label: 'By hour' },
  { id: 'weekday', label: 'By weekday' },
  { id: 'importance', label: 'Feature importance' },
]

/**
 * Charts built ONLY from the metrics file produced by the training run.
 * No synthetic series are generated here.
 */
export default function TrafficCharts({ metrics }) {
  const [tab, setTab] = useState('time')

  const data = useMemo(
    () => (metrics && metrics.available ? metrics.metrics || {} : {}),
    [metrics],
  )
  const byHour = data.traffic_by_hour || []
  const byWeekday = data.traffic_by_weekday || []

  const timeSeries = useMemo(() => {
    const actualVsPredicted = data.actual_vs_predicted || {}
    const timestamps = actualVsPredicted.timestamps || []
    const actual = actualVsPredicted.actual || []
    const predicted = actualVsPredicted.predicted || []
    const points = Math.min(timestamps.length, actual.length, predicted.length)
    return Array.from({ length: points }, (_, index) => ({
      index,
      label: timestamps[index],
      actual: actual[index],
      predicted: predicted[index],
    }))
  }, [data])

  const scatterData = useMemo(
    () =>
      timeSeries.map((point) => ({
        actual: point.actual,
        predicted: point.predicted,
      })),
    [timeSeries],
  )

  const importanceTop = useMemo(
    () => (data.feature_importance || []).slice(0, 12).map((item) => ({ ...item })),
    [data],
  )

  if (!metrics || !metrics.available) {
    return (
      <div className="card">
        <div className="card__header">
          <h2 className="card__title">Traffic analytics</h2>
        </div>
        <p className="alert alert--warn" role="status">
          {metrics ? metrics.detail : 'Metrics are loading...'}
        </p>
      </div>
    )
  }

  return (
    <div className="card">
      <div className="card__header">
        <h2 className="card__title">Traffic analytics</h2>
        <p className="card__subtitle">
          All charts use real evaluation output saved by the training script.
        </p>
      </div>

      <div className="tabs" role="tablist" aria-label="Analytics charts">
        {CHART_TABS.map((item) => (
          <button
            key={item.id}
            type="button"
            role="tab"
            aria-selected={tab === item.id}
            className={`tab ${tab === item.id ? 'tab--active' : ''}`}
            onClick={() => setTab(item.id)}
          >
            {item.label}
          </button>
        ))}
      </div>

      <div className="chart">
        {tab === 'time' ? (
          <>
            <h3 className="section-title">Actual vs predicted volume (test period)</h3>
            <ResponsiveContainer width="100%" height={320}>
              <LineChart data={timeSeries} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="index" tick={{ fontSize: 11 }} minTickGap={40} />
                <YAxis tick={{ fontSize: 11 }} width={70} />
                <Tooltip
                  formatter={(value) => Number(value).toLocaleString()}
                  labelFormatter={(index) =>
                    timeSeries[index] ? `Observation: ${timeSeries[index].label}` : ''
                  }
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="actual"
                  name="Actual"
                  stroke={BLUE}
                  dot={false}
                  strokeWidth={1.6}
                />
                <Line
                  type="monotone"
                  dataKey="predicted"
                  name="Predicted"
                  stroke={AMBER}
                  dot={false}
                  strokeWidth={1.6}
                />
              </LineChart>
            </ResponsiveContainer>

            <h3 className="section-title">Predicted vs actual scatter</h3>
            <ResponsiveContainer width="100%" height={280}>
              <ScatterChart margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis
                  type="number"
                  dataKey="actual"
                  name="Actual"
                  tick={{ fontSize: 11 }}
                  domain={['auto', 'auto']}
                />
                <YAxis
                  type="number"
                  dataKey="predicted"
                  name="Predicted"
                  tick={{ fontSize: 11 }}
                  width={70}
                  domain={['auto', 'auto']}
                />
                <Tooltip formatter={(value) => Number(value).toLocaleString()} />
                <Scatter data={scatterData} fill={BLUE} fillOpacity={0.45} />
              </ScatterChart>
            </ResponsiveContainer>
          </>
        ) : null}

        {tab === 'hour' ? (
          <>
            <h3 className="section-title">Mean traffic volume by hour of day</h3>
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={byHour} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="hour" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} width={70} />
                <Tooltip formatter={(value) => Number(value).toLocaleString()} />
                <Bar dataKey="mean" name="Mean volume" fill={BLUE} radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </>
        ) : null}

        {tab === 'weekday' ? (
          <>
            <h3 className="section-title">Mean traffic volume by day of week</h3>
            <ResponsiveContainer width="100%" height={320}>
              <BarChart
                data={byWeekday}
                margin={{ top: 8, right: 16, bottom: 8, left: 0 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="day_name" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} width={70} />
                <Tooltip formatter={(value) => Number(value).toLocaleString()} />
                <Bar dataKey="mean" name="Mean volume" fill={TEAL} radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </>
        ) : null}

        {tab === 'importance' ? (
          <>
            <h3 className="section-title">
              Feature importance (top {importanceTop.length || 0})
            </h3>
            {importanceTop.length ? (
              <ResponsiveContainer width="100%" height={Math.max(320, importanceTop.length * 34)}>
                <BarChart
                  data={importanceTop}
                  layout="vertical"
                  margin={{ top: 8, right: 24, bottom: 8, left: 40 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis type="number" tick={{ fontSize: 11 }} />
                  <YAxis
                    type="category"
                    dataKey="feature"
                    tick={{ fontSize: 11 }}
                    width={150}
                  />
                  <Tooltip formatter={(value) => Number(value).toFixed(4)} />
                  <Bar dataKey="importance" name="Importance" radius={[0, 3, 3, 0]}>
                    {importanceTop.map((entry) => (
                      <Cell key={entry.feature} fill={BLUE} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <p className="muted">
                Feature importance is not available for this model type.
              </p>
            )}
          </>
        ) : null}
      </div>
    </div>
  )
}
