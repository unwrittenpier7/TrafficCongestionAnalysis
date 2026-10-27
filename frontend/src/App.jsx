import { useCallback, useEffect, useState } from 'react'
import Header from './components/Header'
import PredictionForm from './components/PredictionForm'
import PredictionCard from './components/PredictionCard'
import MetricsPanel from './components/MetricsPanel'
import TrafficCharts from './components/TrafficCharts'
import RoutePlanner from './components/RoutePlanner'
import { fetchHealth, fetchMetrics, predictTraffic } from './api'

export default function App() {
  const [health, setHealth] = useState(null)
  const [healthError, setHealthError] = useState(null)
  const [healthLoading, setHealthLoading] = useState(true)

  const [metrics, setMetrics] = useState(null)

  const [prediction, setPrediction] = useState(null)
  const [predictionError, setPredictionError] = useState(null)
  const [predictionLoading, setPredictionLoading] = useState(false)

  useEffect(() => {
    const controller = new AbortController()
    let active = true

    async function load() {
      setHealthLoading(true)
      try {
        const healthPayload = await fetchHealth(controller.signal)
        if (!active) return
        setHealth(healthPayload)
        setHealthError(null)
      } catch (error) {
        if (!active) return
        setHealthError(error.message)
      } finally {
        if (active) setHealthLoading(false)
      }

      try {
        const metricsPayload = await fetchMetrics(controller.signal)
        if (!active) return
        setMetrics(metricsPayload)
      } catch (error) {
        if (!active) return
        setMetrics({ available: false, detail: error.message })
      }
    }

    load()
    return () => {
      active = false
      controller.abort()
    }
  }, [])

  const handlePredict = useCallback(async (values) => {
    setPredictionLoading(true)
    setPredictionError(null)
    setPrediction(null)
    try {
      const result = await predictTraffic(values)
      setPrediction(result)
    } catch (error) {
      // Never show a stale success: the card stays empty on failure.
      setPredictionError(error.message)
    } finally {
      setPredictionLoading(false)
    }
  }, [])

  const modelUnavailable = health && health.model_available === false

  return (
    <div className="app">
      <Header health={health} healthError={healthError} loading={healthLoading} />

      <main className="app-main">
        <section className="section" aria-labelledby="prediction-heading">
          <h2 id="prediction-heading" className="section__heading">
            Predict traffic volume
          </h2>
          <div className="layout layout--two">
            <PredictionForm
              onSubmit={handlePredict}
              loading={predictionLoading}
              error={predictionError}
              disabled={Boolean(modelUnavailable)}
            />
            <PredictionCard prediction={prediction} />
          </div>
          {modelUnavailable ? (
            <p className="alert alert--warn" role="status">
              The backend reports that no trained model is available. Train it with
              <code> python -m app.train</code> inside the backend directory.
            </p>
          ) : null}
        </section>

        <section className="section" aria-labelledby="analytics-heading">
          <h2 id="analytics-heading" className="section__heading">
            Model analytics
          </h2>
          <MetricsPanel metrics={metrics} />
          <TrafficCharts metrics={metrics} />
        </section>

        <section className="section" aria-labelledby="routes-heading">
          <h2 id="routes-heading" className="section__heading">
            Route analysis
          </h2>
          <RoutePlanner />
        </section>

        <footer className="app-footer">
          <p>
            TrafficSense &mdash; B.Tech CSE Machine Learning Project-Based Learning
            project. Predictions use the UCI Metro Interstate Traffic Volume dataset
            (one Minnesota interstate sensor, 2012&ndash;2018) and are historical
            estimates, not live traffic. Road data &copy; OpenStreetMap contributors
            (ODbL).
          </p>
        </footer>
      </main>
    </div>
  )
}
