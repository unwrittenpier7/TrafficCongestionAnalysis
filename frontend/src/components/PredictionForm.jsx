import { useState } from 'react'
import { DAY_NAMES, MONTH_NAMES, WEATHER_CATEGORIES } from '../api'

/**
 * Traffic prediction form. Client-side validation mirrors the backend schema
 * so users get immediate feedback, but the backend remains the source of truth.
 */
const DEFAULTS = {
  hour: 8,
  day_of_week: 1,
  month: 6,
  temperature: 22,
  temperature_unit: 'celsius',
  rainfall_mm: 0,
  snowfall_mm: 0,
  cloud_cover: 30,
  weather_category: 'clear',
  holiday_category: 'None',
}

export default function PredictionForm({ onSubmit, loading, error, disabled }) {
  const [values, setValues] = useState(DEFAULTS)
  const [validationError, setValidationError] = useState(null)

  const update = (field) => (event) => {
    const raw = event.target.value
    const numericFields = [
      'hour',
      'day_of_week',
      'month',
      'temperature',
      'rainfall_mm',
      'snowfall_mm',
      'cloud_cover',
    ]
    setValues((previous) => ({
      ...previous,
      [field]: numericFields.includes(field) ? (raw === '' ? '' : Number(raw)) : raw,
    }))
  }

  const validate = () => {
    const numericChecks = [
      ['hour', 0, 23, 'Hour must be between 0 and 23.'],
      ['day_of_week', 0, 6, 'Day of week must be between Monday (0) and Sunday (6).'],
      ['month', 1, 12, 'Month must be between 1 and 12.'],
      ['rainfall_mm', 0, 500, 'Rainfall must be between 0 and 500 mm.'],
      ['snowfall_mm', 0, 500, 'Snowfall must be between 0 and 500 mm.'],
      ['cloud_cover', 0, 100, 'Cloud cover must be between 0 and 100 percent.'],
    ]
    for (const [field, min, max, message] of numericChecks) {
      const value = values[field]
      if (value === '' || Number.isNaN(value)) return `${message} (value is required)`
      if (value < min || value > max) return message
    }
    if (values.temperature === '' || Number.isNaN(values.temperature)) {
      return 'Temperature is required.'
    }
    return null
  }

  const handleSubmit = (event) => {
    event.preventDefault()
    const problem = validate()
    setValidationError(problem)
    if (problem) return
    onSubmit({
      ...values,
      temperature: Number(values.temperature),
      rainfall_mm: Number(values.rainfall_mm),
      snowfall_mm: Number(values.snowfall_mm),
      cloud_cover: Number(values.cloud_cover),
    })
  }

  const temperatureUnit = values.temperature_unit === 'celsius' ? '\u00B0C' : 'K'

  return (
    <form className="card prediction-form" onSubmit={handleSubmit} noValidate>
      <div className="card__header">
        <h2 className="card__title">Traffic volume prediction</h2>
        <p className="card__subtitle">
          Enter an hour and weather conditions to estimate hourly traffic volume.
        </p>
      </div>

      <div className="form-grid">
        <label className="field">
          <span className="field__label">Hour of day (0-23)</span>
          <input
            className="field__input"
            type="number"
            min="0"
            max="23"
            step="1"
            value={values.hour}
            onChange={update('hour')}
            disabled={disabled}
            required
          />
        </label>

        <label className="field">
          <span className="field__label">Day of week</span>
          <select
            className="field__input"
            value={values.day_of_week}
            onChange={update('day_of_week')}
            disabled={disabled}
          >
            {DAY_NAMES.map((name, index) => (
              <option key={name} value={index}>
                {name}
              </option>
            ))}
          </select>
        </label>

        <label className="field">
          <span className="field__label">Month</span>
          <select
            className="field__input"
            value={values.month}
            onChange={update('month')}
            disabled={disabled}
          >
            {MONTH_NAMES.map((name, index) => (
              <option key={name} value={index + 1}>
                {name}
              </option>
            ))}
          </select>
        </label>

        <label className="field">
          <span className="field__label">
            Temperature ({temperatureUnit})
          </span>
          <input
            className="field__input"
            type="number"
            step="0.1"
            value={values.temperature}
            onChange={update('temperature')}
            disabled={disabled}
            required
          />
        </label>

        <label className="field">
          <span className="field__label">Temperature unit</span>
          <select
            className="field__input"
            value={values.temperature_unit}
            onChange={update('temperature_unit')}
            disabled={disabled}
          >
            <option value="celsius">Celsius</option>
            <option value="kelvin">Kelvin</option>
          </select>
        </label>

        <label className="field">
          <span className="field__label">Rainfall (mm in last hour)</span>
          <input
            className="field__input"
            type="number"
            min="0"
            max="500"
            step="0.01"
            value={values.rainfall_mm}
            onChange={update('rainfall_mm')}
            disabled={disabled}
          />
        </label>

        <label className="field">
          <span className="field__label">Snowfall (mm in last hour)</span>
          <input
            className="field__input"
            type="number"
            min="0"
            max="500"
            step="0.01"
            value={values.snowfall_mm}
            onChange={update('snowfall_mm')}
            disabled={disabled}
          />
        </label>

        <label className="field">
          <span className="field__label">Cloud cover (%)</span>
          <input
            className="field__input"
            type="number"
            min="0"
            max="100"
            step="1"
            value={values.cloud_cover}
            onChange={update('cloud_cover')}
            disabled={disabled}
          />
        </label>

        <label className="field">
          <span className="field__label">Weather category</span>
          <select
            className="field__input"
            value={values.weather_category}
            onChange={update('weather_category')}
            disabled={disabled}
          >
            {WEATHER_CATEGORIES.map((category) => (
              <option key={category} value={category}>
                {category}
              </option>
            ))}
          </select>
        </label>

        <label className="field">
          <span className="field__label">Holiday</span>
          <select
            className="field__input"
            value={values.holiday_category}
            onChange={update('holiday_category')}
            disabled={disabled}
          >
            <option value="None">Not a holiday</option>
            <option value="Christmas Day">Christmas Day</option>
            <option value="Thanksgiving Day">Thanksgiving Day</option>
            <option value="New Years Day">New Years Day</option>
            <option value="Independence Day">Independence Day</option>
            <option value="Labor Day">Labor Day</option>
            <option value="Columbus Day">Columbus Day</option>
            <option value="Veterans Day">Veterans Day</option>
          </select>
        </label>
      </div>

      {validationError ? (
        <p className="alert alert--error" role="alert">
          {validationError}
        </p>
      ) : null}

      {error ? (
        <p className="alert alert--error" role="alert">
          {error}
        </p>
      ) : null}

      <div className="form-actions">
        <button className="button" type="submit" disabled={loading || disabled}>
          {loading ? 'Predicting...' : 'Predict traffic volume'}
        </button>
        <button
          className="button button--ghost"
          type="button"
          onClick={() => {
            setValues(DEFAULTS)
            setValidationError(null)
          }}
          disabled={loading}
        >
          Reset
        </button>
      </div>
    </form>
  )
}
