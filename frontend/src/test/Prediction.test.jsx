
import { describe, expect, it, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import PredictionForm from '../components/PredictionForm'
import PredictionCard from '../components/PredictionCard'

const SAMPLE_PREDICTION = {
  predicted_traffic_volume: 4321.5,
  congestion_category: 'High',
  congestion_thresholds: { low_max: 1211, moderate_max: 4256, high_max: 5502 },
  thresholds_explanation: 'Training-derived thresholds.',
  model_name: 'random_forest',
  model_created_utc: '2026-01-01T00:00:00+00:00',
  input_echo: { hour: 8 },
  disclaimer:
    'This is a historical-data-based estimate and NOT a live traffic measurement.',
}

describe('PredictionForm', () => {
  it('submits parsed numeric values', async () => {
    const user = userEvent.setup()
    const onSubmit = vi.fn()
    render(<PredictionForm onSubmit={onSubmit} loading={false} error={null} disabled={false} />)

    const hourInput = screen.getByLabelText(/hour of day/i)
    await user.clear(hourInput)
    await user.type(hourInput, '17')

    await user.click(screen.getByRole('button', { name: /predict traffic volume/i }))

    expect(onSubmit).toHaveBeenCalledTimes(1)
    const payload = onSubmit.mock.calls[0][0]
    expect(payload.hour).toBe(17)
    expect(typeof payload.temperature).toBe('number')
    expect(payload.weather_category).toBe('clear')
    expect(payload.holiday_category).toBe('None')
  })

  it('shows a validation error for an out-of-range hour and does not submit', async () => {
    const user = userEvent.setup()
    const onSubmit = vi.fn()
    render(<PredictionForm onSubmit={onSubmit} loading={false} error={null} disabled={false} />)

    const hourInput = screen.getByLabelText(/hour of day/i)
    await user.clear(hourInput)
    await user.type(hourInput, '99')
    await user.click(screen.getByRole('button', { name: /predict traffic volume/i }))

    expect(onSubmit).not.toHaveBeenCalled()
    expect(screen.getByRole('alert')).toHaveTextContent(/hour must be between 0 and 23/i)
  })

  it('shows a loading state on the submit button', () => {
    render(<PredictionForm onSubmit={vi.fn()} loading error={null} disabled={false} />)
    const button = screen.getByRole('button', { name: /predicting/i })
    expect(button).toBeDisabled()
  })

  it('renders a backend error message passed from the parent', () => {
    render(
      <PredictionForm
        onSubmit={vi.fn()}
        loading={false}
        error="Backend is unreachable"
        disabled={false}
      />,
    )
    expect(screen.getByRole('alert')).toHaveTextContent('Backend is unreachable')
  })

  it('resets the form to defaults', async () => {
    const user = userEvent.setup()
    render(<PredictionForm onSubmit={vi.fn()} loading={false} error={null} disabled={false} />)
    const hourInput = screen.getByLabelText(/hour of day/i)
    await user.clear(hourInput)
    await user.type(hourInput, '3')
    await user.click(screen.getByRole('button', { name: /reset/i }))
    expect(screen.getByLabelText(/hour of day/i)).toHaveValue(8)
  })
})

describe('PredictionCard', () => {
  it('shows an empty state without a prediction', () => {
    render(<PredictionCard prediction={null} />)
    expect(screen.getByText(/submit the form to see a model-generated/i)).toBeInTheDocument()
  })

  it('renders the real prediction, category, thresholds and disclaimer', () => {
    render(<PredictionCard prediction={SAMPLE_PREDICTION} />)
    expect(screen.getByText('4,322')).toBeInTheDocument()
    // "High" appears both as the result badge and in the thresholds list.
    expect(screen.getAllByText('High').length).toBeGreaterThan(0)
    expect(screen.getByText(/vehicles \/ hour/i)).toBeInTheDocument()
    expect(screen.getByText(/training-derived/i)).toBeInTheDocument()
    expect(screen.getByRole('note')).toHaveTextContent(/NOT a live traffic measurement/i)
  })
})
