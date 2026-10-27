
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

// Leaflet needs real layout APIs that jsdom lacks; replace the map with a stub.
vi.mock('../components/RouteMap', () => ({
  default: ({ routes, origin, destination }) => (
    <div data-testid="route-map">
      <span data-testid="map-origin">{origin.join(',')}</span>
      <span data-testid="map-destination">{destination.join(',')}</span>
      <span data-testid="map-route-count">{routes.length}</span>
    </div>
  ),
  ROUTE_COLORS: ['#2563eb', '#d97706'],
}))

import RoutePlanner from '../components/RoutePlanner'
import { planRoutes } from '../api'

vi.mock('../api', () => ({
  planRoutes: vi.fn(),
}))

const ROUTE_RESPONSE = {
  scenario: {
    name: 'simulated_peak',
    description: 'Simulated peak-hour congestion.',
    provenance: 'simulated',
    is_uniform_multiplier: false,
    note: 'Multipliers differ per road segment.',
  },
  origin_snap: { requested: [17.44, 78.35], node: [17.44, 78.35], node_id: 1 },
  destination_snap: { requested: [17.45, 78.36], node: [17.45, 78.36], node_id: 2 },
  routes: [
    {
      rank: 1,
      label: 'congestion_aware_fastest',
      distance_km: 2.345,
      free_flow_time_minutes: 3.21,
      congestion_aware_time_minutes: 4.56,
      mean_congestion_multiplier: 1.42,
      max_congestion_multiplier: 1.8,
      highway_mix: { primary: 3 },
      geometry: [
        [17.44, 78.35],
        [17.45, 78.36],
      ],
    },
    {
      rank: 2,
      label: 'shortest_distance',
      distance_km: 2.1,
      free_flow_time_minutes: 3.9,
      congestion_aware_time_minutes: 5.4,
      mean_congestion_multiplier: 1.38,
      max_congestion_multiplier: 1.7,
      highway_mix: { secondary: 2 },
      geometry: [
        [17.44, 78.35],
        [17.445, 78.355],
        [17.45, 78.36],
      ],
    },
  ],
  graph: { nodes: 38, edges: 63 },
  attribution: 'Road data (c) OpenStreetMap contributors, ODbL.',
  limitations: ['Travel times use OpenStreetMap speed limits.'],
}

describe('RoutePlanner', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders the initial empty state and the map stub', () => {
    render(<RoutePlanner />)
    expect(screen.getByText(/no routes calculated yet/i)).toBeInTheDocument()
    expect(screen.getByTestId('route-map')).toBeInTheDocument()
    expect(screen.getByTestId('map-route-count')).toHaveTextContent('0')
  })

  it('requests routes and renders the comparison table', async () => {
    const user = userEvent.setup()
    planRoutes.mockResolvedValue(ROUTE_RESPONSE)

    render(<RoutePlanner />)
    await user.click(screen.getByRole('button', { name: /find routes/i }))

    await waitFor(() => expect(planRoutes).toHaveBeenCalledTimes(1))
    const payload = planRoutes.mock.calls[0][0]
    expect(payload.scenario).toBe('simulated_peak')
    expect(payload.origin.latitude).toBeCloseTo(17.4388, 3)

    expect(await screen.findByText('congestion aware fastest')).toBeInTheDocument()
    expect(screen.getByText('shortest distance')).toBeInTheDocument()
    expect(screen.getByText('2.345')).toBeInTheDocument()
    expect(screen.getByTestId('map-route-count')).toHaveTextContent('2')
    // Provenance must be visible so simulated data is not mistaken for real data.
    expect(screen.getByRole('note')).toHaveTextContent(/SIMULATED congestion scenario/i)
  })

  it('shows a readable error when the backend fails', async () => {
    const user = userEvent.setup()
    planRoutes.mockRejectedValue(new Error('Could not download the road network from OpenStreetMap'))

    render(<RoutePlanner />)
    await user.click(screen.getByRole('button', { name: /find routes/i }))

    expect(await screen.findByRole('alert')).toHaveTextContent(/could not download the road network/i)
    expect(screen.queryByText(/congestion aware fastest/i)).not.toBeInTheDocument()
  })

  it('validates identical origin and destination before calling the API', async () => {
    const user = userEvent.setup()
    render(<RoutePlanner />)

    const originLat = screen.getByLabelText(/origin latitude/i)
    const originLon = screen.getByLabelText(/origin longitude/i)
    const destLat = screen.getByLabelText(/destination latitude/i)
    const destLon = screen.getByLabelText(/destination longitude/i)

    await user.clear(originLat)
    await user.type(originLat, '17.44')
    await user.clear(originLon)
    await user.type(originLon, '78.35')
    await user.clear(destLat)
    await user.type(destLat, '17.44')
    await user.clear(destLon)
    await user.type(destLon, '78.35')

    await user.click(screen.getByRole('button', { name: /find routes/i }))

    expect(planRoutes).not.toHaveBeenCalled()
    expect(screen.getByRole('alert')).toHaveTextContent(/must be different points/i)
  })

  it('rejects an out-of-range latitude', async () => {
    const user = userEvent.setup()
    render(<RoutePlanner />)

    const originLat = screen.getByLabelText(/origin latitude/i)
    await user.clear(originLat)
    await user.type(originLat, '120')
    await user.click(screen.getByRole('button', { name: /find routes/i }))

    expect(planRoutes).not.toHaveBeenCalled()
    expect(screen.getByRole('alert')).toHaveTextContent(/between -90 and 90/i)
  })
})
