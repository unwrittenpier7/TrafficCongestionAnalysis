/**
 * Thin API client for the TrafficSense FastAPI backend.
 *
 * All calls surface a readable error message. Nothing here fabricates a
 * response: a failed request rejects with an Error the UI must handle.
 */

// In development the Vite proxy forwards /api to the backend, so the default
// base URL is empty. In production set VITE_API_BASE_URL at build time.
const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')

const JSON_HEADERS = { 'Content-Type': 'application/json' }

async function request(path, options = {}) {
  const url = `${API_BASE_URL}${path}`
  let response
  try {
    response = await fetch(url, options)
  } catch (cause) {
    throw new Error(
      `Could not reach the TrafficSense backend at ${url}. ` +
        'Make sure the API is running and VITE_API_BASE_URL is correct.',
      { cause },
    )
  }

  const text = await response.text()
  let payload = null
  if (text) {
    try {
      payload = JSON.parse(text)
    } catch {
      payload = { detail: text }
    }
  }

  if (!response.ok) {
    const detail =
      payload && typeof payload.detail === 'string'
        ? payload.detail
        : payload && Array.isArray(payload.detail)
          ? payload.detail.map((item) => item.msg || JSON.stringify(item)).join('; ')
          : `Request failed with status ${response.status}`
    throw new Error(detail)
  }

  return payload
}

export function fetchHealth(signal) {
  return request('/api/health', { signal })
}

export function fetchMetrics(signal) {
  return request('/api/metrics', { signal })
}

export function fetchServiceInfo(signal) {
  return request('/', { signal })
}

export function predictTraffic(body, signal) {
  return request('/api/predict', {
    method: 'POST',
    headers: JSON_HEADERS,
    body: JSON.stringify(body),
    signal,
  })
}

export function planRoutes(body, signal) {
  return request('/api/routes', {
    method: 'POST',
    headers: JSON_HEADERS,
    body: JSON.stringify(body),
    signal,
  })
}

export const CONGESTION_CATEGORIES = ['Low', 'Moderate', 'High', 'Severe']

export const WEATHER_CATEGORIES = [
  'clear',
  'clouds',
  'drizzle',
  'fog',
  'haze',
  'mist',
  'rain',
  'smoke',
  'snow',
  'squall',
  'thunderstorm',
]

export const DAY_NAMES = [
  'Monday',
  'Tuesday',
  'Wednesday',
  'Thursday',
  'Friday',
  'Saturday',
  'Sunday',
]

export const MONTH_NAMES = [
  'January',
  'February',
  'March',
  'April',
  'May',
  'June',
  'July',
  'August',
  'September',
  'October',
  'November',
  'December',
]

export { API_BASE_URL }
