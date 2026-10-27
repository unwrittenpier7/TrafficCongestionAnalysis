
import {
  CircleMarker,
  MapContainer,
  Polyline,
  TileLayer,
  Tooltip,
  useMapEvents,
} from 'react-leaflet'
import 'leaflet/dist/leaflet.css'

// Distinct colours so alternative routes stay visually separable.
export const ROUTE_COLORS = [
  '#2563eb',
  '#d97706',
  '#0f766e',
  '#7c3aed',
  '#be123c',
  '#0369a1',
]

function ClickHandler({ onPick }) {
  useMapEvents({
    click(event) {
      onPick([event.latlng.lat, event.latlng.lng])
    },
  })
  return null
}

/**
 * Interactive OpenStreetMap view with route geometries and origin/destination
 * markers. Attribution and tile usage follow the OpenStreetMap tile policy.
 */
export default function RouteMap({ center, routes, origin, destination, onPick }) {
  return (
    <div className="map-wrapper">
      <MapContainer
        center={center}
        zoom={15}
        style={{ height: '420px', width: '100%' }}
        scrollWheelZoom
      >
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          maxZoom={19}
        />
        <ClickHandler onPick={onPick} />

        <CircleMarker
          center={origin}
          radius={7}
          pathOptions={{ color: '#166534', fillColor: '#22c55e', fillOpacity: 1 }}
        >
          <Tooltip permanent direction="top">
            Origin
          </Tooltip>
        </CircleMarker>
        <CircleMarker
          center={destination}
          radius={7}
          pathOptions={{ color: '#7f1d1d', fillColor: '#ef4444', fillOpacity: 1 }}
        >
          <Tooltip permanent direction="top">
            Destination
          </Tooltip>
        </CircleMarker>

        {routes.map((route, index) => (
          <Polyline
            key={`${route.label}-${index}`}
            positions={route.geometry}
            pathOptions={{
              color: ROUTE_COLORS[index % ROUTE_COLORS.length],
              weight: route.rank === 1 ? 6 : 4,
              opacity: route.rank === 1 ? 0.95 : 0.75,
              dashArray: route.rank === 1 ? undefined : '8 6',
            }}
          >
            <Tooltip sticky>
              <strong>
                #{route.rank} {route.label.replaceAll('_', ' ')}
              </strong>
              <br />
              {route.distance_km.toFixed(3)} km &middot;{' '}
              {route.congestion_aware_time_minutes.toFixed(2)} min (congestion-aware)
            </Tooltip>
          </Polyline>
        ))}
      </MapContainer>
    </div>
  )
}
