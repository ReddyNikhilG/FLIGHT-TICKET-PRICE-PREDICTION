const DEFAULT_API_BASE_URL = 'http://127.0.0.1:8001';

const normalizeBaseUrl = (value) => (value ? String(value).replace(/\/$/, '') : DEFAULT_API_BASE_URL);

export const FLIGHT_API_BASE_URL = normalizeBaseUrl(
  import.meta.env.VITE_FLIGHT_API_BASE_URL || import.meta.env.VITE_API_BASE_URL,
);

async function requestJson(path, options = {}) {
  const response = await fetch(`${FLIGHT_API_BASE_URL}${path}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {}),
    },
  });

  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(data.error || `Request failed with status ${response.status}`);
  }

  return data;
}

export function fetchFlightMetadata() {
  return requestJson('/metadata');
}

export function fetchModelMetrics() {
  return requestJson('/metrics');
}

export function predictFlightPrice(payload) {
  return requestJson('/predict', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export function fetchPredictionHistory() {
  return requestJson('/history');
}