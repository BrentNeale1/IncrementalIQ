const BASE = '';

async function handleResponse(res) {
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    const msg = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail);
    throw new Error(msg);
  }
  return res.json();
}

export function apiGet(path) {
  return fetch(`${BASE}${path}`).then(handleResponse);
}

export function apiPost(path, data) {
  return fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  }).then(handleResponse);
}

export function apiPostFile(path, file, extraFields = {}) {
  const form = new FormData();
  form.append('file', file);
  for (const [k, v] of Object.entries(extraFields)) {
    form.append(k, typeof v === 'string' ? v : JSON.stringify(v));
  }
  return fetch(`${BASE}${path}`, { method: 'POST', body: form }).then(handleResponse);
}

export function apiDelete(path) {
  return fetch(`${BASE}${path}`, { method: 'DELETE' }).then(handleResponse);
}
