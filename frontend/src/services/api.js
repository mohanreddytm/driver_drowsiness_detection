const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000'

export async function startDetection() {
  await fetch(`${BACKEND_URL}/api/start`, { method: 'POST' })
}

export async function stopDetection() {
  await fetch(`${BACKEND_URL}/api/stop`, { method: 'POST' })
}

export function getStreamUrl() {
  return `${BACKEND_URL}/api/stream`
}

export function connectStatusSocket(onMessage) {
  const wsUrl = (BACKEND_URL.replace('http', 'ws')) + '/ws/status'
  const ws = new WebSocket(wsUrl)
  ws.onopen = () => {
    // heartbeat
    const interval = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) ws.send('ping')
    }, 500)
    ws._hb = interval
  }
  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data)
      onMessage?.(data)
    } catch {}
  }
  ws.onclose = () => {
    if (ws._hb) clearInterval(ws._hb)
  }
  return ws
}

export async function setAlertMessage(message) {
  const response = await fetch(`${BACKEND_URL}/api/set_alert_message`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message })
  })
  return response.json()
}

export async function getAlertMessage() {
  const response = await fetch(`${BACKEND_URL}/api/get_alert_message`)
  return response.json()
}
