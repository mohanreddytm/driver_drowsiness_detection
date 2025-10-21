import React, { useEffect, useRef, useState } from 'react'
import { startDetection, stopDetection, getStreamUrl, connectStatusSocket } from '../services/api'
import StatusBadge from '../components/StatusBadge'
import AlertMessageSettings from '../components/AlertMessageSettings'

export default function App() {
  const [running, setRunning] = useState(false)
  const [status, setStatus] = useState('Idle')
  const [elapsed, setElapsed] = useState(0)
  const [target, setTarget] = useState(7)
  const imgRef = useRef(null)
  const wsRef = useRef(null)

  useEffect(() => {
    return () => {
      if (wsRef.current) wsRef.current.close()
    }
  }, [])

  const onStart = async () => {
    await startDetection()
    setRunning(true)
    // open stream
    if (imgRef.current) imgRef.current.src = getStreamUrl() + `?t=${Date.now()}`
    // connect ws
    wsRef.current = connectStatusSocket((msg) => {
      setStatus(msg.status)
      setElapsed(msg.elapsed)
      setTarget(msg.target)
    })
  }

  const onStop = async () => {
    await stopDetection()
    setRunning(false)
    if (wsRef.current) wsRef.current.close()
    if (imgRef.current) imgRef.current.src = ''
    setStatus('Idle')
    setElapsed(0)
  }

  const borderColor = status === 'Drowsy' ? '#e53935' : status === 'Warning' ? '#fdd835' : '#e0e0e0'

  return (
    <div style={{ fontFamily: 'system-ui, Segoe UI, Roboto, Arial', padding: 24, maxWidth: 980, margin: '0 auto' }}>
      <header style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
        <h2 style={{ margin: 0 }}>Driver Drowsiness Detection</h2>
        <StatusBadge status={status} />
      </header>

      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
        <div style={{ flex: '1 1 640px', minWidth: 320 }}>
          <div style={{
            border: `4px solid ${borderColor}`,
            borderRadius: 8,
            overflow: 'hidden',
            background: '#000',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            minHeight: 360
          }}>
            {/* MJPEG stream */}
            <img ref={imgRef} alt="Drowsiness Stream" style={{ width: '100%', display: running ? 'block' : 'none' }} />
            {!running && <div style={{ color: '#888', padding: 24 }}>Stream not started.</div>}
          </div>
        </div>
        <div style={{ width: 320 }}>
          {/* Custom Alert Message Settings */}
          <AlertMessageSettings />
          
          {/* Detection Controls */}
          <div style={{ padding: 16, border: '1px solid #e0e0e0', borderRadius: 8 }}>
            <div style={{ marginBottom: 12 }}>
              <button onClick={onStart} disabled={running} style={{ padding: '10px 14px', marginRight: 8 }}>Start Detection</button>
              <button onClick={onStop} disabled={!running} style={{ padding: '10px 14px' }}>Stop Detection</button>
            </div>
            <div style={{ marginTop: 8, color: '#444' }}>
              <div>Elapsed: {elapsed.toFixed(1)}s / {target?.toFixed?.(1) || 0}s</div>
              <div>Tip: Red border and banner appear if drowsy.</div>
            </div>
          </div>
        </div>
      </div>

      <footer style={{ marginTop: 24, color: '#888' }}>
        Backend: <code>/api/start</code>, <code>/api/stop</code>, <code>/api/stream</code>, <code>/ws/status</code>
      </footer>
    </div>
  )
}
