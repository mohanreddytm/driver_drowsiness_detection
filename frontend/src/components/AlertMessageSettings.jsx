import React, { useState, useEffect } from 'react'
import { setAlertMessage, getAlertMessage } from '../services/api'

export default function AlertMessageSettings() {
  const [message, setMessage] = useState('')
  const [savedMessage, setSavedMessage] = useState('')
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    // Load current message on mount
    getAlertMessage().then(data => {
      setSavedMessage(data.message || '')
    })
  }, [])

  const handleSave = async () => {
    setLoading(true)
    try {
      await setAlertMessage(message)
      setSavedMessage(message)
      setMessage('')
    } catch (error) {
      console.error('Failed to save alert message:', error)
    }
    setLoading(false)
  }

  const handleClear = async () => {
    setLoading(true)
    try {
      await setAlertMessage('')
      setSavedMessage('')
      setMessage('')
    } catch (error) {
      console.error('Failed to clear alert message:', error)
    }
    setLoading(false)
  }

  return (
    <div style={{ padding: 16, border: '1px solid #e0e0e0', borderRadius: 8, marginBottom: 16 }}>
      <h3 style={{ margin: '0 0 12px 0', fontSize: 16 }}>Custom Alert Message</h3>
      
      <div style={{ marginBottom: 12 }}>
        <input
          type="text"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="Enter custom alert message (e.g., 'Hey Ravi, wake up!')"
          style={{
            width: '90%',
            padding: '8px 12px',
            border: '1px solid #ccc',
            borderRadius: 4,
            fontSize: 14
          }}
        />
      </div>
      
      <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
        <button
          onClick={handleSave}
          disabled={loading || !message.trim()}
          style={{
            padding: '8px 16px',
            backgroundColor: '#1976d2',
            color: 'white',
            border: 'none',
            borderRadius: 4,
            cursor: loading || !message.trim() ? 'not-allowed' : 'pointer',
            opacity: loading || !message.trim() ? 0.6 : 1
          }}
        >
          {loading ? 'Saving...' : 'Save Alert Message'}
        </button>
        
        <button
          onClick={handleClear}
          disabled={loading}
          style={{
            padding: '8px 16px',
            backgroundColor: '#fff',
            color: 'red',
            border: '1px solid red',
            borderRadius: 4,
            cursor: loading ? 'not-allowed' : 'pointer',
            opacity: loading ? 0.6 : 1
          }}
        >
          Clear Message
        </button>
      </div>
      
      {savedMessage && (
        <div style={{
          padding: 8,
          backgroundColor: '#e8f5e8',
          border: '1px solid #4caf50',
          borderRadius: 4,
          fontSize: 14
        }}>
          <strong>Current message:</strong> "{savedMessage}"
        </div>
      )}
      
      {!savedMessage && (
        <div style={{
          padding: 8,
          backgroundColor: '#fff3e0',
          border: '1px solid #ff9800',
          borderRadius: 4,
          fontSize: 14
        }}>
          <strong>Default:</strong> Using default alarm sound
        </div>
      )}
    </div>
  )
}
