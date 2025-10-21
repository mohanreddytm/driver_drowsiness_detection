import React from 'react'

export default function StatusBadge({ status='Idle' }) {
  const color = status === 'Drowsy' ? '#e53935' : status === 'Warning' ? '#fdd835' : '#43a047'
  return (
    <span style={{
      padding: '6px 12px',
      backgroundColor: color,
      color: '#000',
      borderRadius: 6,
      fontWeight: 700
    }}>{status}</span>
  )
}
