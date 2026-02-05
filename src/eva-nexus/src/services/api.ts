/**
 * Service API - Communication avec EVA Core
 */

const API_BASE_URL = '/api'

export async function getStatus() {
    const response = await fetch(`${API_BASE_URL}/agents/status`)
    if (!response.ok) throw new Error('API unreachable')
    return response.json()
}

export async function createSession() {
    const response = await fetch(`${API_BASE_URL}/session`, { method: 'POST' })
    if (!response.ok) throw new Error('Session creation failed')
    return response.json()
}

export async function sendChatMessage(message: string, sessionId: string) {
    const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, session_id: sessionId })
    })
    if (!response.ok) throw new Error('Chat failed')
    return response.json()
}

export async function getTradingStatus() {
    // Direct call to Banker if proxied via Core or separate port
    const response = await fetch(`${API_BASE_URL}/trading/status`)
    if (!response.ok) throw new Error('Banker unreachable')
    return response.json()
}
