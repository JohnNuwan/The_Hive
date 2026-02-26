/**
 * THE HIVE — API Service
 */

// ═══ HELPERS ═══
function uuidv4() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
        const r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}



// ═══ AUTH HELPERS ═══
const TOKEN_KEY = 'hive-auth-token'

function getAuthToken(): string | null {
    return localStorage.getItem(TOKEN_KEY)
}

function authHeaders(): Record<string, string> {
    const token = getAuthToken()
    return token ? { 'Authorization': `Bearer ${token}` } : {}
}

// ═══ TYPES ═══
export interface NodeHealth {
    name: string
    status: 'online' | 'offline' | 'degraded'
    latency: number
    details?: Record<string, unknown>
}

export interface KillSwitchStatus {
    is_active: boolean
    message: string
}

export interface TelemetryData {
    service_name: string
    uptime_seconds: number
    requests_total: number
    errors_total: number
    active_role?: string
    active_model?: string
    timestamp: string
}

export interface CircuitBreakerStatus {
    name: string
    state: 'CLOSED' | 'OPEN' | 'HALF_OPEN'
    failures: number
    failure_threshold: number
}

export interface NemesisStatus {
    total_defeats: number
    known_nemeses: Record<string, number>
    trading_blocked: boolean
    blocked_until: string | null
}

export interface NewsFilterStatus {
    is_active: boolean
    blocked_until: string | null
    next_high_impact_events: Array<{ event: string; impact: string; time: string }>
}

export interface TradingPosition {
    ticket: string
    symbol: string
    action: string
    volume: number
    profit: number
    open_price: number
    current_price: number
}

export interface AccountInfo {
    equity: number
    balance: number
    margin: number
    currency: string
}

export interface ChatMessage {
    message: string
    session_id?: string
    thoughts?: string | null  // Reasoning trace from the expert
    metadata?: Record<string, unknown>
}

export interface SystemMetrics {
    cpu: { usage: number; cores: number; model: string; temp: number; freq?: number }
    memory: { used: number; total: number; percent: number }
    gpu: { name: string; usage: number; memory_used: number; memory_total: number; temp: number } | null
    disk: { used: number; total: number; percent: number; read_speed: number; write_speed: number }
    network: { rx_bytes: number; tx_bytes: number; rx_speed: number; tx_speed: number }
    uptime: number
    real_data?: boolean
    hostname?: string
    platform?: string
}

export interface ContainerStats {
    id: string
    name: string
    status: 'running' | 'stopped' | 'restarting' | 'paused'
    cpu_percent: number
    memory_usage: number
    memory_limit: number
    memory_percent: number
    network_rx: number
    network_tx: number
    pids: number
    image: string
    uptime: string
}

// ═══ API HELPERS ═══
async function fetchWithTimeout(url: string, options: RequestInit = {}, timeout = 3000): Promise<Response> {
    const controller = new AbortController()
    const id = setTimeout(() => controller.abort(), timeout)
    try {
        // Merge auth headers
        const headers = {
            ...authHeaders(),
            ...(options.headers || {}),
        }
        const response = await fetch(url, { ...options, headers, signal: controller.signal })
        clearTimeout(id)
        return response
    } catch (error) {
        clearTimeout(id)
        throw error
    }
}

export async function safeFetch<T>(url: string, fallback: T, timeout = 3000): Promise<T> {
    try {
        const res = await fetchWithTimeout(url, {}, timeout)
        if (!res.ok) return fallback
        return await res.json() as T
    } catch {
        return fallback
    }
}

// ═══ HEALTH CHECKS ═══
export async function checkNodeHealth(name: string, url: string): Promise<NodeHealth> {
    const start = performance.now()
    try {
        const res = await fetchWithTimeout(url)
        const latency = Math.round(performance.now() - start)
        if (res.ok) {
            const details = await res.json().catch(() => ({}))
            return { name, status: 'online', latency, details }
        }
        return { name, status: 'degraded', latency }
    } catch {
        return { name, status: 'offline', latency: -1 }
    }
}

export async function getAllNodesHealth(): Promise<NodeHealth[]> {
    const nodes = [
        { name: 'EVA Core', url: '/api/core/health' },
        { name: 'Banker', url: '/api/banker/health' },
        { name: 'Sentinel', url: '/api/sentinel/health' },
        { name: 'Shadow', url: '/api/shadow/health' },
    ]
    return Promise.all(nodes.map(n => checkNodeHealth(n.name, n.url)))
}

export async function getStatus() {
    return safeFetch('/api/core/system/status', {
        core: { status: 'online' },
        banker: { status: 'online' },
        sentinel: { status: 'online' }
    })
}

// ═══ KERNEL ═══
export async function getKillSwitchStatus(): Promise<KillSwitchStatus> {
    return safeFetch('/api/sentinel/health', { is_active: false, message: 'OFFLINE' })
        .then(data => ({
            is_active: (data as any)?.kill_switch_active ?? false,
            message: (data as any)?.message ?? 'OFFLINE'
        }))
}

export async function toggleKillSwitch(action: 'activate' | 'reset'): Promise<KillSwitchStatus> {
    try {
        const res = await fetchWithTimeout('/api/sentinel/kill-switch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action })
        })
        const data = await res.json()
        return { is_active: data.kill_switch_active, message: data.message }
    } catch {
        return { is_active: false, message: 'CONNECTION FAILED' }
    }
}

// ═══ CORE ═══
export async function getCoreTelemetry(): Promise<TelemetryData | null> {
    return safeFetch('/api/core/telemetry', null)
}

export async function getCoreCircuitBreaker(): Promise<CircuitBreakerStatus | null> {
    return safeFetch('/api/core/circuit-breaker/status', null)
}

export async function sendChatMessage(message: string, sessionId: string, image?: string): Promise<ChatMessage> {
    try {
        const body: Record<string, unknown> = { message, session_id: sessionId }
        if (image) body.image = image
        const res = await fetchWithTimeout('/api/core/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        }, 30000)
        if (!res.ok) throw new Error('Chat failed')
        return await res.json()
    } catch {
        return { message: '⚠ CONNECTION LOST — Core unreachable. Check system status.' }
    }
}

export async function createSession(): Promise<{ session_id: string }> {
    const session_id = (typeof window !== 'undefined' && window.crypto && (window.crypto as any).randomUUID)
        ? (window.crypto as any).randomUUID()
        : uuidv4()
    return safeFetch('/api/core/session', { session_id })
}


// ═══ BANKER ═══
export interface OrderRequest {
    symbol: string
    action: 'BUY' | 'SELL'
    volume: number
    stop_loss?: number
    take_profit?: number
}


export interface OrderResponse {
    success: boolean
    ticket?: number
    message: string
}

export async function createOrder(order: OrderRequest): Promise<OrderResponse> {
    try {
        const res = await fetchWithTimeout('/api/banker/orders', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(order)
        })
        return await res.json()
    } catch {
        return { success: false, message: 'ORDER FAILED — CONNECTION LOST' }
    }
}

export async function closePosition(ticket: string | number): Promise<any> {
    try {
        const res = await fetchWithTimeout(`/api/banker/positions/${ticket}`, {
            method: 'DELETE'
        })
        return await res.json()
    } catch {
        return { success: false, message: 'CLOSE FAILED' }
    }
}

export async function getBankerTelemetry(): Promise<TelemetryData | null> {
    return safeFetch('/api/banker/telemetry', null)
}

export async function getBankerCircuitBreaker(): Promise<CircuitBreakerStatus | null> {
    return safeFetch('/api/banker/circuit-breaker/status', null)
}

export async function getNemesisStatus(): Promise<NemesisStatus> {
    return safeFetch('/api/banker/nemesis/status', {
        total_defeats: 0,
        known_nemeses: {},
        trading_blocked: false,
        blocked_until: null
    })
}

export async function getNewsFilter(): Promise<NewsFilterStatus> {
    return safeFetch('/api/banker/news/filter', {
        is_active: false,
        blocked_until: null,
        next_high_impact_events: []
    })
}

export async function getTradingStatus(): Promise<{ account: AccountInfo; positions: TradingPosition[]; risk: any }> {
    return safeFetch('/api/banker/trading/status', {
        account: { equity: 0, balance: 0, margin: 0, currency: 'USD' },
        positions: [],
        risk: { daily_drawdown_percent: 0, trading_allowed: true }
    })
}

export async function toggleAutoTrading(enable: boolean): Promise<any> {
    try {
        const res = await fetchWithTimeout('/api/banker/trading/auto', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ enable })
        })
        return await res.json()
    } catch {
        return { status: 'ERROR', active: false }
    }
}

export async function getPropFirmAccounts(): Promise<any[]> {
    return safeFetch('/api/banker/accounts/propfirm', [])
}

// ═══ MONITORING — REAL DOCKER & SYSTEM DATA ═══
export async function getSystemMetrics(): Promise<SystemMetrics | null> {
    return safeFetch('/api/core/system/metrics', null, 5000)
}

export async function getDockerContainers(): Promise<ContainerStats[]> {
    return safeFetch('/api/core/docker/containers', [], 8000)
}

export async function getSystemStatus() {
    const response = await fetch('/api/core/system/status')
    if (!response.ok) throw new Error('Sentinel unreachable')
    return response.json()
}

export async function getMemoryGraph(limit = 50, similarityThreshold = 0.8) {
    return safeFetch(`/api/core/memory/graph?limit=${limit}&similarity_threshold=${similarityThreshold}`, { nodes: [], links: [] })
}

// ═══ ACCOUNTANT ═══
export interface AccountantReport {
    summary: {
        gross: number
        tax: number
        expenses: number
        net: number
    }
    expenses: any[]
    timestamp: string
}

export async function getAccountantReport(): Promise<AccountantReport | null> {
    return safeFetch('/api/accountant/report', null)
}
