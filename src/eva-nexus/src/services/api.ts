/**
 * THE HIVE â€” API Service
 */

// â•â•â• HELPERS â•â•â•
function uuidv4() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
        const r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

export function createClientUuid() {
    const cryptoApi = globalThis.crypto
    if (cryptoApi && typeof cryptoApi.randomUUID === 'function') {
        return cryptoApi.randomUUID()
    }
    return uuidv4()
}



// â•â•â• AUTH HELPERS â•â•â•
const TOKEN_KEY = 'hive-auth-token'
const LOCAL_BANKER_URL = 'http://127.0.0.1:8100'
const CAN_BROWSER_USE_LOCAL_BANKER = ['127.0.0.1', 'localhost'].includes(window.location.hostname)
let bankerBaseUrlCache: { value: string; expiresAt: number } | null = null

function getAuthToken(): string | null {
    return localStorage.getItem(TOKEN_KEY)
}

function authHeaders(): Record<string, string> {
    const token = getAuthToken()
    return token ? { 'Authorization': `Bearer ${token}` } : {}
}

// â•â•â• TYPES â•â•â•
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

// â•â•â• API HELPERS â•â•â•
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

// â•â•â• HEALTH CHECKS â•â•â•
async function probeNodeHealth(name: string, url: string): Promise<NodeHealth> {
    const start = performance.now()
    try {
        const res = await fetchWithTimeout(url)
        const latency = Math.round(performance.now() - start)
        if (res.ok) {
            const details = await res.json().catch(() => ({}))
            const rawStatus = typeof (details as any)?.status === 'string'
                ? String((details as any).status).toLowerCase()
                : typeof (details as any)?.health === 'string'
                    ? String((details as any).health).toLowerCase()
                    : 'online'

            const status = rawStatus === 'offline'
                ? 'offline'
                : rawStatus === 'degraded' || rawStatus === 'unknown'
                    ? 'degraded'
                    : 'online'

            return { name, status, latency, details }
        }
        return { name, status: 'degraded', latency }
    } catch {
        return { name, status: 'offline', latency: -1 }
    }
}

async function probeLocalBankerHealth(): Promise<NodeHealth> {
    return probeNodeHealth('Banker', `${LOCAL_BANKER_URL}/health`)
}

async function resolveBankerBaseUrl(): Promise<string> {
    const now = Date.now()
    if (bankerBaseUrlCache && bankerBaseUrlCache.expiresAt > now) {
        return bankerBaseUrlCache.value
    }

    const serverHealth = await probeNodeHealth('Banker', '/api/banker/health')
    if (serverHealth.status === 'online') {
        bankerBaseUrlCache = { value: '/api/banker', expiresAt: now + 5000 }
        return '/api/banker'
    }

    if (!CAN_BROWSER_USE_LOCAL_BANKER) {
        bankerBaseUrlCache = { value: '/api/banker', expiresAt: now + 3000 }
        return '/api/banker'
    }

    const localHealth = await probeLocalBankerHealth()
    if (localHealth.status === 'online') {
        bankerBaseUrlCache = { value: LOCAL_BANKER_URL, expiresAt: now + 5000 }
        return LOCAL_BANKER_URL
    }

    bankerBaseUrlCache = { value: '/api/banker', expiresAt: now + 3000 }
    return '/api/banker'
}

async function safeBankerFetch<T>(path: string, fallback: T, timeout = 3000): Promise<T> {
    try {
        const baseUrl = await resolveBankerBaseUrl()
        return await safeFetch(`${baseUrl}${path}`, fallback, timeout)
    } catch {
        return fallback
    }
}

export async function checkNodeHealth(name: string, url: string): Promise<NodeHealth> {
    const primary = await probeNodeHealth(name, url)
    if (name !== 'Banker' || primary.status === 'online') {
        return primary
    }

    if (!CAN_BROWSER_USE_LOCAL_BANKER) {
        return primary
    }

    const local = await probeLocalBankerHealth()
    if (local.status === 'online') {
        return {
            ...local,
            name: 'Banker',
            details: {
                ...(local.details || {}),
                mode: 'hybrid-local',
                server_status: primary.status,
            },
        }
    }

    return primary
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

// â•â•â• KERNEL â•â•â•
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

// â•â•â• CORE â•â•â•
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
        return { message: 'CONNEXION PERDUE - Core indisponible. Verifie l etat systeme.' }
    }
}

export async function createSession(): Promise<{ session_id: string }> {
    const session_id = createClientUuid()
    try {
        const res = await fetchWithTimeout('/api/core/session', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id }),
        })
        if (!res.ok) {
            return { session_id }
        }
        return await res.json()
    } catch {
        return { session_id }
    }
}


// â•â•â• BANKER â•â•â•
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

export interface TradingConnectionStatus {
    mt5_connected: boolean
    mock_mode: boolean
}

export interface TradingAccountStatus {
    equity: number
    balance: number
    margin: number
    free_margin: number
    currency: string
    leverage: number
}

export interface TradingRiskStatus {
    daily_drawdown_percent: number
    trading_allowed: boolean
    open_positions: number
    anti_tilt_active: boolean
    news_filter_active: boolean
}

export interface TradingDecisionStatus {
    price: number
    action: string
    rsi: number
    vwap?: number
    adx?: number
    cortex_bias?: string
    gnn_bias?: string
    comment?: string
}

export interface TradingLiveUniverseStatus {
    horizon?: string
    symbols?: string[]
    count?: number
    source?: string
    restricted?: boolean
    selection?: string
    engine_label?: string
}

export interface TradingStatusResponse {
    status: string
    connection: TradingConnectionStatus
    account: TradingAccountStatus
    positions: TradingPosition[]
    risk: TradingRiskStatus
    decisions: Record<string, TradingDecisionStatus>
    universe: {
        dynamic: boolean
        symbols_total: number
        batch_size: number
        lab_live?: TradingLiveUniverseStatus | null
    }
}

export async function createOrder(order: OrderRequest): Promise<OrderResponse> {
    try {
        const baseUrl = await resolveBankerBaseUrl()
        const res = await fetchWithTimeout(`${baseUrl}/orders`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(order)
        })
        return await res.json()
    } catch {
        return { success: false, message: 'ORDRE ECHOUE - CONNEXION PERDUE' }
    }
}

export async function closePosition(ticket: string | number): Promise<any> {
    try {
        const baseUrl = await resolveBankerBaseUrl()
        const res = await fetchWithTimeout(`${baseUrl}/positions/${ticket}`, {
            method: 'DELETE'
        })
        return await res.json()
    } catch {
        return { success: false, message: 'FERMETURE ECHOUEE' }
    }
}

export async function getBankerTelemetry(): Promise<TelemetryData | null> {
    return safeBankerFetch('/telemetry', null)
}

export async function getBankerCircuitBreaker(): Promise<CircuitBreakerStatus | null> {
    return safeBankerFetch('/circuit-breaker/status', null)
}

export async function getNemesisStatus(): Promise<NemesisStatus> {
    return safeBankerFetch('/nemesis/status', {
        total_defeats: 0,
        known_nemeses: {},
        trading_blocked: false,
        blocked_until: null
    })
}

export async function getNewsFilter(): Promise<NewsFilterStatus> {
    return safeBankerFetch('/news/filter', {
        is_active: false,
        blocked_until: null,
        next_high_impact_events: []
    })
}

export async function getTradingStatus(): Promise<TradingStatusResponse> {
    return safeBankerFetch('/trading/status', {
        status: 'offline',
        connection: { mt5_connected: false, mock_mode: false },
        account: { equity: 0, balance: 0, margin: 0, free_margin: 0, currency: 'USD', leverage: 0 },
        positions: [],
        risk: {
            daily_drawdown_percent: 0,
            trading_allowed: false,
            open_positions: 0,
            anti_tilt_active: false,
            news_filter_active: false,
        },
        decisions: {},
        universe: {
            dynamic: false,
            symbols_total: 0,
            batch_size: 0,
            lab_live: null,
        },
    })
}

export interface ModelPerformanceRow {
    label: string
    closed_trades: number
    wins: number
    losses: number
    win_rate: number
    net_profit: number
    avg_profit: number
    gross_profit: number
    symbols: string[]
    last_closed_at: string | null
}

export interface ModelPerformanceSummary {
    closed_trades: number
    wins: number
    losses: number
    win_rate: number
    net_profit: number
    realized_pnl?: number
    window_label?: string
    from: string
    to: string
}

export interface ModelPerformanceReport {
    status: string
    window_days: number
    summary: ModelPerformanceSummary
    by_model: ModelPerformanceRow[]
    by_family: ModelPerformanceRow[]
    recent_trades: Array<{
        position_id: number
        symbol: string
        action: string
        label: string
        family: string
        entry_time: string | null
        close_time: string | null
        entry_price: number | null
        exit_price: number | null
        volume: number
        net_profit: number
        gross_profit: number
        swap: number
        commission: number
        magic: number
    }>
}

export async function getModelPerformance(days = 7, limit = 5): Promise<ModelPerformanceReport> {
    return safeBankerFetch(`/performance/models?days=${days}&limit=${limit}`, {
        status: 'offline',
        window_days: days,
        summary: {
            closed_trades: 0,
            wins: 0,
            losses: 0,
            win_rate: 0,
            net_profit: 0,
            realized_pnl: 0,
            window_label: `${days}j`,
            from: '',
            to: '',
        },
        by_model: [],
        by_family: [],
        recent_trades: [],
    }, 8000)
}

export async function toggleAutoTrading(enable: boolean): Promise<any> {
    try {
        const baseUrl = await resolveBankerBaseUrl()
        const res = await fetchWithTimeout(`${baseUrl}/trading/auto`, {
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
    return safeBankerFetch('/accounts/propfirm', [])
}

// â•â•â• LAB / CHAMPIONS â•â•â•
export interface ModelArtifactInfo {
    path: string | null
    exists: boolean
    size_bytes: number | null
    modified_at: string | null
}

export interface HorizonChampionStatus {
    horizon: string
    champion_id: string | null
    registry_champion_id?: string | null
    live_champion_id?: string | null
    candidate_id?: string | null
    selection_policy: string
    engine_label: string
    selection: string
    promotion_gate?: {
        allowed: boolean
        status: string
        reason: string
        checks: Record<string, boolean>
        thresholds: {
            require_positive_metrics: boolean
            min_win_rate: number
            min_return_pct: number
            min_profit_factor: number
            min_total_trades?: number
            min_eval_games?: number
            min_eval_symbols?: number
            min_expectancy_pct?: number
            max_drawdown_pct?: number
            min_positive_episode_rate?: number
        }
        metrics: {
            win_rate: number
            return_pct: number
            profit_factor: number
            total_trades?: number
            evaluation_games?: number
            evaluation_symbols?: number
            expectancy_pct?: number
            max_drawdown_pct?: number
            positive_episode_rate?: number
        }
    } | null
    live_universe?: {
        horizon: string
        symbols: string[]
        count: number
        source: string
        restricted: boolean
    } | null
    live_checkpoint: ModelArtifactInfo
    champion_checkpoint: ModelArtifactInfo
    latest_model: ModelArtifactInfo
    latest_checkpoint: ModelArtifactInfo
    manifest: Record<string, unknown> | null
    arena_report: Record<string, unknown> | null
}

export interface ChampionPerformanceSummary {
    champion: string
    win_rate: number
    return_pct: number
}

export interface LabChampionStatus {
    status: string
    selection_policy: string
    dreamer_gate: {
        enable_training: boolean
        training_active: boolean
        inference_count: number
        mode: string
        engine: string
        muzero_loaded: boolean
        live_selection_policy: string
        jax_agents: Record<string, { path?: string; selection?: string; policy?: string }>
        legacy_agent_loaded: boolean
    }
    champions: Record<string, string>
    registry_champions?: Record<string, string>
    live_champions?: Record<string, string | null>
    performance_summary: Record<string, ChampionPerformanceSummary>
    horizons: Record<string, HorizonChampionStatus>
    nightly_summary: Record<string, unknown> | null
}

export interface TrainingDependencyStatus {
    name: string
    ok: boolean
    state: string
    host?: string
    port?: number
    error?: string
    container?: string
    pid?: string | number
    updated_at?: string
}

export interface TrainingCurrentStep {
    name?: string
    status?: string
    phase?: string
    horizon?: string
    symbol?: string
    symbol_index?: number
    symbol_total?: number
    part_index?: number
    part_total?: number
    epoch_current?: number
    epoch_total?: number
    training_step_current?: number
    training_step_total?: number
    updated_at?: string
}

export interface TrainingRunPayload {
    run_id: string | null
    active: boolean
    status: string
    trigger: string | null
    strategy: string | null
    reason: string | null
    skip_reason?: string | null
    started_at?: string | null
    updated_at?: string | null
    finished_at?: string | null
    step_label?: string
    has_active_run?: boolean
    current_step?: TrainingCurrentStep | null
    completed_steps?: string[]
    failed_step?: Record<string, unknown> | null
    launcher?: Record<string, unknown>
}

export interface TrainingUniverseSummary {
    history_dir?: string
    total_symbols: number
    family_counts: Record<string, number>
    timeframe_counts: Record<string, number>
    family_samples: Record<string, string[]>
    sample_symbols: string[]
    horizon_universe: Record<string, { timeframe: string; count: number; sample_symbols: string[] }>
}

export interface TrainingRunStatus {
    status: string
    run: TrainingRunPayload
    dependencies: Record<string, TrainingDependencyStatus>
    universe: TrainingUniverseSummary
    logs: string[]
    nightly_summary: Record<string, unknown> | null
    status_path?: string
    log_path?: string
}

export async function getLabChampionStatus(): Promise<LabChampionStatus> {
    return safeFetch('/api/lab/champions/status', {
        status: 'offline',
        selection_policy: 'champion_only',
        dreamer_gate: {
            enable_training: false,
            training_active: false,
            inference_count: 0,
            mode: 'UNKNOWN',
            engine: 'RSI Heuristic',
            muzero_loaded: false,
            live_selection_policy: 'champion_only',
            jax_agents: {},
            legacy_agent_loaded: false,
        },
        champions: {},
        registry_champions: {},
        live_champions: {},
        performance_summary: {},
        horizons: {},
        nightly_summary: null,
    }, 8000)
}

export async function getLabTrainingStatus(): Promise<TrainingRunStatus> {
    return safeFetch('/api/lab/training/status', {
        status: 'offline',
        run: {
            run_id: null,
            active: false,
            status: 'idle',
            trigger: null,
            strategy: null,
            reason: null,
            skip_reason: null,
            started_at: null,
            updated_at: null,
            finished_at: null,
            current_step: null,
            completed_steps: [],
            failed_step: null,
            launcher: {},
        },
        dependencies: {},
        universe: {
            total_symbols: 0,
            family_counts: {},
            timeframe_counts: {},
            family_samples: {},
            sample_symbols: [],
            horizon_universe: {},
        },
        logs: [],
        nightly_summary: null,
        status_path: '',
        log_path: '',
    }, 8000)
}

// â•â•â• MONITORING â€” REAL DOCKER & SYSTEM DATA â•â•â•
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

export async function getGNNGraph() {
    return safeFetch(`/api/core/gnn/graph`, { nodes: [], links: [] })
}

// â•â•â• ACCOUNTANT â•â•â•
export interface AccountantReport {
    summary: {
        gross: number
        tax: number
        expenses: number
        net: number
    }
    expenses: any[]
    timestamp: string
    currency?: string
    expense_count?: number
    pnl_entries?: number
}

export async function getAccountantReport(): Promise<AccountantReport | null> {
    const fallback: AccountantReport = {
        summary: {
            gross: 0,
            tax: 0,
            expenses: 0,
            net: 0,
        },
        expenses: [],
        timestamp: new Date(0).toISOString(),
        currency: 'EUR',
        expense_count: 0,
        pnl_entries: 0,
    }

    const raw = await safeFetch<Record<string, unknown> | null>('/api/accountant/report', null)
    if (!raw) {
        return null
    }

    if (typeof raw.summary === 'object' && raw.summary !== null) {
        const summary = raw.summary as Record<string, unknown>
        return {
            summary: {
                gross: Number(summary.gross ?? 0),
                tax: Number(summary.tax ?? 0),
                expenses: Number(summary.expenses ?? 0),
                net: Number(summary.net ?? 0),
            },
            expenses: Array.isArray(raw.expenses) ? raw.expenses : [],
            timestamp: String(raw.timestamp ?? raw.generated_at ?? fallback.timestamp),
            currency: typeof raw.currency === 'string' ? raw.currency : fallback.currency,
            expense_count: Number(raw.expense_count ?? 0),
            pnl_entries: Number(raw.pnl_entries ?? 0),
        }
    }

    return {
        summary: {
            gross: Number(raw.gross_profit ?? 0),
            tax: Number(raw.total_taxes ?? 0),
            expenses: Number(raw.total_expenses ?? 0),
            net: Number(raw.net_roi ?? 0),
        },
        expenses: [],
        timestamp: String(raw.generated_at ?? fallback.timestamp),
        currency: typeof raw.currency === 'string' ? raw.currency : fallback.currency,
        expense_count: Number(raw.expense_count ?? 0),
        pnl_entries: Number(raw.pnl_entries ?? 0),
    }
}

// â•â•â• BUILDER â•â•â•
export interface BuilderHealth {
    status: string
    service: string
    active_pipelines: number
    builds_completed: number
    forge_runs: number
    public_api_entries: number
    mutation_enabled: boolean
    deploy_enabled: boolean
}

export interface BuilderPublicApiEntry {
    name: string
    url: string
    description: string
    auth: string
    https: boolean
    cors: string
    category: string
}

export interface BuilderHistoryEntry {
    action: string
    service: string
    status: string
    details: string
    timestamp: string
}

export interface BuilderBuildRequest {
    prompt: string
    filename: string
    language: string
    auto_validate: boolean
    use_public_api_catalog: boolean
    api_context_query?: string
    api_context_limit?: number
}

export interface BuilderBuildResponse {
    status: string
    message?: string
    filename?: string
    project_dir?: string
    files?: Record<string, string>
    validation?: {
        success?: boolean
        output?: string
        stderr?: string
        error?: string | null
        executed?: boolean
        reason?: string
    } | null
    git_status?: string
    api_suggestions?: BuilderPublicApiEntry[]
}

export interface BuilderDeployRequest {
    service: string
    target: 'local' | 'proxmox'
    force_rebuild: boolean
    dry_run: boolean
    compose_file?: string | null
}

export interface BuilderMutationRequest {
    change_summary: string
    dry_run: boolean
}

export interface BuilderPipelineRequest {
    build: BuilderBuildRequest
    deploy?: BuilderDeployRequest | null
    mutation?: BuilderMutationRequest | null
}

export interface BuilderPipelineResult {
    build: BuilderBuildResponse
    deploy: any
    mutation: any
}

export async function getBuilderHealth(): Promise<BuilderHealth> {
    return safeFetch('/api/builder/health', {
        status: 'offline',
        service: 'builder',
        active_pipelines: 0,
        builds_completed: 0,
        forge_runs: 0,
        public_api_entries: 0,
        mutation_enabled: false,
        deploy_enabled: false,
    })
}

export async function getBuilderHistory(): Promise<{ history: BuilderHistoryEntry[]; total: number }> {
    return safeFetch('/api/builder/build/history', { history: [], total: 0 })
}

export async function syncBuilderPublicApiCatalog(): Promise<any> {
    try {
        const res = await fetchWithTimeout('/api/builder/catalog/public-apis/sync', {
            method: 'POST',
        }, 15000)
        return await res.json()
    } catch {
        return { status: 'error', message: 'Synchronisation du catalogue impossible.' }
    }
}

export async function searchBuilderPublicApis(query: string, limit = 6, category?: string): Promise<{ results: BuilderPublicApiEntry[]; total: number }> {
    const params = new URLSearchParams()
    if (query.trim()) params.set('query', query.trim())
    params.set('limit', String(limit))
    if (category?.trim()) params.set('category', category.trim())
    return safeFetch(`/api/builder/catalog/public-apis/search?${params.toString()}`, { results: [], total: 0 }, 8000)
}

export async function buildBuilderProject(request: BuilderBuildRequest): Promise<BuilderBuildResponse> {
    try {
        const res = await fetchWithTimeout('/api/builder/factory/build', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(request),
        }, 60000)
        return await res.json()
    } catch {
        return { status: 'error', message: 'Generation Builder indisponible.' }
    }
}

export async function triggerBuilderDeploy(request: BuilderDeployRequest): Promise<any> {
    try {
        const res = await fetchWithTimeout('/api/builder/deploy', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(request),
        }, 30000)
        return await res.json()
    } catch {
        return { status: 'error', message: 'Deploiement Builder indisponible.' }
    }
}

export async function triggerBuilderMutation(request: BuilderMutationRequest): Promise<any> {
    try {
        const res = await fetchWithTimeout('/api/builder/mutation/trigger', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(request),
        }, 30000)
        return await res.json()
    } catch {
        return { status: 'error', message: 'Mutation Builder indisponible.' }
    }
}

export async function runBuilderPipeline(request: BuilderPipelineRequest): Promise<BuilderPipelineResult> {
    const build = await buildBuilderProject(request.build)

    if (build.status === 'error') {
        return {
            build,
            deploy: { status: 'skipped', reason: 'Build en erreur.' },
            mutation: { status: 'skipped', reason: 'Build en erreur.' },
        }
    }

    const deploy = request.deploy
        ? await triggerBuilderDeploy(request.deploy)
        : { status: 'skipped', reason: 'Aucun deploiement demande.' }

    const mutationSummary = request.mutation?.change_summary?.trim() || (
        `Pipeline Nexus Builder pour ${request.build.filename} (${build.status})`
    )
    const mutation = request.mutation
        ? await triggerBuilderMutation({
            ...request.mutation,
            change_summary: mutationSummary,
        })
        : { status: 'skipped', reason: 'Aucune mutation demandee.' }

    return { build, deploy, mutation }
}

