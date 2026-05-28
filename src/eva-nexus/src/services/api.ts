/**
 * Service API principal de THE HIVE.
 */

// Aides locales.
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



// Aides d'authentification.
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

// Types publics.
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
    thoughts?: string | null  // Trace de raisonnement retournee par l'expert
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

// Aides API.
async function fetchWithTimeout(url: string, options: RequestInit = {}, timeout = 3000): Promise<Response> {
    const controller = new AbortController()
    const id = setTimeout(() => controller.abort(), timeout)
    try {
        // Fusionne les en-tetes d'authentification avec ceux de l'appel.
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

// Verification de sante.
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

// Services noyau.
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

// Services EVA Core.
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


// Services banker.
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
    live_champion_id_muzero?: string | null
    live_champion_id_dreamer?: string | null
    top_live_symbols_by_engine?: Record<string, string[]>
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
    runtime_mode?: string
    connectors?: Record<string, unknown>
    execution_mechanics?: Record<string, unknown>
    decision_audit?: Record<string, unknown>
    ensemble_decision_stats?: Record<string, number>
    live_family?: string | null
    live_champion_id_muzero?: string | null
    live_champion_id_dreamer?: string | null
    degraded_fallback_reason?: string | null
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
        runtime_mode: 'offline',
        connectors: {},
        execution_mechanics: {},
        decision_audit: {},
        ensemble_decision_stats: {},
        live_family: null,
        live_champion_id_muzero: null,
        live_champion_id_dreamer: null,
        degraded_fallback_reason: null,
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

// Services lab et champions.
export interface ModelArtifactInfo {
    path: string | null
    exists: boolean
    size_bytes: number | null
    modified_at: string | null
}

export interface HorizonChampionStatus {
    horizon: string
    engine?: string
    family?: string
    feature_profile?: string | null
    dataset_id?: string | null
    dataset_source?: string | null
    mechanics_profile_version?: string | null
    dataset_coverage?: Record<string, unknown>
    failure_mode?: string | null
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
            directional_bias?: string
            metrics_by_position_mechanics?: Record<string, number | string | null | undefined>
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
    metrics_by_position_mechanics?: Record<string, number | string | null | undefined>
    top_live_symbols?: string[]
    metrics_by_symbol?: Record<string, Record<string, unknown>>
}

export interface ChampionPerformanceSummary {
    champion: string
    win_rate: number
    return_pct: number
}

export interface EngineChampionMatrix {
    [engine: string]: Record<string, HorizonChampionStatus>
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
    live_champions_by_engine?: Record<string, Record<string, string | null>>
    performance_summary: Record<string, ChampionPerformanceSummary>
    horizons: Record<string, HorizonChampionStatus>
    engines?: EngineChampionMatrix
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

export interface TrainingArenaMetrics {
    score?: number
    family?: string
    feature_profile?: string | null
    dataset_id?: string | null
    dataset_source?: string | null
    mechanics_profile_version?: string | null
    dataset_coverage?: Record<string, unknown>
    evaluation_games?: number
    evaluation_symbols?: number
    return_pct?: number
    net_realized_pct?: number
    profit_factor?: number
    win_rate?: number
    total_trades?: number
    max_drawdown_pct?: number
    expectancy_pct?: number
    positive_episode_rate?: number
    long_entry_share?: number
    short_entry_share?: number
    directional_imbalance?: number
    directional_bias?: string
    ema200_blocked_buy?: number
    ema200_blocked_sell?: number
    metrics_by_position_mechanics?: Record<string, number | string | null | undefined>
}

export interface TrainingArenaSymbolProgress {
    symbol: string
    order?: number
    challenger?: TrainingArenaMetrics
    champion?: TrainingArenaMetrics
}

export interface TrainingArenaSideProgress {
    id?: string | null
    path?: string | null
    score?: number
    metrics?: TrainingArenaMetrics
}

export interface TrainingArenaProgress {
    status: string
    horizon?: string
    family?: string
    feature_profile?: string | null
    dataset_id?: string | null
    dataset_source?: string | null
    mechanics_profile_version?: string | null
    dataset_coverage?: Record<string, unknown>
    timeframe?: string
    started_at?: string | null
    updated_at?: string | null
    eval_symbols?: string[]
    symbol_total?: number
    current_role?: string | null
    current_symbol?: string | null
    symbol_index?: number
    challenger?: TrainingArenaSideProgress
    champion?: TrainingArenaSideProgress
    symbols?: Record<string, TrainingArenaSymbolProgress>
    outcome?: string
    action_required?: string
    validation?: Record<string, unknown>
}

export interface TrainingCurrentStep {
    name?: string
    status?: string
    phase?: string
    horizon?: string
    family?: string
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
    engine?: string | null
    trigger: string | null
    strategy: string | null
    reason: string | null
    family?: string | null
    feature_profile?: string | null
    dataset_id?: string | null
    dataset_source?: string | null
    mechanics_profile_version?: string | null
    ga_status?: string | null
    ga_generation?: number | null
    ga_trial?: string | null
    trial_mode?: string | null
    trial_cost_profile?: string | null
    replay_cache_status?: string | null
    replay_cache_key?: string | null
    replay_cache_entries?: number | null
    replay_cache_source?: string | null
    shadow_buffer_size?: number | null
    sequence_length?: number | null
    sequence_stride?: number | null
    world_model_steps?: number | null
    dataset_coverage?: Record<string, unknown>
    metrics_by_position_mechanics?: Record<string, number | string | null | undefined>
    skip_reason?: string | null
    started_at?: string | null
    updated_at?: string | null
    finished_at?: string | null
    step_label?: string
    has_active_run?: boolean
    current_step?: TrainingCurrentStep | null
    arena_progress?: TrainingArenaProgress | null
    reported_step?: TrainingCurrentStep | null
    observed_step?: TrainingCurrentStep | null
    effective_step?: TrainingCurrentStep | null
    reported_step_label?: string
    observed_step_label?: string
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
    engine?: string | null
    family?: string | null
    feature_profile?: string | null
    dataset_id?: string | null
    dataset_source?: string | null
    mechanics_profile_version?: string | null
    ga_status?: string | null
    ga_generation?: number | null
    ga_trial?: string | null
    trial_mode?: string | null
    trial_cost_profile?: string | null
    replay_cache_status?: string | null
    replay_cache_key?: string | null
    replay_cache_entries?: number | null
    replay_cache_source?: string | null
    shadow_buffer_size?: number | null
    sequence_length?: number | null
    sequence_stride?: number | null
    world_model_steps?: number | null
    dataset_coverage?: Record<string, unknown>
    metrics_by_position_mechanics?: Record<string, number | string | null | undefined>
    run: TrainingRunPayload
    dependencies: Record<string, TrainingDependencyStatus>
    universe: TrainingUniverseSummary
    logs: string[]
    nightly_summary: Record<string, unknown> | null
    status_path?: string
    log_path?: string
}

export interface MarketGnnArtifactInfo {
    path: string | null
    exists: boolean
    size_bytes: number | null
    modified_at: string | null
}

export interface MarketGnnRegistry {
    name: string
    version: string | null
    status: 'draft' | 'validated' | 'live' | 'blocked' | 'stale' | 'unavailable' | string
    status_reason?: string | null
    trained_at: string | null
    checkpoint_path: string | null
    source_run_id?: string | null
    last_refresh_requested_at?: string | null
    last_refresh_started_at?: string | null
    last_refresh_finished_at?: string | null
    last_refresh_status?: string | null
    coverage_summary?: Record<string, unknown>
    refresh_state?: Record<string, unknown>
    timeframes: string[]
    universe: {
        symbols: string[]
        count: number
        family_counts: Record<string, number>
    }
    metrics: {
        loss: number
        scalp_accuracy: number
        intraday_accuracy: number
        swing_accuracy: number
        epochs: number
        batch_size: number
        samples: number
    }
    artifacts: {
        registry: MarketGnnArtifactInfo
        checkpoint: MarketGnnArtifactInfo
        metrics: MarketGnnArtifactInfo
    }
}

export interface MarketGnnStatusResponse {
    status: string
    gnn: MarketGnnRegistry
    graph_readiness?: {
        status?: string
        reason?: string
        selected_timeframe?: string | null
        candidate_timeframes?: string[]
        overlap_points?: number
        missing_symbols?: string[]
    }
    refresh?: MarketGnnRefreshState
}

export interface MarketGnnMetricsResponse {
    status: string
    version: string | null
    model_status: string
    status_reason?: string | null
    trained_at: string | null
    source_run_id?: string | null
    coverage_summary?: Record<string, unknown>
    last_refresh_requested_at?: string | null
    last_refresh_started_at?: string | null
    last_refresh_finished_at?: string | null
    last_refresh_status?: string | null
    metrics: MarketGnnRegistry['metrics']
    universe: MarketGnnRegistry['universe']
    timeframes: string[]
    artifacts: MarketGnnRegistry['artifacts']
}

export interface MarketGnnRefreshState {
    status: string
    queued: boolean
    requested_at?: string | null
    started_at?: string | null
    finished_at?: string | null
    run_id?: string | null
    failure_reason?: string | null
    source_run_id?: string | null
    requested_by?: string | null
}

export interface MarketGnnGraphNode {
    id: string
    label: string
    role: 'core' | 'asset' | string
    family?: string
    centrality?: number
    timestamp?: string | null
}

export interface MarketGnnGraphLink {
    source: string
    target: string
    value: number
    correlation?: number
    kind?: string
}

export interface MarketGnnGraphSnapshot {
    status: 'ok' | 'unavailable' | string
    reason: string
    version?: string | null
    model_status?: string
    status_reason?: string | null
    trained_at?: string | null
    timeframes?: string[]
    universe?: MarketGnnRegistry['universe']
    metrics?: MarketGnnRegistry['metrics']
    graph_timeframe?: string | null
    selected_timeframe?: string | null
    candidate_timeframes?: string[]
    overlap_points?: number
    missing_symbols?: string[]
    coverage_summary?: Record<string, unknown>
    displayed_symbol_count?: number
    universe_symbol_count?: number
    correlation_points?: number
    nodes: MarketGnnGraphNode[]
    links: MarketGnnGraphLink[]
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
        live_champions_by_engine: {},
        performance_summary: {},
        horizons: {},
        engines: {},
        nightly_summary: null,
    }, 8000)
}

export async function getLabTrainingStatus(): Promise<TrainingRunStatus> {
    return safeFetch('/api/lab/training/status', {
        status: 'offline',
        engine: null,
        run: {
            run_id: null,
            active: false,
            status: 'idle',
            engine: null,
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

export interface RedTeamWeakness {
    symbol: string
    fragility_score: number
    trades_analyzed: number
    hard_negatives: number
}

export interface RedTeamReport {
    generated_at: string | null
    champion_id?: string
    total_trades_analyzed?: number
    hard_negatives_found?: number
    scenarios_generated?: number
    weaknesses: RedTeamWeakness[]
    failure_type_distribution?: Record<string, number>
    symbol_weakness_score?: Record<string, number>
    champion_survival_score: number
}

export async function getLatestRedTeamReport(): Promise<RedTeamReport> {
    return safeFetch('/api/lab/redteam/latest', {
        generated_at: null,
        champion_survival_score: 100.0,
        weaknesses: [],
    }, 8000)
}

export async function getMarketGnnStatus(): Promise<MarketGnnStatusResponse> {
    return safeFetch('/api/lab/gnn/status', {
        status: 'offline',
        gnn: {
            name: 'market_gnn',
            version: null,
            status: 'unavailable',
            status_reason: 'GNN indisponible.',
            trained_at: null,
            checkpoint_path: null,
            source_run_id: null,
            last_refresh_requested_at: null,
            last_refresh_started_at: null,
            last_refresh_finished_at: null,
            last_refresh_status: 'idle',
            coverage_summary: {},
            refresh_state: {},
            timeframes: [],
            universe: {
                symbols: [],
                count: 0,
                family_counts: {},
            },
            metrics: {
                loss: 0,
                scalp_accuracy: 0,
                intraday_accuracy: 0,
                swing_accuracy: 0,
                epochs: 0,
                batch_size: 0,
                samples: 0,
            },
            artifacts: {
                registry: { path: null, exists: false, size_bytes: null, modified_at: null },
                checkpoint: { path: null, exists: false, size_bytes: null, modified_at: null },
                metrics: { path: null, exists: false, size_bytes: null, modified_at: null },
            },
        },
        graph_readiness: {
            status: 'unavailable',
            reason: 'Graphe indisponible.',
            selected_timeframe: null,
            candidate_timeframes: [],
            overlap_points: 0,
            missing_symbols: [],
        },
        refresh: {
            status: 'idle',
            queued: false,
            requested_at: null,
            started_at: null,
            finished_at: null,
            run_id: null,
            failure_reason: null,
            source_run_id: null,
            requested_by: null,
        },
    }, 8000)
}

export async function getMarketGnnMetrics(): Promise<MarketGnnMetricsResponse> {
    return safeFetch('/api/lab/gnn/metrics', {
        status: 'offline',
        version: null,
        model_status: 'unavailable',
        status_reason: 'GNN indisponible.',
        trained_at: null,
        source_run_id: null,
        coverage_summary: {},
        last_refresh_requested_at: null,
        last_refresh_started_at: null,
        last_refresh_finished_at: null,
        last_refresh_status: 'idle',
        metrics: {
            loss: 0,
            scalp_accuracy: 0,
            intraday_accuracy: 0,
            swing_accuracy: 0,
            epochs: 0,
            batch_size: 0,
            samples: 0,
        },
        universe: {
            symbols: [],
            count: 0,
            family_counts: {},
        },
        timeframes: [],
        artifacts: {
            registry: { path: null, exists: false, size_bytes: null, modified_at: null },
            checkpoint: { path: null, exists: false, size_bytes: null, modified_at: null },
            metrics: { path: null, exists: false, size_bytes: null, modified_at: null },
        },
    }, 8000)
}

// Monitoring systeme et Docker.
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

export async function getMarketGnnGraph(): Promise<MarketGnnGraphSnapshot> {
    return safeFetch('/api/lab/gnn/graph', {
        status: 'unavailable',
        reason: 'GNN indisponible',
        version: null,
        model_status: 'unavailable',
        status_reason: 'GNN indisponible.',
        trained_at: null,
        timeframes: [],
        universe: {
            symbols: [],
            count: 0,
            family_counts: {},
        },
        metrics: {
            loss: 0,
            scalp_accuracy: 0,
            intraday_accuracy: 0,
            swing_accuracy: 0,
            epochs: 0,
            batch_size: 0,
            samples: 0,
        },
        graph_timeframe: null,
        selected_timeframe: null,
        candidate_timeframes: [],
        overlap_points: 0,
        missing_symbols: [],
        coverage_summary: {},
        displayed_symbol_count: 0,
        universe_symbol_count: 0,
        correlation_points: 0,
        nodes: [],
        links: [],
    })
}

export async function requestMarketGnnRefresh(): Promise<{ status: string; refresh: MarketGnnRefreshState; message?: string }> {
    return postJsonWithFallback('/api/lab/gnn/refresh', {}, {
        status: 'error',
        refresh: {
            status: 'idle',
            queued: false,
            requested_at: null,
            started_at: null,
            finished_at: null,
            run_id: null,
            failure_reason: null,
            source_run_id: null,
            requested_by: null,
        },
        message: 'Refresh GNN indisponible.',
    }, 15000)
}

export async function getMarketGnnRefreshStatus(): Promise<{ status: string; refresh: MarketGnnRefreshState; running: boolean; log_path?: string }> {
    return safeFetch('/api/lab/gnn/refresh/status', {
        status: 'offline',
        refresh: {
            status: 'idle',
            queued: false,
            requested_at: null,
            started_at: null,
            finished_at: null,
            run_id: null,
            failure_reason: null,
            source_run_id: null,
            requested_by: null,
        },
        running: false,
        log_path: '',
    })
}

// Services comptables.
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

// Services builder.
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
    const fallback: BuilderHealth = {
        status: 'offline',
        service: 'builder',
        active_pipelines: 0,
        builds_completed: 0,
        forge_runs: 0,
        public_api_entries: 0,
        mutation_enabled: false,
        deploy_enabled: false,
    }
    const payload = await safeFetch('/api/builder/health', fallback)
    return {
        ...fallback,
        ...(payload || {}),
    }
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

export interface MemoryFragment {
    id: string
    content: string
    role: string
    timestamp: string
}

export interface MemorySearchResult {
    id: string
    score: number
    content: string
    role: string
    session_id?: string
    timestamp: string
}

export interface MemoryGraphNode {
    id: string
    label: string
    role?: string
    expert?: string
    timestamp?: string
}

export interface MemoryGraphLink {
    source: string
    target: string
    value: number
}

export interface CoreAgentStatus {
    status: string
    updated_at?: string | null
    last_seen?: string | null
    payload?: Record<string, unknown>
}

export interface ResearchResult {
    title: string
    url: string
    summary: string
    source?: string
    relevance_score?: number
}

export interface ResearchSearchResponse {
    query: string
    domain: string
    results: ResearchResult[]
    synthesis: string
    search_time_ms: number
    timestamp: string
    review_queue?: ResearchQueueSummary
}

export interface ResearchPaper {
    title: string
    summary: string
    url: string
    published?: string
    authors?: string[]
}

export interface ResearchPapersResponse {
    query: string
    category: string
    papers: ResearchPaper[]
    total: number
    review_queue?: ResearchQueueSummary
}

export interface ResearchTrendSource {
    key?: string
    source_key?: string
    source_type?: string
    source_name: string
    family?: string
    last_sync?: string | null
    queued?: number
    approved?: number
    rejected?: number
    ingested?: number
    failed_ingestion?: number
    auto_approved?: number
    auto_ingested?: number
    duplicates?: number
    errors?: number
    review_mode?: 'auto' | 'manual' | string
    trust_level?: 'trusted' | 'review_required' | string
    last_error?: string | null
    ingestion_errors?: number
    durable_ingestion_ready?: boolean
    url?: string
    categories?: string[]
    auto_approve?: boolean
    auto_approve_pattern?: string | null
}

export interface ResearchTrendsResponse {
    domain: string
    sources: string[]
    ingest_sources: ResearchTrendSource[]
    message: string
    timestamp: string
}

export interface ResearchHistoryEntry {
    query: string
    results_count: number
    timestamp: string
}

export interface ResearchQueueSummary {
    queued: number
    auto_approved?: number
    auto_ingested?: number
    duplicates: number
    errors: number
    items?: Array<{ id?: string | null; status: string; reason?: string }>
}

export interface KnowledgeReviewItem {
    id: string
    source_type: string
    source_name: string
    title: string
    url: string
    published_at?: string | null
    authors?: string[]
    origin?: string
    summary_raw?: string
    summary_curated?: string
    tags: string[]
    family: string
    confidence_score: number
    priority_score?: number
    content_hash?: string
    review_status: 'pending' | 'approved' | 'rejected' | 'ingested' | 'failed_ingestion' | string
    metadata?: Record<string, unknown>
    source_key?: string
    review_mode?: 'auto' | 'manual' | string
    trust_level?: 'trusted' | 'review_required' | string
    collected_at?: string
    reviewed_at?: string | null
    reviewed_by?: string | null
    rejection_reason?: string | null
    ingested_at?: string | null
    failed_ingestion_at?: string | null
    ingestion_error?: string | null
}

export interface IngestionDependencyStatus {
    status: 'ok' | 'error' | string
    detail?: string
}

export interface IngestionLogEntry {
    ts?: string
    level?: string
    message?: string
    [key: string]: unknown
}

export interface IngestionStatusResponse {
    status: string
    counts: {
        pending: number
        approved: number
        rejected: number
        ingested: number
        failed_ingestion: number
    }
    active_run: {
        run_id?: string | null
        active: boolean
        status: string
        trigger?: string | null
        strategy?: string | null
        reason?: string | null
        started_at?: string | null
        updated_at?: string | null
        finished_at?: string | null
        current_source?: string | null
    }
    last_run?: Record<string, unknown> | null
    auto_review?: {
        enabled: boolean
        sources: string[]
        policies?: Array<{
            source_key: string
            review_mode: 'auto' | 'manual' | string
            trust_level: 'trusted' | 'review_required' | string
            auto_approve_pattern?: string | null
        }>
    }
    source_stats: Record<string, unknown>
    pending_by_source?: Record<string, number>
    duplicate_rate: number
    dependencies: Record<string, IngestionDependencyStatus>
    durable_ingestion_ready?: boolean
    logs: Array<string | IngestionLogEntry | Record<string, unknown>>
}

export interface IngestionSourcesResponse {
    status: string
    sources: ResearchTrendSource[]
}

export interface ReviewQueueResponse {
    status: string
    review_status: string
    total: number
    items: KnowledgeReviewItem[]
    filters?: Record<string, unknown>
}

export interface ApprovedKnowledgeResponse {
    status: string
    total: number
    items: KnowledgeReviewItem[]
}

export interface ShadowMonitor {
    id: string
    keyword: string
    category: string
    interval_minutes: number
    created_at: string
    status: string
    last_check?: string | null
    hits?: number
}

export interface ShadowAlert {
    id?: string
    severity?: string
    category?: string
    type?: string
    message: string
    timestamp: string
    source?: string
}

export interface ShadowThreatAnalysis {
    indicator: string
    type: string
    threat_score: number
    severity: string
    details?: Record<string, unknown>
    mode?: string
    analyzed_at?: string
}

export interface SentinelAlert {
    severity?: string
    category?: string
    message: string
    timestamp: string
}

export interface SentinelAuditLog {
    action: string
    actor?: string
    target?: string
    details?: string
    severity?: string
    timestamp: string
}

export interface ComplianceAlert {
    severity?: string
    category?: string
    message: string
    timestamp: string
}

export interface ComplianceHistoryEntry {
    timestamp?: string
    amount?: number
    category?: string
    description?: string
    [key: string]: unknown
}

export interface ComplianceUrssafReport {
    period: string
    gross_revenue_quarter: number
    cotisations_urssaf: number
    total_provisions: number
    transactions_count: number
    generated_at: string
}

export interface AccountantDashboard {
    summary: {
        gross_profit: number
        total_taxes: number
        total_expenses: number
        net_roi: number
        currency: string
    }
    expenses_by_category: Record<string, number>
    recent_pnl: Array<Record<string, unknown>>
    expense_count: number
    pnl_count: number
}

export interface RwaHealth {
    status: string
    service: string
    total_assets: number
    total_valuation: number
}

export interface MuseStats {
    total_generations: number
    available_templates: number
    model: string
    mode: string
}

export interface MuseNiche {
    id: string
    label: string
    description: string
    enabled: boolean
    is_nsfw: boolean
    post_interval_hours: number
    recommended_loras: Array<{ filename: string; strength: number }>
}

export interface KernelFeedMessage {
    id: string
    agent: string
    type: string
    content: string
    timestamp: string
    target?: string
}

async function postJsonWithFallback<T>(url: string, body: Record<string, unknown>, fallback: T, timeout = 15000): Promise<T> {
    try {
        const res = await fetchWithTimeout(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        }, timeout)
        if (!res.ok) {
            return fallback
        }
        return await res.json() as T
    } catch {
        return fallback
    }
}

export async function getMemoryFragments(limit = 50): Promise<MemoryFragment[]> {
    return safeFetch(`/api/core/memory/fragments?limit=${limit}`, [], 8000)
}

export async function searchCoreMemory(query: string, limit = 8): Promise<MemorySearchResult[]> {
    if (!query.trim()) {
        return []
    }
    return safeFetch(`/api/core/memory/search?query=${encodeURIComponent(query.trim())}&limit=${limit}`, [], 8000)
}

export async function getCoreAgentsStatus(): Promise<Record<string, CoreAgentStatus>> {
    return safeFetch('/api/core/agents/status', {}, 8000)
}

export async function getCoreAutonomyContext(): Promise<Record<string, unknown> | null> {
    return safeFetch('/api/core/intelligence/autonomy/context', null, 8000)
}

export async function getCoreIntelligenceStatus(): Promise<Record<string, unknown> | null> {
    return safeFetch('/api/core/intelligence/status', null, 8000)
}

export async function searchResearcher(query: string, domain = 'general', maxResults = 5): Promise<ResearchSearchResponse> {
    return postJsonWithFallback('/api/researcher/search', {
        query,
        domain,
        max_results: maxResults,
    }, {
        query,
        domain,
        results: [],
        synthesis: 'Recherche indisponible.',
        search_time_ms: 0,
        timestamp: new Date(0).toISOString(),
        review_queue: {
            queued: 0,
            auto_approved: 0,
            auto_ingested: 0,
            duplicates: 0,
            errors: 0,
            items: [],
        },
    }, 20000)
}

export async function searchResearchPapers(query: string, category = 'cs.AI', maxResults = 5): Promise<ResearchPapersResponse> {
    return postJsonWithFallback('/api/researcher/papers', {
        query,
        category,
        max_results: maxResults,
    }, {
        query,
        category,
        papers: [],
        total: 0,
        review_queue: {
            queued: 0,
            auto_approved: 0,
            auto_ingested: 0,
            duplicates: 0,
            errors: 0,
            items: [],
        },
    }, 20000)
}

export async function getResearcherTrends(domain = 'tech'): Promise<ResearchTrendsResponse> {
    return safeFetch(`/api/researcher/trends?domain=${encodeURIComponent(domain)}`, {
        domain,
        sources: [],
        ingest_sources: [],
        message: 'Veille indisponible.',
        timestamp: new Date(0).toISOString(),
    }, 8000)
}

export async function getResearcherHistory(limit = 20): Promise<{ history: ResearchHistoryEntry[] }> {
    return safeFetch(`/api/researcher/history?limit=${limit}`, { history: [] }, 8000)
}

export async function syncResearchSources(includeArxiv = true, includeNews = true, maxItemsPerSource?: number): Promise<Record<string, unknown>> {
    const payload: Record<string, unknown> = {
        include_arxiv: includeArxiv,
        include_news: includeNews,
        trigger: 'nexus_manual',
    }
    if (typeof maxItemsPerSource === 'number') {
        payload.max_items_per_source = maxItemsPerSource
    }
    return postJsonWithFallback('/api/researcher/ingest/sources/sync', payload, {
        status: 'error',
        message: 'Synchronisation indisponible.',
    }, 30000)
}

export async function getResearchIngestStatus(tail = 20): Promise<IngestionStatusResponse> {
    return safeFetch(`/api/researcher/ingest/status?tail=${tail}`, {
        status: 'offline',
        counts: {
            pending: 0,
            approved: 0,
            rejected: 0,
            ingested: 0,
            failed_ingestion: 0,
        },
        active_run: {
            active: false,
            status: 'idle',
        },
        last_run: null,
        auto_review: {
            enabled: false,
            sources: [],
            policies: [],
        },
        source_stats: {},
        pending_by_source: {},
        duplicate_rate: 0,
        dependencies: {},
        durable_ingestion_ready: false,
        logs: [],
    }, 8000)
}

export async function getResearchReviewQueue(
    reviewStatus = 'pending',
    limit = 50,
    offset = 0,
    filters: {
        sourceKey?: string
        family?: string
        trustLevel?: string
        reviewMode?: string
        search?: string
    } = {},
): Promise<ReviewQueueResponse> {
    const params = new URLSearchParams({
        review_status: reviewStatus,
        limit: String(limit),
        offset: String(offset),
    })
    if (filters.sourceKey) {
        params.set('source_key', filters.sourceKey)
    }
    if (filters.family) {
        params.set('family', filters.family)
    }
    if (filters.trustLevel) {
        params.set('trust_level', filters.trustLevel)
    }
    if (filters.reviewMode) {
        params.set('review_mode', filters.reviewMode)
    }
    if (filters.search) {
        params.set('search', filters.search)
    }
    return safeFetch(
        `/api/researcher/ingest/review?${params.toString()}`,
        {
            status: 'offline',
            review_status: reviewStatus,
            total: 0,
            items: [],
            filters: {},
        },
        8000,
    )
}

export async function approveResearchReviewItem(itemId: string, reviewedBy = 'nexus'): Promise<{ status: string; item?: KnowledgeReviewItem }> {
    return postJsonWithFallback(`/api/researcher/ingest/review/${encodeURIComponent(itemId)}/approve`, {
        reviewed_by: reviewedBy,
        reason: 'Validation depuis Nexus',
    }, { status: 'error' }, 20000)
}

export async function rejectResearchReviewItem(itemId: string, reason: string, reviewedBy = 'nexus'): Promise<{ status: string; item?: KnowledgeReviewItem }> {
    return postJsonWithFallback(`/api/researcher/ingest/review/${encodeURIComponent(itemId)}/reject`, {
        reviewed_by: reviewedBy,
        reason,
    }, { status: 'error' }, 20000)
}

export async function retryResearchReviewIngestion(itemId: string, reviewedBy = 'nexus:retry'): Promise<{ status: string; item?: KnowledgeReviewItem; reason?: string }> {
    return postJsonWithFallback(`/api/researcher/ingest/review/${encodeURIComponent(itemId)}/retry-ingestion`, {
        reviewed_by: reviewedBy,
        reason: 'Reprise de l ingestion durable depuis Nexus',
    }, { status: 'error' }, 20000)
}

export async function autoApproveResearchReviewItems(sourcePattern?: string, limit = 500, reviewedBy = 'nexus:auto'): Promise<Record<string, unknown>> {
    const payload: Record<string, unknown> = {
        reviewed_by: reviewedBy,
        limit,
    }
    if (sourcePattern) {
        payload.source_pattern = sourcePattern
    }
    return postJsonWithFallback('/api/researcher/ingest/review/auto-approve', payload, {
        status: 'error',
        matched: 0,
        approved: 0,
        ingested: 0,
        errors: 0,
        items: [],
    }, 30000)
}

export async function getApprovedKnowledge(limit = 30): Promise<ApprovedKnowledgeResponse> {
    return safeFetch(`/api/researcher/ingest/approved?limit=${limit}`, {
        status: 'offline',
        total: 0,
        items: [],
    }, 8000)
}

export async function getResearchSources(): Promise<IngestionSourcesResponse> {
    return safeFetch('/api/researcher/ingest/sources', {
        status: 'offline',
        sources: [],
    }, 8000)
}

export async function getResearchPeaAnalysis(): Promise<Record<string, unknown>> {
    return safeFetch('/api/researcher/pea-analysis', {
        status: 'info',
        message: 'Analyse PEA indisponible.',
        targets: [],
    }, 8000)
}

export async function searchShadow(query: string, maxResults = 10): Promise<{ query: string; results: ResearchResult[]; count: number }> {
    const payload = await safeFetch<{ query: string; results: Array<Record<string, unknown>>; count: number }>(`/api/shadow/search?q=${encodeURIComponent(query)}&max_results=${maxResults}`, {
        query,
        results: [],
        count: 0,
    }, 10000)
    return {
        query: payload.query,
        count: payload.count,
        results: (payload.results || []).map((result) => ({
            title: String(result.title || 'Resultat'),
            url: String(result.url || ''),
            summary: String(result.summary || result.snippet || ''),
            source: typeof result.source === 'string' ? result.source : 'shadow',
            relevance_score: Number(result.relevance_score || 0),
        })),
    }
}

export async function reconShadow(target: string): Promise<Record<string, unknown>> {
    const payload = await safeFetch<Record<string, unknown>>(`/api/shadow/recon?target=${encodeURIComponent(target)}`, {}, 12000)
    if (Array.isArray(payload.web_findings)) {
        payload.web_findings = payload.web_findings.map((result) => ({
            ...result,
            summary: String((result as Record<string, unknown>).summary || (result as Record<string, unknown>).snippet || ''),
        }))
    }
    return payload
}

export async function getShadowThreatHistory(): Promise<{ analyses: ShadowThreatAnalysis[]; total: number }> {
    return safeFetch('/api/shadow/threats/history', { analyses: [], total: 0 }, 8000)
}

export async function getShadowMonitors(): Promise<{ monitors: ShadowMonitor[]; total_active: number }> {
    return safeFetch('/api/shadow/monitor', { monitors: [], total_active: 0 }, 8000)
}

export async function getShadowAlerts(limit = 50): Promise<{ alerts: ShadowAlert[]; total: number }> {
    return safeFetch(`/api/shadow/alerts?limit=${limit}`, { alerts: [], total: 0 }, 8000)
}

export async function getSentinelMetrics(): Promise<Record<string, unknown> | null> {
    return safeFetch('/api/sentinel/metrics', null, 8000)
}

export async function getSentinelAlerts(limit = 50): Promise<{ alerts: SentinelAlert[]; total: number }> {
    return safeFetch(`/api/sentinel/alerts?limit=${limit}`, { alerts: [], total: 0 }, 8000)
}

export async function getSentinelAuditLogs(limit = 50): Promise<{ logs: SentinelAuditLog[]; total: number }> {
    return safeFetch(`/api/sentinel/audit/logs?limit=${limit}`, { logs: [], total: 0 }, 8000)
}

export async function getSentinelQuarantine(): Promise<{ quarantined: Array<Record<string, unknown>>; total: number }> {
    return safeFetch('/api/sentinel/quarantine', { quarantined: [], total: 0 }, 8000)
}

export async function getSentinelComplianceCheck(): Promise<Record<string, unknown> | null> {
    return safeFetch('/api/sentinel/compliance/check', null, 8000)
}

export async function getSubstrateMetrics(): Promise<Record<string, unknown> | null> {
    return safeFetch('/api/substrate/metrics', null, 8000)
}

export async function getSubstrateGpu(): Promise<Record<string, unknown> | null> {
    return safeFetch('/api/substrate/gpu', null, 8000)
}

export async function getSubstrateMetricsHistory(limit = 60): Promise<{ history: Array<Record<string, unknown>>; count: number }> {
    return safeFetch(`/api/substrate/metrics/history?limit=${limit}`, { history: [], count: 0 }, 8000)
}

export async function getSubstrateAlerts(limit = 50): Promise<{ alerts: Array<Record<string, unknown>>; total: number }> {
    return safeFetch(`/api/substrate/alerts?limit=${limit}`, { alerts: [], total: 0 }, 8000)
}

export async function getSubstrateEcoQueue(): Promise<{ queue: Array<Record<string, unknown>>; total: number }> {
    return safeFetch('/api/substrate/eco/queue', { queue: [], total: 0 }, 8000)
}

export async function getComplianceHealth(): Promise<Record<string, unknown> | null> {
    return safeFetch('/api/compliance/health', null, 8000)
}

export async function getComplianceHistory(limit = 50): Promise<{ provisions: ComplianceHistoryEntry[]; total: number }> {
    return safeFetch(`/api/compliance/history?limit=${limit}`, { provisions: [], total: 0 }, 8000)
}

export async function getComplianceUrssafReport(): Promise<ComplianceUrssafReport | null> {
    return safeFetch('/api/compliance/report/urssaf', null, 8000)
}

export async function getComplianceAlerts(limit = 50): Promise<{ alerts: ComplianceAlert[]; total: number }> {
    return safeFetch(`/api/compliance/alerts?limit=${limit}`, { alerts: [], total: 0 }, 8000)
}

export async function getAccountantDashboard(): Promise<AccountantDashboard | null> {
    return safeFetch('/api/accountant/dashboard', null, 8000)
}

export async function getRwaHealth(): Promise<RwaHealth | null> {
    return safeFetch('/api/rwa/health', null, 8000)
}

export async function getRwaPortfolio(): Promise<Record<string, unknown> | null> {
    return safeFetch('/api/rwa/portfolio', null, 8000)
}

export async function getRwaStrategy(): Promise<Record<string, unknown> | null> {
    return safeFetch('/api/rwa/strategy', null, 8000)
}

export async function getRwaRecommendations(): Promise<Record<string, unknown> | null> {
    return safeFetch('/api/rwa/strategy/recommendations', null, 8000)
}

export async function getRwaTelemetry(): Promise<Record<string, unknown> | null> {
    return safeFetch('/api/rwa/iot/telemetry', null, 8000)
}

export async function getRwaEnergyHistory(days = 7): Promise<Record<string, unknown> | null> {
    return safeFetch(`/api/rwa/iot/energy/history?days=${days}`, null, 8000)
}

export async function getMuseStats(): Promise<MuseStats | null> {
    return safeFetch('/api/muse/stats', null, 8000)
}

export async function getMuseNiches(): Promise<MuseNiche[]> {
    const payload = await safeFetch<{ niches?: MuseNiche[] }>('/api/muse/niches', {}, 8000)
    return payload.niches || []
}

export async function getMuseNicheScores(): Promise<Record<string, number>> {
    const payload = await safeFetch<{ scores?: Record<string, number> }>('/api/muse/niches/scores', {}, 12000)
    return payload.scores || {}
}

export async function getKernelRecentFeed(limit = 20): Promise<{ available: boolean; messages: KernelFeedMessage[] }> {
    try {
        const res = await fetchWithTimeout(`/api/kernel/feed/recent?limit=${limit}`, {}, 5000)
        if (res.status === 404) {
            return { available: false, messages: [] }
        }
        if (!res.ok) {
            return { available: true, messages: [] }
        }
        const payload = await res.json()
        const rawMessages = Array.isArray(payload?.messages) ? payload.messages : []
        const messages = rawMessages.map((item: Record<string, unknown>, index: number) => ({
            id: typeof item.id === 'string' && item.id.trim() ? item.id : `${index}-${String(item.timestamp || '')}`,
            agent: typeof item.agent === 'string' && item.agent.trim() ? item.agent : String(item.source_agent || item.source || item.service || 'system'),
            type: typeof item.type === 'string' && item.type.trim() ? item.type : String(item.kind || item.message_type || 'message'),
            content: typeof item.content === 'string' && item.content.trim()
                ? item.content
                : typeof item.message === 'string' && item.message.trim()
                    ? item.message
                    : typeof item.summary === 'string' && item.summary.trim()
                        ? item.summary
                        : 'Message indisponible',
            timestamp: typeof item.timestamp === 'string' && item.timestamp.trim()
                ? item.timestamp
                : new Date().toISOString(),
            target: typeof item.target === 'string' && item.target.trim() ? item.target : undefined,
        }))
        return { available: true, messages }
    } catch {
        return { available: false, messages: [] }
    }
}
