import { useState, useEffect, useRef } from 'react'
import {
    getAllNodesHealth, getKillSwitchStatus, getNemesisStatus,
    getNewsFilter, getCoreTelemetry, getCoreCircuitBreaker,
    getTradingStatus, toggleKillSwitch,
    type NodeHealth, type KillSwitchStatus, type NemesisStatus,
    type NewsFilterStatus, type TelemetryData, type CircuitBreakerStatus
} from '../services/api'

// ═══ SIMULATED ACTIVITY LOG ═══
const SIM_LOGS = [
    { agent: 'KERNEL', msg: 'Heartbeat received — All systems nominal', type: 'info' },
    { agent: 'BANKER', msg: 'Risk assessment complete — Exposure: 1.2%', type: 'info' },
    { agent: 'SENTINEL', msg: 'Network scan — 0 anomalies detected', type: 'success' },
    { agent: 'NERVOUS', msg: 'Route signal P0 → kernel_action (0.3ms)', type: 'info' },
    { agent: 'QUANT', msg: 'Monte Carlo VaR: $234.56 (10K paths, 8ms)', type: 'info' },
    { agent: 'CORE', msg: 'LLM routing complete — Expert: Researcher', type: 'info' },
    { agent: 'KERNEL', msg: 'Constitution integrity verified — Hash: 0xAF3C', type: 'success' },
    { agent: 'BANKER', msg: 'Drone #3 scaled — Volatility threshold crossed', type: 'warning' },
    { agent: 'SENTINEL', msg: 'Threat level: GREEN — Perimeter secure', type: 'success' },
    { agent: 'NERVOUS', msg: 'Swarm heartbeat broadcast (5 agents)', type: 'info' },
    { agent: 'QUANT', msg: 'Black-Scholes CALL: $12.45 — Greeks calculated', type: 'info' },
    { agent: 'CORE', msg: 'Memory consolidation — 847 vectors indexed', type: 'info' },
]

interface LogEntry {
    time: string
    agent: string
    msg: string
    type: string
}

export default function Dashboard() {
    const [nodes, setNodes] = useState<NodeHealth[]>([])
    const [killSwitch, setKillSwitch] = useState<KillSwitchStatus>({ is_active: false, message: 'LOADING...' })
    const [nemesis, setNemesis] = useState<NemesisStatus>({ total_defeats: 0, known_nemeses: {}, trading_blocked: false, blocked_until: null })
    const [newsFilter, setNewsFilter] = useState<NewsFilterStatus>({ is_active: false, blocked_until: null, next_high_impact_events: [] })
    const [telemetry, setTelemetry] = useState<TelemetryData | null>(null)
    const [circuitBreaker, setCircuitBreaker] = useState<CircuitBreakerStatus | null>(null)
    const [tradingData, setTradingData] = useState<{ equity: number; pnl: number; positions: number }>({ equity: 0, pnl: 0, positions: 0 })
    const [logs, setLogs] = useState<LogEntry[]>([])
    const [killSwitchLoading, setKillSwitchLoading] = useState(false)
    const logRef = useRef<HTMLDivElement>(null)

    // Simulated log feed
    useEffect(() => {
        const addLog = () => {
            const entry = SIM_LOGS[Math.floor(Math.random() * SIM_LOGS.length)]
            const now = new Date()
            const time = now.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
            setLogs(prev => [...prev.slice(-30), { ...entry, time }])
        }
        addLog()
        const interval = setInterval(addLog, 3000 + Math.random() * 4000)
        return () => clearInterval(interval)
    }, [])

    // Auto-scroll logs
    useEffect(() => {
        if (logRef.current) {
            logRef.current.scrollTop = logRef.current.scrollHeight
        }
    }, [logs])

    // Fetch all data
    useEffect(() => {
        const fetchAll = async () => {
            const [nodesData, ksData, nemData, newsData, telData, cbData, tradData] = await Promise.all([
                getAllNodesHealth(),
                getKillSwitchStatus(),
                getNemesisStatus(),
                getNewsFilter(),
                getCoreTelemetry(),
                getCoreCircuitBreaker(),
                getTradingStatus(),
            ])
            setNodes(nodesData)
            setKillSwitch(ksData)
            setNemesis(nemData)
            setNewsFilter(newsData)
            setTelemetry(telData)
            setCircuitBreaker(cbData)
            const pnl = tradData.positions.reduce((sum: number, p: any) => sum + (p.profit || 0), 0)
            setTradingData({ equity: tradData.account.equity, pnl, positions: tradData.positions.length })
        }
        fetchAll()
        const interval = setInterval(fetchAll, 8000)
        return () => clearInterval(interval)
    }, [])

    const handleKillSwitch = async () => {
        setKillSwitchLoading(true)
        const action = killSwitch.is_active ? 'reset' : 'activate'
        const result = await toggleKillSwitch(action)
        setKillSwitch(result)
        setKillSwitchLoading(false)
    }

    const nodeIcons: Record<string, string> = {
        'EVA Core': '🧠', 'Banker': '💰', 'Kernel': '🔒', 'Nervous': '⚡',
        'Sentinel': '👁', 'Quant': '🧮'
    }
    const nodeLangs: Record<string, string> = {
        'EVA Core': 'PYTHON', 'Banker': 'PYTHON', 'Kernel': 'RUST',
        'Nervous': 'GO', 'Sentinel': 'PYTHON', 'Quant': 'JULIA'
    }

    return (
        <div className="h-full overflow-y-auto p-4 space-y-3 animate-fade-in">
            {/* ═══ TOP ROW: System Nodes ═══ */}
            <div className="grid grid-cols-4 gap-3">
                {nodes.map((node, i) => (
                    <div key={node.name} className="cyber-panel hud-corners p-3" style={{ animationDelay: `${i * 100}ms` }}>
                        <div className="flex items-center justify-between mb-2">
                            <span className="text-lg">{nodeIcons[node.name] || '⬡'}</span>
                            <div className={node.status === 'online' ? 'status-online' : node.status === 'degraded' ? 'status-warning' : 'status-offline'} />
                        </div>
                        <div className="text-[11px] font-bold text-white/90 tracking-wider">{node.name.toUpperCase()}</div>
                        <div className="flex items-center justify-between mt-1">
                            <span className="text-[9px] text-matrix/50 tracking-widest">{nodeLangs[node.name] || 'SYS'}</span>
                            <span className={`text-[9px] tracking-wider ${node.status === 'online' ? 'text-matrix' : node.status === 'degraded' ? 'text-cyber-amber' : 'text-cyber-pink'}`}>
                                {node.status === 'online' ? `${node.latency}ms` : node.status.toUpperCase()}
                            </span>
                        </div>
                    </div>
                ))}
            </div>

            {/* ═══ MIDDLE ROW: Kill-Switch / Equity / Nemesis / Circuit Breaker ═══ */}
            <div className="grid grid-cols-4 gap-3">
                {/* Kill-Switch */}
                <div className={`cyber-panel hud-corners p-4 ${killSwitch.is_active ? 'border-cyber-pink/40' : 'border-matrix/20'}`}>
                    <div className="text-[9px] uppercase tracking-[0.2em] text-matrix/40 mb-3">KILL-SWITCH</div>
                    <div className="flex items-center justify-between mb-3">
                        <span className={`font-display text-xl font-bold tracking-wider ${killSwitch.is_active ? 'neon-text-pink' : 'neon-text'}`}>
                            {killSwitch.is_active ? 'HALTED' : 'ARMED'}
                        </span>
                        <div className={`w-3 h-3 rounded-full ${killSwitch.is_active ? 'bg-cyber-pink animate-pulse shadow-[0_0_10px_rgba(255,0,60,0.6)]' : 'bg-matrix shadow-[0_0_10px_rgba(0,255,65,0.4)]'}`} />
                    </div>
                    <button
                        onClick={handleKillSwitch}
                        disabled={killSwitchLoading}
                        className={`w-full py-2 text-[10px] uppercase tracking-[0.15em] transition-all ${killSwitch.is_active ? 'cyber-btn' : 'cyber-btn cyber-btn-danger'}`}
                    >
                        {killSwitchLoading ? '...' : killSwitch.is_active ? '↻ RESET SYSTEM' : '⚡ EMERGENCY HALT'}
                    </button>
                </div>

                {/* Equity / P&L */}
                <div className="cyber-panel hud-corners p-4">
                    <div className="text-[9px] uppercase tracking-[0.2em] text-matrix/40 mb-3">PORTFOLIO</div>
                    <div className="font-display text-xl font-bold tracking-wider neon-text-cyan mb-1">
                        ${tradingData.equity > 0 ? tradingData.equity.toLocaleString() : '—'}
                    </div>
                    <div className="flex items-center gap-2">
                        <span className={`text-xs font-bold ${tradingData.pnl >= 0 ? 'text-matrix' : 'text-cyber-pink'}`}>
                            {tradingData.pnl >= 0 ? '+' : ''}{tradingData.pnl.toFixed(2)}$
                        </span>
                        <span className="text-[9px] text-white/20">•</span>
                        <span className="text-[9px] text-white/40">{tradingData.positions} positions</span>
                    </div>
                </div>

                {/* Nemesis */}
                <div className={`cyber-panel hud-corners p-4 ${nemesis.trading_blocked ? 'border-cyber-amber/30' : ''}`}>
                    <div className="text-[9px] uppercase tracking-[0.2em] text-matrix/40 mb-3">NEMESIS SYSTEM</div>
                    <div className="flex items-center gap-3 mb-2">
                        <span className="font-display text-xl font-bold tracking-wider neon-text-amber">{nemesis.total_defeats}</span>
                        <span className="text-[10px] text-white/30">DEFEATS</span>
                    </div>
                    <div className="flex flex-wrap gap-1">
                        {Object.keys(nemesis.known_nemeses).length > 0
                            ? Object.entries(nemesis.known_nemeses).map(([type, count]) => (
                                <span key={type} className="text-[8px] px-1.5 py-0.5 bg-cyber-amber/10 border border-cyber-amber/20 text-cyber-amber">
                                    {type.replace('_', ' ')} ×{count}
                                </span>
                            ))
                            : <span className="text-[9px] text-white/20">NO KNOWN THREATS</span>
                        }
                    </div>
                    {nemesis.trading_blocked && (
                        <div className="mt-2 text-[9px] text-cyber-pink animate-pulse">⚠ TRADING BLOCKED — MEDITATION</div>
                    )}
                </div>

                {/* Circuit Breaker */}
                <div className="cyber-panel hud-corners p-4">
                    <div className="text-[9px] uppercase tracking-[0.2em] text-matrix/40 mb-3">CIRCUIT BREAKER</div>
                    {circuitBreaker ? (
                        <>
                            <div className={`font-display text-lg font-bold tracking-wider mb-2 ${circuitBreaker.state === 'CLOSED' ? 'neon-text' : circuitBreaker.state === 'OPEN' ? 'neon-text-pink' : 'neon-text-amber'
                                }`}>
                                {circuitBreaker.state}
                            </div>
                            <div className="flex items-center gap-2">
                                <div className="flex-1 h-1 bg-white/5 rounded overflow-hidden">
                                    <div
                                        className={`h-full transition-all ${circuitBreaker.failures > 3 ? 'bg-cyber-pink' : 'bg-matrix/60'}`}
                                        style={{ width: `${(circuitBreaker.failures / circuitBreaker.failure_threshold) * 100}%` }}
                                    />
                                </div>
                                <span className="text-[9px] text-white/40">{circuitBreaker.failures}/{circuitBreaker.failure_threshold}</span>
                            </div>
                        </>
                    ) : (
                        <div className="text-[10px] text-white/20">NO DATA</div>
                    )}
                </div>
            </div>

            {/* ═══ BOTTOM ROW: Telemetry / News / Activity Feed ═══ */}
            <div className="grid grid-cols-3 gap-3">
                {/* Telemetry */}
                <div className="cyber-panel hud-corners p-4">
                    <div className="text-[9px] uppercase tracking-[0.2em] text-matrix/40 mb-3">TELEMETRY</div>
                    {telemetry ? (
                        <div className="space-y-2">
                            <TelemetryRow label="UPTIME" value={formatUptime(telemetry.uptime_seconds)} />
                            <TelemetryRow label="REQUESTS" value={telemetry.requests_total.toLocaleString()} />
                            <TelemetryRow label="ERRORS" value={String(telemetry.errors_total)} color={telemetry.errors_total > 0 ? 'pink' : 'green'} />
                        </div>
                    ) : (
                        <div className="space-y-2">
                            <TelemetryRow label="UPTIME" value="—" />
                            <TelemetryRow label="REQUESTS" value="—" />
                            <TelemetryRow label="ERRORS" value="—" />
                        </div>
                    )}
                </div>

                {/* News Filter */}
                <div className={`cyber-panel hud-corners p-4 ${newsFilter.is_active ? 'border-cyber-pink/30' : ''}`}>
                    <div className="text-[9px] uppercase tracking-[0.2em] text-matrix/40 mb-3">NEWS FILTER</div>
                    <div className={`font-display text-sm font-bold tracking-wider mb-2 ${newsFilter.is_active ? 'neon-text-pink' : 'neon-text'}`}>
                        {newsFilter.is_active ? '🚫 BLOCKED' : '✓ CLEAR'}
                    </div>
                    {newsFilter.is_active && newsFilter.blocked_until && (
                        <div className="text-[9px] text-cyber-pink mb-2">Until: {new Date(newsFilter.blocked_until).toLocaleTimeString()}</div>
                    )}
                    <div className="space-y-1">
                        {newsFilter.next_high_impact_events.length > 0
                            ? newsFilter.next_high_impact_events.slice(0, 3).map((e, i) => (
                                <div key={i} className="text-[9px] text-white/40 flex items-center gap-1.5">
                                    <span className={`w-1.5 h-1.5 rounded-full ${e.impact === 'High' ? 'bg-cyber-pink' : 'bg-cyber-amber'}`} />
                                    {e.event}
                                </div>
                            ))
                            : <div className="text-[9px] text-white/20">No upcoming events</div>
                        }
                    </div>
                </div>

                {/* Activity Feed */}
                <div className="cyber-panel hud-corners p-4 md:row-span-1">
                    <div className="text-[9px] uppercase tracking-[0.2em] text-matrix/40 mb-3">ACTIVITY FEED</div>
                    <div ref={logRef} className="h-[120px] overflow-y-auto space-y-1 font-mono">
                        {logs.map((log, i) => (
                            <div key={i} className="text-[10px] leading-relaxed flex gap-2 animate-fade-in">
                                <span className="text-white/15 shrink-0">{log.time}</span>
                                <span className={`shrink-0 ${log.type === 'warning' ? 'text-cyber-amber' : log.type === 'success' ? 'text-matrix' : 'text-cyber-cyan/60'}`}>
                                    {log.agent}
                                </span>
                                <span className="text-white/35 truncate">{'>'} {log.msg}</span>
                            </div>
                        ))}
                        <div className="terminal-cursor text-[10px]">&nbsp;</div>
                    </div>
                </div>
            </div>

            {/* ═══ NEW ROW: Visual Probes (Grafana) ═══ */}
            <div className="grid grid-cols-2 gap-3">
                <div className="cyber-panel hud-corners p-4">
                    <div className="text-[9px] uppercase tracking-[0.2em] text-matrix/40 mb-3">VISUAL PROBE: SWARM LOGS (LOKI)</div>
                    <div className="aspect-video bg-black/40 border border-matrix/10 rounded flex flex-col items-center justify-center space-y-3">
                        <span className="text-[10px] text-matrix/60">LIVE LOKI FEED AGGREGATOR</span>
                        <a
                            href="http://localhost:3000/explore?orgId=1&left=%5B%22now-1h%22,%22now%22,%22Loki%22,%7B%22expr%22:%22%7Bcontainer%3D~%5C%22hive-.*%5C%22%7D%22%7D%5D"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="cyber-btn text-[9px] px-4 py-1.5"
                        >
                            🔗 OPEN EXPLORER
                        </a>
                    </div>
                </div>
                <div className="cyber-panel hud-corners p-4">
                    <div className="text-[9px] uppercase tracking-[0.2em] text-matrix/40 mb-3">VISUAL PROBE: PERFORMANCE (PROMETHEUS)</div>
                    <div className="aspect-video bg-black/40 border border-matrix/10 rounded flex flex-col items-center justify-center space-y-3">
                        <span className="text-[10px] text-matrix/60">NERVOUS SYSTEM THROUGHPUT</span>
                        <a
                            href="http://localhost:3000/dashboards"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="cyber-btn text-[9px] px-4 py-1.5"
                        >
                            🔗 OPEN DASHBOARDS
                        </a>
                    </div>
                </div>
            </div>
        </div>
    )
}

function TelemetryRow({ label, value, color }: { label: string; value: string; color?: string }) {
    return (
        <div className="flex items-center justify-between">
            <span className="text-[9px] text-white/30 tracking-wider">{label}</span>
            <span className={`text-xs font-bold tracking-wider ${color === 'pink' ? 'text-cyber-pink' : color === 'green' ? 'text-matrix' : 'text-white/70'}`}>
                {value}
            </span>
        </div>
    )
}

function formatUptime(seconds: number): string {
    if (!seconds) return '—'
    const h = Math.floor(seconds / 3600)
    const m = Math.floor((seconds % 3600) / 60)
    const s = Math.floor(seconds % 60)
    return `${h}h ${m}m ${s}s`
}
