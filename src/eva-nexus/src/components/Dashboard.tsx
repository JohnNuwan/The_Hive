import { useEffect, useMemo, useRef, useState } from 'react'
import { ShieldAlert } from 'lucide-react'

import {
    getAllNodesHealth,
    getCoreCircuitBreaker,
    getCoreTelemetry,
    getKernelRecentFeed,
    getKillSwitchStatus,
    getNemesisStatus,
    getNewsFilter,
    getTradingStatus,
    toggleKillSwitch,
    type CircuitBreakerStatus,
    type KernelFeedMessage,
    type KillSwitchStatus,
    type NemesisStatus,
    type NewsFilterStatus,
    type NodeHealth,
    type TelemetryData,
} from '../services/api'

interface LogEntry {
    id: string
    time: string
    agent: string
    msg: string
    type: 'info' | 'warning' | 'success'
}

function formatUptime(seconds: number): string {
    if (!seconds) {
        return '--'
    }
    const hours = Math.floor(seconds / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    const remainingSeconds = Math.floor(seconds % 60)
    return `${hours}h ${minutes}m ${remainingSeconds}s`
}

function formatFeedMessage(message: KernelFeedMessage): LogEntry {
    const parsedTime = new Date(message.timestamp)
    const time = Number.isNaN(parsedTime.getTime())
        ? '--:--:--'
        : parsedTime.toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
    const type = message.type.toLowerCase() === 'result'
        ? 'success'
        : message.type.toLowerCase() === 'error'
            ? 'warning'
            : 'info'

    return {
        id: message.id,
        time,
        agent: message.agent.toUpperCase(),
        msg: message.content,
        type,
    }
}

function TelemetryRow({ label, value, color }: { label: string; value: string; color?: 'green' | 'pink' }) {
    return (
        <div className="flex items-center justify-between">
            <span className="text-[9px] text-white/30 tracking-wider">{label}</span>
            <span className={`text-xs font-bold tracking-wider ${color === 'pink' ? 'text-cyber-pink' : color === 'green' ? 'text-matrix' : 'text-white/70'}`}>
                {value}
            </span>
        </div>
    )
}

export default function Dashboard() {
    const grafanaBaseUrl = `${window.location.protocol}//${window.location.hostname}:3000`
    const [nodes, setNodes] = useState<NodeHealth[]>([])
    const [bankerOnline, setBankerOnline] = useState(false)
    const [killSwitch, setKillSwitch] = useState<KillSwitchStatus>({ is_active: false, message: 'CHARGEMENT...' })
    const [nemesis, setNemesis] = useState<NemesisStatus>({ total_defeats: 0, known_nemeses: {}, trading_blocked: false, blocked_until: null })
    const [newsFilter, setNewsFilter] = useState<NewsFilterStatus>({ is_active: false, blocked_until: null, next_high_impact_events: [] })
    const [telemetry, setTelemetry] = useState<TelemetryData | null>(null)
    const [circuitBreaker, setCircuitBreaker] = useState<CircuitBreakerStatus | null>(null)
    const [tradingData, setTradingData] = useState<{ equity: number; pnl: number; positions: number }>({ equity: 0, pnl: 0, positions: 0 })
    const [logs, setLogs] = useState<LogEntry[]>([])
    const [logsAvailable, setLogsAvailable] = useState<boolean | null>(null)
    const [killSwitchLoading, setKillSwitchLoading] = useState(false)
    const logRef = useRef<HTMLDivElement>(null)

    useEffect(() => {
        if (logRef.current) {
            logRef.current.scrollTop = logRef.current.scrollHeight
        }
    }, [logs])

    useEffect(() => {
        const fetchAll = async () => {
            const nodesData = await getAllNodesHealth()
            const bankerOnline = nodesData.some((node) => node.name === 'Banker' && node.status === 'online')
            const [killSwitchData, nemesisData, newsData, telemetryData, breakerData, tradingStatus] = await Promise.all([
                getKillSwitchStatus(),
                bankerOnline
                    ? getNemesisStatus()
                    : Promise.resolve({ total_defeats: 0, known_nemeses: {}, trading_blocked: false, blocked_until: null }),
                bankerOnline
                    ? getNewsFilter()
                    : Promise.resolve({ is_active: false, blocked_until: null, next_high_impact_events: [] }),
                getCoreTelemetry(),
                getCoreCircuitBreaker(),
                bankerOnline
                    ? getTradingStatus()
                    : Promise.resolve({ account: { equity: 0 }, positions: [], risk: { daily_drawdown_percent: 0, trading_allowed: false } }),
            ])

            setNodes(nodesData)
            setBankerOnline(bankerOnline)
            setKillSwitch(killSwitchData)
            setNemesis(nemesisData)
            setNewsFilter(newsData)
            setTelemetry(telemetryData)
            setCircuitBreaker(breakerData)
            const pnl = tradingStatus.positions.reduce((sum: number, position: any) => sum + (position.profit || 0), 0)
            setTradingData({
                equity: tradingStatus.account.equity,
                pnl,
                positions: tradingStatus.positions.length,
            })
        }

        void fetchAll()
        const interval = setInterval(() => {
            void fetchAll()
        }, 8000)
        return () => clearInterval(interval)
    }, [])

    useEffect(() => {
        const refreshLogs = async () => {
            const payload = await getKernelRecentFeed(20)
            setLogsAvailable(payload.available)
            setLogs(payload.available ? payload.messages.map(formatFeedMessage) : [])
        }

        void refreshLogs()
        const interval = setInterval(() => {
            void refreshLogs()
        }, 5000)
        return () => clearInterval(interval)
    }, [])

    const handleKillSwitch = async () => {
        if (!bankerOnline) {
            return
        }
        setKillSwitchLoading(true)
        const action = killSwitch.is_active ? 'reset' : 'activate'
        const result = await toggleKillSwitch(action)
        setKillSwitch(result)
        setKillSwitchLoading(false)
    }

    const nodeIcons: Record<string, string> = {
        'EVA Core': 'EC',
        Banker: 'BK',
        Sentinel: 'ST',
        Shadow: 'SH',
        Researcher: 'RS',
        Wraith: 'WR',
    }

    const nodeLangs: Record<string, string> = {
        'EVA Core': 'PYTHON',
        Banker: 'PYTHON',
        Sentinel: 'PYTHON',
        Shadow: 'PYTHON',
        Researcher: 'PYTHON',
        Wraith: 'PYTHON',
    }

    const logsStatusLabel = useMemo(() => {
        if (logsAvailable === false) {
            return 'donnee indisponible'
        }
        return 'donnee reelle'
    }, [logsAvailable])

    return (
        <div className="h-full overflow-y-auto p-4 space-y-3 animate-fade-in">
            <div className="grid grid-cols-4 gap-3">
                {nodes.map((node, index) => (
                    <div key={node.name} className="cyber-panel hud-corners p-3" style={{ animationDelay: `${index * 100}ms` }}>
                        <div className="flex items-center justify-between mb-2">
                            <span className="text-lg">{nodeIcons[node.name] || 'NA'}</span>
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

            <div className="grid grid-cols-4 gap-3">
                <div className={`cyber-panel hud-corners p-4 transition-all duration-300 ${killSwitch.is_active ? 'border-red-500/50 shadow-[0_0_30px_rgba(239,68,68,0.15)]' : ''}`}>
                    <div className="flex items-center justify-between mb-3">
                        <div className="text-[9px] uppercase tracking-[0.2em] text-matrix/40">CONTROLE SYSTEME</div>
                        {killSwitch.is_active && <ShieldAlert size={12} className="text-red-500 animate-pulse" />}
                    </div>

                    <button
                        onClick={handleKillSwitch}
                        disabled={killSwitchLoading || !bankerOnline}
                        className={`w-full h-10 rounded text-[10px] font-black uppercase tracking-[0.2em] transition-all duration-300 flex items-center justify-center gap-2 ${killSwitch.is_active
                            ? 'bg-red-500 text-black hover:bg-red-400 shadow-[0_0_15px_#ef4444] scale-[1.02]'
                            : 'bg-white/5 text-slate-400 hover:bg-red-500/10 hover:text-red-400 border border-white/5 hover:border-red-500/30'
                            }`}
                    >
                        {killSwitchLoading ? (
                            <span className="animate-pulse">TRAITEMENT...</span>
                        ) : (
                            <>
                                <div className={`w-1.5 h-1.5 rounded-full ${killSwitch.is_active ? 'bg-black animate-pulse' : 'bg-red-500'}`} />
                                {killSwitch.is_active ? 'RESET PROTOCOL' : 'KILL-SWITCH'}
                            </>
                        )}
                    </button>

                    <div className="text-[8px] text-center mt-2 font-mono">
                        {!bankerOnline ? (
                            <span className="text-cyber-amber">BANKER INDISPONIBLE - CONTROLE EN ATTENTE</span>
                        ) : killSwitch.is_active ? (
                            <span className="text-red-500 animate-pulse">TRADING STOPPE - SECURITE ACTIVE</span>
                        ) : (
                            <span className="text-matrix/30">SYSTEMES NOMINAUX - TRADING ACTIF</span>
                        )}
                    </div>
                </div>

                <div className="cyber-panel hud-corners p-4">
                    <div className="text-[9px] uppercase tracking-[0.2em] text-matrix/40 mb-3">PORTEFEUILLE</div>
                    <div className="font-display text-xl font-bold tracking-wider neon-text-cyan mb-1">
                        {bankerOnline ? `$${tradingData.equity > 0 ? tradingData.equity.toLocaleString('fr-FR') : '--'}` : 'indisponible'}
                    </div>
                    <div className="flex items-center gap-2">
                        {bankerOnline ? (
                            <>
                                <span className={`text-xs font-bold ${tradingData.pnl >= 0 ? 'text-matrix' : 'text-cyber-pink'}`}>
                                    {tradingData.pnl >= 0 ? '+' : ''}{tradingData.pnl.toFixed(2)}$
                                </span>
                                <span className="text-[9px] text-white/20">|</span>
                                <span className="text-[9px] text-white/40">{tradingData.positions} positions</span>
                            </>
                        ) : (
                            <span className="text-[9px] text-cyber-amber/70">Donnee portefeuille indisponible</span>
                        )}
                    </div>
                </div>

                <div className={`cyber-panel hud-corners p-4 ${nemesis.trading_blocked ? 'border-cyber-amber/30' : ''}`}>
                    <div className="text-[9px] uppercase tracking-[0.2em] text-matrix/40 mb-3">SYSTEME NEMESIS</div>
                    {!bankerOnline ? (
                        <div className="text-[10px] text-cyber-amber/70">Banker hors ligne. Etat Nemesis indisponible.</div>
                    ) : (
                        <>
                            <div className="flex items-center gap-3 mb-2">
                                <span className="font-display text-xl font-bold tracking-wider neon-text-amber">{nemesis.total_defeats}</span>
                                <span className="text-[10px] text-white/30">DEFAITES</span>
                            </div>
                            <div className="flex flex-wrap gap-1">
                                {Object.keys(nemesis.known_nemeses).length > 0 ? (
                                    Object.entries(nemesis.known_nemeses).map(([type, count]) => (
                                        <span key={type} className="text-[8px] px-1.5 py-0.5 bg-cyber-amber/10 border border-cyber-amber/20 text-cyber-amber">
                                            {type.replace('_', ' ')} x{count}
                                        </span>
                                    ))
                                ) : (
                                    <span className="text-[9px] text-white/20">AUCUNE MENACE CONNUE</span>
                                )}
                            </div>
                            {nemesis.trading_blocked && (
                                <div className="mt-2 text-[9px] text-cyber-pink animate-pulse">TRADING BLOQUE - MEDITATION</div>
                            )}
                        </>
                    )}
                </div>

                <div className="cyber-panel hud-corners p-4">
                    <div className="text-[9px] uppercase tracking-[0.2em] text-matrix/40 mb-3">COUPE-CIRCUIT</div>
                    {circuitBreaker ? (
                        <>
                            <div className={`font-display text-lg font-bold tracking-wider mb-2 ${circuitBreaker.state === 'CLOSED' ? 'neon-text' : circuitBreaker.state === 'OPEN' ? 'neon-text-pink' : 'neon-text-amber'}`}>
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
                        <div className="text-[10px] text-white/20">DONNEE INDISPONIBLE</div>
                    )}
                </div>
            </div>

            <div className="grid grid-cols-3 gap-3">
                <div className="cyber-panel hud-corners p-4">
                    <div className="text-[9px] uppercase tracking-[0.2em] text-matrix/40 mb-3">TELEMETRIE</div>
                    {telemetry ? (
                        <div className="space-y-2">
                            <TelemetryRow label="UPTIME" value={formatUptime(telemetry.uptime_seconds)} />
                            <TelemetryRow label="REQUETES" value={telemetry.requests_total.toLocaleString('fr-FR')} />
                            <TelemetryRow label="ERREURS" value={String(telemetry.errors_total)} color={telemetry.errors_total > 0 ? 'pink' : 'green'} />
                        </div>
                    ) : (
                        <div className="space-y-2">
                            <TelemetryRow label="UPTIME" value="indisponible" />
                            <TelemetryRow label="REQUETES" value="indisponible" />
                            <TelemetryRow label="ERREURS" value="indisponible" />
                        </div>
                    )}
                </div>

                <div className={`cyber-panel hud-corners p-4 ${newsFilter.is_active ? 'border-cyber-pink/30' : ''}`}>
                    <div className="text-[9px] uppercase tracking-[0.2em] text-matrix/40 mb-3">FILTRE NEWS</div>
                    {!bankerOnline ? (
                        <div className="text-[10px] text-cyber-amber/70">Filtre news indisponible tant que Banker est hors ligne.</div>
                    ) : (
                        <>
                            <div className={`font-display text-sm font-bold tracking-wider mb-2 ${newsFilter.is_active ? 'neon-text-pink' : 'neon-text'}`}>
                                {newsFilter.is_active ? 'BLOQUE' : 'CLAIR'}
                            </div>
                            {newsFilter.is_active && newsFilter.blocked_until && (
                                <div className="text-[9px] text-cyber-pink mb-2">Jusqu a: {new Date(newsFilter.blocked_until).toLocaleTimeString('fr-FR')}</div>
                            )}
                            <div className="space-y-1">
                                {newsFilter.next_high_impact_events.length > 0 ? (
                                    newsFilter.next_high_impact_events.slice(0, 3).map((event, index) => (
                                        <div key={index} className="text-[9px] text-white/40 flex items-center gap-1.5">
                                            <span className={`w-1.5 h-1.5 rounded-full ${event.impact === 'High' ? 'bg-cyber-pink' : 'bg-cyber-amber'}`} />
                                            {event.event}
                                        </div>
                                    ))
                                ) : (
                                    <div className="text-[9px] text-white/20">Aucun evenement proche</div>
                                )}
                            </div>
                        </>
                    )}
                </div>

                <div className="cyber-panel hud-corners p-4 md:row-span-1">
                    <div className="flex items-center justify-between gap-3 mb-3">
                        <div className="text-[9px] uppercase tracking-[0.2em] text-matrix/40">FLUX ACTIVITE</div>
                        <span className={`text-[8px] uppercase tracking-[0.18em] ${logsAvailable === false ? 'text-cyber-amber/70' : 'text-matrix/50'}`}>
                            {logsStatusLabel}
                        </span>
                    </div>
                    <div ref={logRef} className="h-[120px] overflow-y-auto space-y-1 font-mono">
                        {logsAvailable === false ? (
                            <div className="text-[10px] text-white/25">Flux kernel indisponible. Aucun journal simule n est affiche.</div>
                        ) : logs.length === 0 ? (
                            <div className="text-[10px] text-white/25">Aucune donnee recente recue.</div>
                        ) : logs.map((log) => (
                            <div key={log.id} className="text-[10px] leading-relaxed flex gap-2 animate-fade-in">
                                <span className="text-white/15 shrink-0">{log.time}</span>
                                <span className={`shrink-0 ${log.type === 'warning' ? 'text-cyber-amber' : log.type === 'success' ? 'text-matrix' : 'text-cyber-cyan/60'}`}>
                                    {log.agent}
                                </span>
                                <span className="text-white/35 truncate">&gt; {log.msg}</span>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
                <div className="cyber-panel hud-corners p-4">
                    <div className="text-[9px] uppercase tracking-[0.2em] text-matrix/40 mb-3">SONDE VISUELLE : LOGS SWARM (LOKI)</div>
                    <div className="aspect-video bg-black/40 border border-matrix/10 rounded flex flex-col items-center justify-center space-y-3">
                        <span className="text-[10px] text-matrix/60">AGREGATEUR LOKI</span>
                        <a
                            href={`${grafanaBaseUrl}/explore?orgId=1&left=%5B%22now-1h%22,%22now%22,%22Loki%22,%7B%22expr%22:%22%7Bcontainer%3D~%5C%22hive-.*%5C%22%7D%22%7D%5D`}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="cyber-btn text-[9px] px-4 py-1.5"
                        >
                            OUVRIR EXPLORER
                        </a>
                    </div>
                </div>
                <div className="cyber-panel hud-corners p-4">
                    <div className="text-[9px] uppercase tracking-[0.2em] text-matrix/40 mb-3">SONDE VISUELLE : PERFORMANCE (PROMETHEUS)</div>
                    <div className="aspect-video bg-black/40 border border-matrix/10 rounded flex flex-col items-center justify-center space-y-3">
                        <span className="text-[10px] text-matrix/60">DEBIT DU NERVOUS SYSTEM</span>
                        <a
                            href={`${grafanaBaseUrl}/dashboards`}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="cyber-btn text-[9px] px-4 py-1.5"
                        >
                            OUVRIR DASHBOARDS
                        </a>
                    </div>
                </div>
            </div>
        </div>
    )
}
