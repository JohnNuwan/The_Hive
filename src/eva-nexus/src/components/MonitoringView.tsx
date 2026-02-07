import { useState, useEffect, useCallback } from 'react'
import { getSystemMetrics, getDockerContainers, type SystemMetrics, type ContainerStats } from '../services/api'

// ═══════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════

function formatBytes(bytes: number): string {
    if (bytes < 1024) return `${bytes.toFixed(0)} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`
}

function formatUptime(seconds: number): string {
    const d = Math.floor(seconds / 86400)
    const h = Math.floor((seconds % 86400) / 3600)
    const m = Math.floor((seconds % 3600) / 60)
    return `${d}d ${h}h ${m}m`
}

function getUsageColor(percent: number): string {
    if (percent < 50) return 'bg-matrix'
    if (percent < 75) return 'bg-cyber-amber'
    return 'bg-cyber-pink'
}

function getUsageTextColor(percent: number): string {
    if (percent < 50) return 'neon-text'
    if (percent < 75) return 'neon-text-amber'
    return 'neon-text-pink'
}

function getTempColor(temp: number): string {
    if (temp < 50) return 'neon-text'
    if (temp < 70) return 'neon-text-amber'
    return 'neon-text-pink'
}

// ═══════════════════════════════════════════════════════
// GAUGE COMPONENT
// ═══════════════════════════════════════════════════════

function CircularGauge({ value, label, size = 100 }: {
    value: number; label: string; size?: number
}) {
    const radius = (size - 10) / 2
    const circumference = 2 * Math.PI * radius
    const progress = Math.min(value / 100, 1) * circumference
    const textClass = getUsageTextColor(value)

    return (
        <div className="flex flex-col items-center gap-1">
            <svg width={size} height={size} className="transform -rotate-90">
                <circle cx={size / 2} cy={size / 2} r={radius} fill="none" stroke="rgba(255,255,255,0.03)" strokeWidth={4} />
                <circle cx={size / 2} cy={size / 2} r={radius} fill="none"
                    stroke={value < 50 ? 'rgba(0,255,65,0.7)' : value < 75 ? 'rgba(240,165,0,0.7)' : 'rgba(255,0,60,0.7)'}
                    strokeWidth={4} strokeDasharray={circumference} strokeDashoffset={circumference - progress}
                    strokeLinecap="round" style={{ transition: 'stroke-dashoffset 0.5s ease, stroke 0.5s ease' }} />
            </svg>
            <div className="absolute flex flex-col items-center justify-center" style={{ width: size, height: size }}>
                <span className={`text-lg font-bold ${textClass}`}>{value.toFixed(1)}%</span>
            </div>
            <span className="text-[8px] text-white/25 tracking-wider uppercase">{label}</span>
        </div>
    )
}

// ═══════════════════════════════════════════════════════
// USAGE BAR
// ═══════════════════════════════════════════════════════

function UsageBar({ percent, label, used, total }: {
    percent: number; label: string; used?: string; total?: string
}) {
    return (
        <div className="space-y-1">
            <div className="flex items-center justify-between">
                <span className="text-[9px] text-white/30 tracking-wider">{label}</span>
                <span className={`text-[9px] font-bold ${getUsageTextColor(percent)}`}>{percent.toFixed(1)}%</span>
            </div>
            <div className="h-[6px] bg-white/[0.03] rounded-sm overflow-hidden">
                <div className={`h-full rounded-sm transition-all duration-700 ${getUsageColor(percent)}`}
                    style={{
                        width: `${Math.min(percent, 100)}%`,
                        boxShadow: percent > 75 ? '0 0 8px rgba(255,0,60,0.4)' : percent > 50 ? '0 0 8px rgba(240,165,0,0.3)' : '0 0 8px rgba(0,255,65,0.3)'
                    }} />
            </div>
            {used && total && (
                <div className="flex justify-between text-[8px] text-white/15">
                    <span>{used}</span>
                    <span>{total}</span>
                </div>
            )}
        </div>
    )
}

// ═══════════════════════════════════════════════════════
// CONTAINER ROW
// ═══════════════════════════════════════════════════════

function ContainerRow({ container }: { container: ContainerStats }) {
    const isRunning = container.status === 'running'
    const statusColor = isRunning ? 'bg-matrix' : container.status === 'restarting' ? 'bg-cyber-amber animate-pulse' : 'bg-cyber-pink'
    const statusTextColor = isRunning ? 'text-matrix/60' : container.status === 'restarting' ? 'text-cyber-amber/60' : 'text-cyber-pink/60'

    return (
        <div className={`flex items-center gap-3 py-2 px-3 border-b border-white/[0.02] hover:bg-white/[0.01] transition-all ${!isRunning ? 'opacity-50' : ''}`}>
            <div className={`w-2 h-2 rounded-full shrink-0 ${statusColor}`}
                style={isRunning ? { boxShadow: '0 0 6px rgba(0,255,65,0.4)' } : {}} />
            <div className="w-36 shrink-0">
                <div className="text-[10px] text-white/70 font-bold tracking-wider">{container.name}</div>
                <div className="text-[7px] text-white/15 truncate">{container.image}</div>
            </div>
            <div className={`w-20 shrink-0 text-[8px] ${statusTextColor} tracking-wider uppercase`}>
                {container.status}
            </div>
            <div className="w-20 shrink-0">
                <div className="text-[9px] text-white/15 tracking-wider mb-0.5">CPU</div>
                <div className="h-[3px] bg-white/[0.03] rounded overflow-hidden">
                    <div className={`h-full ${getUsageColor(container.cpu_percent)} transition-all`}
                        style={{ width: `${Math.min(container.cpu_percent, 100)}%` }} />
                </div>
                <div className={`text-[8px] mt-0.5 ${getUsageTextColor(container.cpu_percent)}`}>
                    {container.cpu_percent.toFixed(1)}%
                </div>
            </div>
            <div className="w-24 shrink-0">
                <div className="text-[9px] text-white/15 tracking-wider mb-0.5">MEM</div>
                <div className="h-[3px] bg-white/[0.03] rounded overflow-hidden">
                    <div className={`h-full ${getUsageColor(container.memory_percent)} transition-all`}
                        style={{ width: `${Math.min(container.memory_percent, 100)}%` }} />
                </div>
                <div className="text-[8px] text-white/30 mt-0.5">
                    {container.memory_usage.toFixed(0)} / {container.memory_limit} MB
                </div>
            </div>
            <div className="w-28 shrink-0 text-[8px]">
                <span className="text-matrix/40">↓</span>
                <span className="text-white/25 ml-1">{formatBytes(container.network_rx)}</span>
                <span className="text-cyber-cyan/40 ml-2">↑</span>
                <span className="text-white/25 ml-1">{formatBytes(container.network_tx)}</span>
            </div>
            <div className="flex-1 flex items-center justify-end gap-4 text-[8px] text-white/15">
                <span>PIDs: {container.pids}</span>
                <span>{container.uptime}</span>
            </div>
        </div>
    )
}

// ═══════════════════════════════════════════════════════
// SPARKLINE CHART
// ═══════════════════════════════════════════════════════

function Sparkline({ data, color = 'rgba(0,255,65,0.6)', height = 40 }: { data: number[]; color?: string; height?: number }) {
    const width = 200
    const max = Math.max(...data, 1)
    const points = data.map((v, i) => {
        const x = (i / (data.length - 1)) * width
        const y = height - (v / max) * height
        return `${x},${y}`
    }).join(' ')

    return (
        <svg width={width} height={height} className="w-full" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
            <defs>
                <linearGradient id="spark-fill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={color} stopOpacity="0.2" />
                    <stop offset="100%" stopColor={color} stopOpacity="0" />
                </linearGradient>
            </defs>
            <polygon points={`0,${height} ${points} ${width},${height}`} fill="url(#spark-fill)" />
            <polyline points={points} fill="none" stroke={color} strokeWidth="1.5" />
        </svg>
    )
}

// ═══════════════════════════════════════════════════════
// MAIN MONITORING VIEW
// ═══════════════════════════════════════════════════════

export default function MonitoringView() {
    const [metrics, setMetrics] = useState<SystemMetrics | null>(null)
    const [containers, setContainers] = useState<ContainerStats[]>([])
    const [cpuHistory, setCpuHistory] = useState<number[]>(() => Array.from({ length: 60 }, () => 0))
    const [memHistory, setMemHistory] = useState<number[]>(() => Array.from({ length: 60 }, () => 0))
    const [netRxHistory, setNetRxHistory] = useState<number[]>(() => Array.from({ length: 60 }, () => 0))
    const [sortBy, setSortBy] = useState<'name' | 'cpu' | 'memory'>('cpu')
    const [dataSource, setDataSource] = useState<'real' | 'unavailable'>('unavailable')
    const [lastUpdate, setLastUpdate] = useState<Date | null>(null)

    // Fetch real metrics from backend
    const refreshMetrics = useCallback(async () => {
        const data = await getSystemMetrics()
        if (data) {
            setMetrics(data)
            setDataSource(data.real_data ? 'real' : 'unavailable')
            setCpuHistory(prev => [...prev.slice(-59), data.cpu.usage])
            setMemHistory(prev => [...prev.slice(-59), data.memory.percent])
            setNetRxHistory(prev => [...prev.slice(-59), data.network.rx_speed || 0])
            setLastUpdate(new Date())
        } else {
            setDataSource('unavailable')
        }
    }, [])

    const refreshContainers = useCallback(async () => {
        const data = await getDockerContainers()
        if (data && data.length > 0) {
            setContainers(data)
        }
    }, [])

    useEffect(() => {
        refreshMetrics()
        refreshContainers()
        const metricsInterval = setInterval(refreshMetrics, 3000)
        const containerInterval = setInterval(refreshContainers, 5000)
        return () => {
            clearInterval(metricsInterval)
            clearInterval(containerInterval)
        }
    }, [refreshMetrics, refreshContainers])

    const sortedContainers = [...containers].sort((a, b) => {
        if (sortBy === 'cpu') return b.cpu_percent - a.cpu_percent
        if (sortBy === 'memory') return b.memory_percent - a.memory_percent
        return a.name.localeCompare(b.name)
    })

    const runningCount = containers.filter(c => c.status === 'running').length
    const totalCpuUsage = containers.reduce((s, c) => s + c.cpu_percent, 0)
    const totalMemUsage = containers.reduce((s, c) => s + c.memory_usage, 0)

    return (
        <div className="h-full overflow-y-auto p-4 space-y-4 animate-fade-in">
            {/* ═══ DATA SOURCE BANNER ═══ */}
            <div className={`px-3 py-2 border text-[9px] tracking-wider flex items-center justify-between ${dataSource === 'real'
                ? 'border-matrix/20 bg-matrix/5 text-matrix/60'
                : 'border-cyber-amber/20 bg-cyber-amber/5 text-cyber-amber/60'}`}>
                <div className="flex items-center gap-2">
                    <div className={`w-2 h-2 rounded-full ${dataSource === 'real' ? 'bg-matrix animate-pulse' : 'bg-cyber-amber'}`}
                        style={dataSource === 'real' ? { boxShadow: '0 0 6px rgba(0,255,65,0.4)' } : {}} />
                    <span>{dataSource === 'real' ? '● LIVE DATA — Connected to EVA Core via psutil + Docker' : '● AWAITING CONNECTION — Start EVA Core for live metrics'}</span>
                </div>
                {lastUpdate && (
                    <span className="text-white/15">
                        Last: {lastUpdate.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                    </span>
                )}
            </div>

            {/* ═══ TOP GAUGES ═══ */}
            {metrics ? (
                <>
                    <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
                        {/* CPU */}
                        <div className="cyber-panel hud-corners p-4">
                            <div className="text-[9px] uppercase tracking-[0.2em] text-matrix/40 mb-3">CPU</div>
                            <div className="flex items-center gap-4">
                                <div className="relative">
                                    <CircularGauge value={metrics.cpu.usage} label="" size={80} />
                                </div>
                                <div className="flex-1 space-y-1">
                                    <div className="text-[9px] text-white/40">{metrics.cpu.model}</div>
                                    <div className="text-[9px] text-white/25">{metrics.cpu.cores} Cores</div>
                                    {metrics.cpu.temp > 0 && (
                                        <div className="flex items-center gap-1">
                                            <span className="text-[8px] text-white/20">TEMP:</span>
                                            <span className={`text-[9px] font-bold ${getTempColor(metrics.cpu.temp)}`}>{metrics.cpu.temp}°C</span>
                                        </div>
                                    )}
                                </div>
                            </div>
                            <div className="mt-3"><Sparkline data={cpuHistory} color="rgba(0,255,65,0.6)" height={30} /></div>
                        </div>

                        {/* RAM */}
                        <div className="cyber-panel hud-corners p-4">
                            <div className="text-[9px] uppercase tracking-[0.2em] text-matrix/40 mb-3">MEMORY</div>
                            <div className="flex items-center gap-4">
                                <div className="relative">
                                    <CircularGauge value={metrics.memory.percent} label="" size={80} />
                                </div>
                                <div className="flex-1 space-y-1">
                                    <div className="text-[10px] text-white/50 font-bold">{metrics.memory.used.toFixed(1)} GB</div>
                                    <div className="text-[9px] text-white/20">/ {metrics.memory.total.toFixed(0)} GB</div>
                                </div>
                            </div>
                            <div className="mt-3"><Sparkline data={memHistory} color="rgba(0,212,255,0.6)" height={30} /></div>
                        </div>

                        {/* GPU */}
                        <div className="cyber-panel hud-corners p-4">
                            <div className="text-[9px] uppercase tracking-[0.2em] text-matrix/40 mb-3">GPU</div>
                            {metrics.gpu ? (
                                <>
                                    <div className="flex items-center gap-4">
                                        <div className="relative">
                                            <CircularGauge value={metrics.gpu.usage} label="" size={80} />
                                        </div>
                                        <div className="flex-1 space-y-1">
                                            <div className="text-[9px] text-white/40">{metrics.gpu.name}</div>
                                            <div className="text-[9px] text-white/25">VRAM: {metrics.gpu.memory_used.toFixed(1)}/{metrics.gpu.memory_total}GB</div>
                                            <div className="flex items-center gap-1">
                                                <span className="text-[8px] text-white/20">TEMP:</span>
                                                <span className={`text-[9px] font-bold ${getTempColor(metrics.gpu.temp)}`}>{metrics.gpu.temp}°C</span>
                                            </div>
                                        </div>
                                    </div>
                                    <div className="mt-3">
                                        <UsageBar percent={(metrics.gpu.memory_used / metrics.gpu.memory_total) * 100} label="VRAM"
                                            used={`${metrics.gpu.memory_used.toFixed(1)}GB`} total={`${metrics.gpu.memory_total}GB`} />
                                    </div>
                                </>
                            ) : (
                                <div className="flex items-center justify-center h-20 text-[10px] text-white/15">
                                    NO GPU DETECTED
                                </div>
                            )}
                        </div>

                        {/* DISK */}
                        <div className="cyber-panel hud-corners p-4">
                            <div className="text-[9px] uppercase tracking-[0.2em] text-matrix/40 mb-3">STORAGE</div>
                            <div className="flex items-center gap-4 mb-3">
                                <div className="relative">
                                    <CircularGauge value={metrics.disk.percent} label="" size={80} />
                                </div>
                                <div className="flex-1 space-y-1">
                                    <div className="text-[10px] text-white/50 font-bold">{metrics.disk.used} GB</div>
                                    <div className="text-[9px] text-white/20">/ {metrics.disk.total} GB</div>
                                </div>
                            </div>
                            <div className="space-y-2">
                                <div className="flex items-center justify-between text-[8px]">
                                    <span className="text-white/20">READ</span>
                                    <span className="text-matrix/50">{metrics.disk.read_speed.toFixed(1)} MB/s</span>
                                </div>
                                <div className="flex items-center justify-between text-[8px]">
                                    <span className="text-white/20">WRITE</span>
                                    <span className="text-cyber-cyan/50">{metrics.disk.write_speed.toFixed(1)} MB/s</span>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* ═══ NETWORK + SUMMARY ═══ */}
                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
                        <div className="cyber-panel hud-corners p-4">
                            <div className="text-[9px] uppercase tracking-[0.2em] text-matrix/40 mb-3">🌐 NETWORK I/O</div>
                            <div className="grid grid-cols-2 gap-4 mb-3">
                                <div>
                                    <div className="text-[8px] text-white/20 tracking-wider mb-1">DOWNLOAD</div>
                                    <div className="text-lg font-bold neon-text">{metrics.network.rx_speed.toFixed(1)}</div>
                                    <div className="text-[8px] text-white/20">MB/s</div>
                                    <div className="text-[8px] text-white/10 mt-1">Total: {formatBytes(metrics.network.rx_bytes)}</div>
                                </div>
                                <div>
                                    <div className="text-[8px] text-white/20 tracking-wider mb-1">UPLOAD</div>
                                    <div className="text-lg font-bold neon-text-cyan">{metrics.network.tx_speed.toFixed(1)}</div>
                                    <div className="text-[8px] text-white/20">MB/s</div>
                                    <div className="text-[8px] text-white/10 mt-1">Total: {formatBytes(metrics.network.tx_bytes)}</div>
                                </div>
                            </div>
                            <Sparkline data={netRxHistory} color="rgba(0,212,255,0.6)" height={30} />
                        </div>

                        <div className="cyber-panel hud-corners p-4">
                            <div className="text-[9px] uppercase tracking-[0.2em] text-matrix/40 mb-3">🐳 DOCKER SUMMARY</div>
                            <div className="grid grid-cols-2 gap-3">
                                <div className="p-2 border border-white/[0.03] bg-white/[0.01]">
                                    <div className="text-[7px] text-white/15 tracking-wider">RUNNING</div>
                                    <div className="text-xl font-bold neon-text">{runningCount}</div>
                                </div>
                                <div className="p-2 border border-white/[0.03] bg-white/[0.01]">
                                    <div className="text-[7px] text-white/15 tracking-wider">TOTAL</div>
                                    <div className="text-xl font-bold text-white/50">{containers.length}</div>
                                </div>
                                <div className="p-2 border border-white/[0.03] bg-white/[0.01]">
                                    <div className="text-[7px] text-white/15 tracking-wider">CPU (ALL)</div>
                                    <div className={`text-sm font-bold ${getUsageTextColor(totalCpuUsage / Math.max(metrics.cpu.cores, 1))}`}>
                                        {totalCpuUsage.toFixed(1)}%
                                    </div>
                                </div>
                                <div className="p-2 border border-white/[0.03] bg-white/[0.01]">
                                    <div className="text-[7px] text-white/15 tracking-wider">MEM (ALL)</div>
                                    <div className="text-sm font-bold text-white/50">{(totalMemUsage / 1024).toFixed(1)} GB</div>
                                </div>
                            </div>
                        </div>

                        <div className="cyber-panel hud-corners p-4">
                            <div className="text-[9px] uppercase tracking-[0.2em] text-matrix/40 mb-3">⚡ SYSTEM</div>
                            <div className="space-y-2">
                                <div className="flex justify-between text-[9px]">
                                    <span className="text-white/20">UPTIME</span>
                                    <span className="text-white/50 font-bold">{formatUptime(metrics.uptime)}</span>
                                </div>
                                <div className="flex justify-between text-[9px]">
                                    <span className="text-white/20">HOSTNAME</span>
                                    <span className="text-white/35">{metrics.hostname || 'N/A'}</span>
                                </div>
                                <div className="flex justify-between text-[9px]">
                                    <span className="text-white/20">PLATFORM</span>
                                    <span className="text-white/35">{metrics.platform || 'Docker Compose'} (Dev)</span>
                                </div>
                                <div className="flex justify-between text-[9px]">
                                    <span className="text-white/20">TARGET</span>
                                    <span className="text-cyber-cyan/40">Proxmox VE (Prod)</span>
                                </div>
                                <div className="flex justify-between text-[9px]">
                                    <span className="text-white/20">DATA SOURCE</span>
                                    <span className={dataSource === 'real' ? 'neon-text text-[9px]' : 'text-cyber-amber text-[9px]'}>
                                        {dataSource === 'real' ? 'LIVE (psutil)' : 'AWAITING'}
                                    </span>
                                </div>
                            </div>
                        </div>
                    </div>
                </>
            ) : (
                /* No metrics available — show placeholder */
                <div className="cyber-panel hud-corners p-8 text-center">
                    <div className="text-3xl mb-4">📊</div>
                    <div className="text-sm text-white/30 tracking-wider mb-2">SYSTEM METRICS UNAVAILABLE</div>
                    <div className="text-[10px] text-white/15 max-w-md mx-auto">
                        Start EVA Core (port 8000) to get real-time system metrics via psutil.
                        <br />CPU, RAM, GPU, Disk, and Network data will appear here.
                    </div>
                    <div className="mt-4 text-[9px] text-matrix/30 tracking-wider">
                        <code>docker-compose up core</code> or <code>uvicorn eva_core.main:app --port 8000</code>
                    </div>
                </div>
            )}

            {/* ═══ DOCKER CONTAINERS TABLE ═══ */}
            <div className="cyber-panel hud-corners p-4">
                <div className="flex items-center justify-between mb-3">
                    <div className="text-[9px] uppercase tracking-[0.2em] text-matrix/40">
                        🐳 CONTAINERS ({containers.length})
                        {containers.length === 0 && <span className="ml-2 text-white/15">— Start Docker for live data</span>}
                    </div>
                    <div className="flex items-center gap-2">
                        <span className="text-[8px] text-white/15">Sort:</span>
                        {(['name', 'cpu', 'memory'] as const).map(s => (
                            <button key={s} onClick={() => setSortBy(s)}
                                className={`text-[8px] px-2 py-0.5 border transition-all ${sortBy === s
                                    ? 'border-matrix/30 text-matrix/70 bg-matrix/5'
                                    : 'border-white/[0.04] text-white/20 hover:text-white/40'}`}>
                                {s.toUpperCase()}
                            </button>
                        ))}
                    </div>
                </div>

                {containers.length > 0 ? (
                    <>
                        <div className="flex items-center gap-3 py-1.5 px-3 text-[8px] text-white/15 tracking-wider border-b border-white/[0.04] mb-1">
                            <div className="w-2 shrink-0" />
                            <div className="w-36 shrink-0">NAME</div>
                            <div className="w-20 shrink-0">STATUS</div>
                            <div className="w-20 shrink-0">CPU</div>
                            <div className="w-24 shrink-0">MEMORY</div>
                            <div className="w-28 shrink-0">NETWORK</div>
                            <div className="flex-1 text-right">PIDS / UPTIME</div>
                        </div>
                        <div className="max-h-[400px] overflow-y-auto custom-scrollbar">
                            {sortedContainers.map(c => <ContainerRow key={c.id} container={c} />)}
                        </div>
                    </>
                ) : (
                    <div className="py-6 text-center text-[10px] text-white/15">
                        No Docker containers detected. Make sure Docker is running.
                    </div>
                )}
            </div>

            {/* ═══ RESOURCE ALLOCATION ═══ */}
            {sortedContainers.filter(c => c.status === 'running').length > 0 && (
                <div className="cyber-panel hud-corners p-4">
                    <div className="text-[9px] uppercase tracking-[0.2em] text-matrix/40 mb-3">📊 RESOURCE ALLOCATION BY SERVICE</div>
                    <div className="space-y-2">
                        {sortedContainers
                            .filter(c => c.status === 'running')
                            .slice(0, 10)
                            .map(c => (
                                <div key={c.name} className="flex items-center gap-3">
                                    <div className="w-28 text-[9px] text-white/40 tracking-wider truncate">{c.name}</div>
                                    <div className="flex-1 flex items-center gap-2">
                                        <div className="flex-1">
                                            <div className="h-[4px] bg-white/[0.03] rounded overflow-hidden">
                                                <div className={`h-full ${getUsageColor(c.cpu_percent)} transition-all duration-700`}
                                                    style={{ width: `${Math.min(c.cpu_percent * 5, 100)}%` }} />
                                            </div>
                                        </div>
                                        <span className="text-[8px] text-white/25 w-12 text-right">{c.cpu_percent.toFixed(1)}%</span>
                                        <div className="flex-1">
                                            <div className="h-[4px] bg-white/[0.03] rounded overflow-hidden">
                                                <div className="h-full bg-cyber-cyan/50 transition-all duration-700"
                                                    style={{ width: `${Math.min(c.memory_percent, 100)}%` }} />
                                            </div>
                                        </div>
                                        <span className="text-[8px] text-white/25 w-16 text-right">{c.memory_usage.toFixed(0)}MB</span>
                                    </div>
                                </div>
                            ))}
                    </div>
                    <div className="flex items-center gap-4 mt-3 text-[7px] text-white/15">
                        <span className="flex items-center gap-1"><span className="w-3 h-1 bg-matrix/60 rounded" /> CPU</span>
                        <span className="flex items-center gap-1"><span className="w-3 h-1 bg-cyber-cyan/50 rounded" /> MEMORY</span>
                    </div>
                </div>
            )}
        </div>
    )
}
