import { useEffect, useMemo, useRef, useState } from 'react'
import ForceGraph2D from 'react-force-graph-2d'
import {
    Activity,
    CircleOff,
    Clock3,
    Database,
    GitBranch,
    Loader2,
    Network,
    ShieldCheck,
    Target,
} from 'lucide-react'

import {
    getMarketGnnGraph,
    getMarketGnnMetrics,
    getMarketGnnRefreshStatus,
    getMarketGnnStatus,
    requestMarketGnnRefresh,
    type MarketGnnGraphNode,
    type MarketGnnGraphSnapshot,
    type MarketGnnMetricsResponse,
    type MarketGnnRefreshState,
    type MarketGnnStatusResponse,
} from '../services/api'

function formatDateTime(value?: string | null): string {
    if (!value) {
        return '--'
    }
    const date = new Date(value)
    if (Number.isNaN(date.getTime())) {
        return '--'
    }
    return date.toLocaleString('fr-FR', {
        day: '2-digit',
        month: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
    })
}

function formatPercent(value?: number | null, digits = 2): string {
    if (typeof value !== 'number' || Number.isNaN(value)) {
        return '--'
    }
    return `${value.toFixed(digits)}%`
}

function statusTone(status?: string): string {
    switch ((status || '').toLowerCase()) {
        case 'live':
            return 'text-emerald-400 border-emerald-500/30 bg-emerald-500/10'
        case 'validated':
            return 'text-sky-400 border-sky-500/30 bg-sky-500/10'
        case 'draft':
            return 'text-amber-400 border-amber-500/30 bg-amber-500/10'
        case 'stale':
            return 'text-rose-400 border-rose-500/30 bg-rose-500/10'
        case 'unavailable':
        default:
            return 'text-slate-400 border-white/10 bg-white/[0.04]'
    }
}

function familyTone(family: string): string {
    switch (family) {
        case 'crypto':
            return 'text-sky-300 border-sky-500/20 bg-sky-500/10'
        case 'forex':
            return 'text-emerald-300 border-emerald-500/20 bg-emerald-500/10'
        case 'index_cfd':
            return 'text-amber-300 border-amber-500/20 bg-amber-500/10'
        case 'metal':
            return 'text-violet-300 border-violet-500/20 bg-violet-500/10'
        default:
            return 'text-slate-300 border-white/10 bg-white/[0.04]'
    }
}

function infoCard(label: string, value: string, meta?: string) {
    return (
        <div className="rounded-2xl border border-white/8 bg-black/20 px-4 py-3">
            <div className="text-[9px] font-black uppercase tracking-[0.22em] text-slate-500">{label}</div>
            <div className="mt-2 text-lg font-black text-white/90">{value}</div>
            {meta ? <div className="mt-2 text-[10px] text-slate-500">{meta}</div> : null}
        </div>
    )
}

export default function GraphView() {
    const [isLoading, setIsLoading] = useState(true)
    const [statusData, setStatusData] = useState<MarketGnnStatusResponse | null>(null)
    const [metricsData, setMetricsData] = useState<MarketGnnMetricsResponse | null>(null)
    const [graphData, setGraphData] = useState<MarketGnnGraphSnapshot | null>(null)
    const [refreshState, setRefreshState] = useState<MarketGnnRefreshState | null>(null)
    const [refreshFeedback, setRefreshFeedback] = useState<string>('')
    const [isRefreshing, setIsRefreshing] = useState(false)
    const [selectedNode, setSelectedNode] = useState<MarketGnnGraphNode | null>(null)
    const fgRef = useRef<any>(null)

    useEffect(() => {
        let mounted = true

        const loadData = async () => {
            try {
                const [gnnStatus, gnnMetrics, gnnGraph] = await Promise.all([
                    getMarketGnnStatus(),
                    getMarketGnnMetrics(),
                    getMarketGnnGraph(),
                ])
                const refreshPayload = await getMarketGnnRefreshStatus()
                if (!mounted) {
                    return
                }
                setStatusData(gnnStatus)
                setMetricsData(gnnMetrics)
                setGraphData(gnnGraph)
                setRefreshState(refreshPayload.refresh)
            } catch (error) {
                console.error('Chargement du GNN impossible', error)
            } finally {
                if (mounted) {
                    setIsLoading(false)
                }
            }
        }

        loadData()
        const interval = setInterval(loadData, 20000)
        return () => {
            mounted = false
            clearInterval(interval)
        }
    }, [])

    const registry = statusData?.gnn
    const graph = graphData ?? {
        status: 'unavailable',
        reason: 'GNN indisponible',
        nodes: [],
        links: [],
    }
    const graphReadiness = statusData?.graph_readiness
    const graphPayload = useMemo(
        () => ({
            nodes: graph.nodes ?? [],
            links: graph.links ?? [],
        }),
        [graph.nodes, graph.links],
    )

    const handleRefresh = async () => {
        setIsRefreshing(true)
        setRefreshFeedback('')
        const response = await requestMarketGnnRefresh()
        setRefreshState(response.refresh)
        setRefreshFeedback(response.message || (response.status === 'started' ? 'Refresh GNN demarre.' : 'Refresh GNN planifie.'))
        const [gnnStatus, gnnMetrics, gnnGraph] = await Promise.all([
            getMarketGnnStatus(),
            getMarketGnnMetrics(),
            getMarketGnnGraph(),
        ])
        setStatusData(gnnStatus)
        setMetricsData(gnnMetrics)
        setGraphData(gnnGraph)
        setIsRefreshing(false)
    }

    if (isLoading) {
        return (
            <div className="flex h-full items-center justify-center">
                <div className="flex flex-col items-center gap-4">
                    <Loader2 size={44} className="animate-spin text-emerald-400" />
                    <div className="text-[11px] font-black uppercase tracking-[0.28em] text-emerald-400/70">
                        Chargement du Market GNN
                    </div>
                </div>
            </div>
        )
    }

    return (
        <div className="grid h-full grid-cols-1 gap-6 xl:grid-cols-[1.1fr_0.9fr]">
            <div className="glass rounded-[2rem] border border-white/5 p-6 shadow-2xl">
                <div className="mb-6 flex items-start justify-between gap-4">
                    <div>
                        <div className="flex items-center gap-3">
                            <Network size={18} className="text-emerald-400" />
                            <h3 className="text-sm font-black uppercase tracking-[0.28em] text-white">
                                Market GNN
                            </h3>
                            <span className={`rounded-xl border px-3 py-1 text-[9px] font-black uppercase tracking-[0.2em] ${statusTone(registry?.status)}`}>
                                {registry?.status || 'unavailable'}
                            </span>
                        </div>
                        <p className="mt-3 text-[11px] text-slate-400">
                            Vue honnête du GNN de marché: version, métriques réelles, univers d'entraînement et graphe dérivé des historiques.
                        </p>
                    </div>
                    <button
                        onClick={() => fgRef.current?.zoomToFit(400)}
                        className="rounded-2xl border border-white/10 bg-white/[0.04] p-3 text-slate-400 transition hover:border-emerald-500/30 hover:text-emerald-300"
                    >
                        <Target size={18} />
                    </button>
                </div>

                <div className="mb-4 flex flex-wrap items-center gap-3">
                    <button
                        onClick={handleRefresh}
                        disabled={isRefreshing}
                        className="rounded-2xl border border-emerald-500/20 bg-emerald-500/10 px-4 py-2 text-[10px] font-black uppercase tracking-[0.2em] text-emerald-300 disabled:opacity-40"
                    >
                        {isRefreshing ? 'Refresh en cours...' : 'Refresh GNN'}
                    </button>
                    <div className="text-[10px] text-slate-400">
                        {refreshState?.queued
                            ? 'Refresh en file d attente jusqu a liberation du GPU.'
                            : registry?.status_reason || 'Aucun diagnostic GNN detaille.'}
                    </div>
                </div>

                <div className="mb-6 grid grid-cols-2 gap-3 xl:grid-cols-4">
                    {infoCard('Version', registry?.version || '--', registry?.checkpoint_path ? registry.checkpoint_path.split(/[\\/]/).pop() : 'Aucun checkpoint')}
                    {infoCard('Dernier entrainement', formatDateTime(registry?.trained_at), registry?.source_run_id || 'Run inconnu')}
                    {infoCard('Univers versionne', String(registry?.universe.count || 0), `${graph.displayed_symbol_count || 0} affiches sur le graphe`)}
                    {infoCard('Timeframe graphe', graph.graph_timeframe || '--', `${graph.correlation_points || 0} points de correlation`)}
                </div>

                <div className="mb-6 grid grid-cols-2 gap-3 xl:grid-cols-4">
                    {infoCard('Derniere demande', formatDateTime(registry?.last_refresh_requested_at || refreshState?.requested_at), refreshState?.status || 'idle')}
                    {infoCard('Dernier depart', formatDateTime(registry?.last_refresh_started_at || refreshState?.started_at), registry?.last_refresh_status || '--')}
                    {infoCard('Derniere fin', formatDateTime(registry?.last_refresh_finished_at || refreshState?.finished_at), refreshState?.failure_reason || 'Aucune erreur')}
                    {infoCard('Readiness graphe', graphReadiness?.status || graph.status || '--', graphReadiness?.reason || graph.reason)}
                </div>

                {refreshFeedback ? (
                    <div className="mb-4 rounded-2xl border border-white/10 bg-white/[0.04] px-4 py-3 text-[11px] text-slate-300">
                        {refreshFeedback}
                    </div>
                ) : null}

                {graph.status === 'ok' && graphPayload.nodes.length > 0 ? (
                    <div className="relative h-[640px] overflow-hidden rounded-[1.75rem] border border-white/5 bg-black/30">
                        {selectedNode ? (
                            <div className="absolute bottom-6 left-6 z-10 w-80 rounded-3xl border border-emerald-500/20 bg-slate-950/90 p-5 shadow-xl backdrop-blur-2xl">
                                <div className="flex items-center gap-3">
                                    <div className={`rounded-xl border px-2 py-1 text-[9px] font-black uppercase tracking-[0.2em] ${selectedNode.role === 'core' ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-300' : familyTone(selectedNode.family || '')}`}>
                                        {selectedNode.role === 'core' ? 'core' : selectedNode.family || 'asset'}
                                    </div>
                                    <div className="text-sm font-black text-white">{selectedNode.label}</div>
                                </div>
                                <div className="mt-4 grid grid-cols-2 gap-2">
                                    <div className="rounded-xl border border-white/8 bg-white/[0.03] px-3 py-2">
                                        <div className="text-[8px] font-black uppercase tracking-[0.18em] text-slate-500">Centralite</div>
                                        <div className="mt-1 text-[11px] font-black text-white/85">
                                            {typeof selectedNode.centrality === 'number' ? selectedNode.centrality.toFixed(4) : '--'}
                                        </div>
                                    </div>
                                    <div className="rounded-xl border border-white/8 bg-white/[0.03] px-3 py-2">
                                        <div className="text-[8px] font-black uppercase tracking-[0.18em] text-slate-500">Horodatage</div>
                                        <div className="mt-1 text-[11px] font-black text-white/85">
                                            {formatDateTime(selectedNode.timestamp)}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        ) : null}

                        <ForceGraph2D
                            ref={fgRef}
                            graphData={graphPayload}
                            backgroundColor="transparent"
                            nodeLabel="label"
                            nodeRelSize={6}
                            onNodeClick={(node) => setSelectedNode(node as MarketGnnGraphNode)}
                            linkWidth={(link: any) => Math.max(1, Number(link.value || 0.5) * 3)}
                            linkColor={(link: any) => {
                                if (link.kind === 'core') {
                                    return 'rgba(16, 185, 129, 0.30)'
                                }
                                return Number(link.correlation || 0) >= 0
                                    ? 'rgba(14, 165, 233, 0.28)'
                                    : 'rgba(244, 63, 94, 0.24)'
                            }}
                            nodeCanvasObject={(node: any, ctx, globalScale) => {
                                const label = String(node.label || '')
                                const fontSize = 12 / globalScale
                                ctx.font = `${fontSize}px Space Grotesk`

                                ctx.beginPath()
                                ctx.arc(node.x, node.y, node.role === 'core' ? 9 : 5, 0, 2 * Math.PI, false)
                                ctx.shadowBlur = node.role === 'core' ? 18 : 10
                                ctx.shadowColor = node.role === 'core' ? '#10b981' : '#38bdf8'
                                ctx.fillStyle = node.role === 'core' ? '#34d399' : '#38bdf8'
                                ctx.fill()
                                ctx.shadowBlur = 0

                                if (globalScale > 1.5) {
                                    const textWidth = ctx.measureText(label).width
                                    ctx.fillStyle = 'rgba(2, 6, 23, 0.82)'
                                    ctx.fillRect(node.x - textWidth / 2 - 4, node.y - 18, textWidth + 8, fontSize + 6)
                                    ctx.textAlign = 'center'
                                    ctx.textBaseline = 'middle'
                                    ctx.fillStyle = node.role === 'core' ? '#a7f3d0' : '#e2e8f0'
                                    ctx.fillText(label, node.x, node.y - 12)
                                }
                            }}
                            cooldownTicks={120}
                            d3VelocityDecay={0.32}
                        />
                    </div>
                ) : (
                    <div className="flex h-[640px] flex-col items-center justify-center rounded-[1.75rem] border border-white/5 bg-black/30 text-center">
                        <CircleOff size={36} className="text-rose-400/80" />
                        <div className="mt-4 text-sm font-black uppercase tracking-[0.24em] text-white">
                            Graphe GNN indisponible
                        </div>
                        <p className="mt-3 max-w-xl text-[12px] text-slate-400">
                            {graph.reason || 'Le graphe reel du GNN n est pas encore disponible pour cette version du modele.'}
                        </p>
                        <div className="mt-4 grid grid-cols-1 gap-2 text-left text-[11px] text-slate-400">
                            <div>Timeframes candidats: {(graph.candidate_timeframes || []).join(' -> ') || '--'}</div>
                            <div>Timeframe retenu: {graph.selected_timeframe || '--'}</div>
                            <div>Overlap: {graph.overlap_points || 0}</div>
                            <div>Symboles manquants: {(graph.missing_symbols || []).slice(0, 8).join(', ') || 'Aucun'}</div>
                        </div>
                    </div>
                )}
            </div>

            <div className="flex h-full flex-col gap-6">
                <div className="glass rounded-[2rem] border border-white/5 p-6 shadow-2xl">
                    <div className="mb-4 flex items-center gap-3">
                        <ShieldCheck size={18} className="text-emerald-400" />
                        <div className="text-[10px] font-black uppercase tracking-[0.28em] text-emerald-300">
                            Qualite du GNN
                        </div>
                    </div>
                    <div className="grid grid-cols-2 gap-3">
                        {infoCard('Loss', typeof metricsData?.metrics.loss === 'number' ? metricsData.metrics.loss.toFixed(4) : '--')}
                        {infoCard('Samples', String(metricsData?.metrics.samples || 0))}
                        {infoCard('Accuracy scalp', formatPercent(metricsData?.metrics.scalp_accuracy))}
                        {infoCard('Accuracy intraday', formatPercent(metricsData?.metrics.intraday_accuracy))}
                        {infoCard('Accuracy swing', formatPercent(metricsData?.metrics.swing_accuracy))}
                        {infoCard('Epochs', String(metricsData?.metrics.epochs || 0), `Batch ${metricsData?.metrics.batch_size || 0}`)}
                    </div>
                    <div className="mt-4 rounded-2xl border border-white/8 bg-black/20 px-4 py-3 text-[11px] text-slate-400">
                        {metricsData?.status_reason || registry?.status_reason || 'Aucun diagnostic detaille disponible.'}
                    </div>
                </div>

                <div className="glass rounded-[2rem] border border-white/5 p-6 shadow-2xl">
                    <div className="mb-4 flex items-center gap-3">
                        <Database size={18} className="text-sky-400" />
                        <div className="text-[10px] font-black uppercase tracking-[0.28em] text-sky-300">
                            Univers versionne
                        </div>
                    </div>
                    <div className="grid grid-cols-2 gap-2">
                        {Object.entries(registry?.universe.family_counts || {}).map(([family, count]) => (
                            <div key={family} className={`rounded-xl border px-3 py-2 text-[10px] font-black uppercase tracking-[0.18em] ${familyTone(family)}`}>
                                {family} | {count}
                            </div>
                        ))}
                    </div>
                    <div className="mt-4">
                        <div className="text-[9px] font-black uppercase tracking-[0.2em] text-slate-500">Timeframes</div>
                        <div className="mt-2 flex flex-wrap gap-2">
                            {(registry?.timeframes || []).map((timeframe) => (
                                <span
                                    key={timeframe}
                                    className="rounded-xl border border-white/10 bg-white/[0.04] px-3 py-1 text-[9px] font-black uppercase tracking-[0.18em] text-white/75"
                                >
                                    {timeframe}
                                </span>
                            ))}
                        </div>
                    </div>
                    <div className="mt-4">
                        <div className="text-[9px] font-black uppercase tracking-[0.2em] text-slate-500">Echantillon de symboles</div>
                        <div className="mt-3 flex flex-wrap gap-2">
                            {(registry?.universe.symbols || []).slice(0, 16).map((symbol) => (
                                <span
                                    key={symbol}
                                    className={`rounded-xl border px-3 py-1 text-[9px] font-black uppercase tracking-[0.18em] ${familyTone(symbol.includes('.cash') ? 'index_cfd' : '')}`}
                                >
                                    {symbol}
                                </span>
                            ))}
                        </div>
                    </div>
                    <div className="mt-4 rounded-2xl border border-white/8 bg-black/20 px-4 py-3">
                        <div className="text-[9px] font-black uppercase tracking-[0.2em] text-slate-500">Couverture</div>
                        <div className="mt-2 text-[11px] text-white/75">
                            {(registry?.coverage_summary?.graph_reason as string | undefined) || 'Couverture non calculee.'}
                        </div>
                        <div className="mt-2 text-[10px] text-slate-500">
                            Timeframe: {(registry?.coverage_summary?.selected_timeframe as string | undefined) || '--'} | Overlap: {(registry?.coverage_summary?.overlap_points as number | undefined) || 0}
                        </div>
                    </div>
                </div>

                <div className="glass rounded-[2rem] border border-white/5 p-6 shadow-2xl">
                    <div className="mb-4 flex items-center gap-3">
                        <GitBranch size={18} className="text-violet-400" />
                        <div className="text-[10px] font-black uppercase tracking-[0.28em] text-violet-300">
                            Artefacts
                        </div>
                    </div>
                    <div className="space-y-3">
                        <div className="rounded-2xl border border-white/8 bg-black/20 px-4 py-3">
                            <div className="flex items-center gap-2 text-[9px] font-black uppercase tracking-[0.2em] text-slate-500">
                                <Clock3 size={12} />
                                Checkpoint
                            </div>
                            <div className="mt-2 text-[11px] font-black text-white/85">
                                {registry?.artifacts.checkpoint.path || 'Aucun checkpoint'}
                            </div>
                        </div>
                        <div className="rounded-2xl border border-white/8 bg-black/20 px-4 py-3">
                            <div className="flex items-center gap-2 text-[9px] font-black uppercase tracking-[0.2em] text-slate-500">
                                <Activity size={12} />
                                Metriques
                            </div>
                            <div className="mt-2 text-[11px] font-black text-white/85">
                                {registry?.artifacts.metrics.path || 'Aucun rapport'}
                            </div>
                        </div>
                        <div className="rounded-2xl border border-white/8 bg-black/20 px-4 py-3">
                            <div className="text-[9px] font-black uppercase tracking-[0.2em] text-slate-500">Registre</div>
                            <div className="mt-2 text-[11px] font-black text-white/85">
                                {registry?.artifacts.registry.path || 'Aucun registre'}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}
