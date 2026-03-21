import { useCallback, useEffect, useMemo, useState } from 'react'
import {
    Brain,
    CheckCircle2,
    Clock3,
    ExternalLink,
    Globe2,
    RefreshCw,
    Search,
    ShieldCheck,
    Sparkles,
    Workflow,
} from 'lucide-react'

import {
    approveResearchReviewItem,
    getApprovedKnowledge,
    getMemoryGraph,
    getResearchIngestStatus,
    getResearchReviewQueue,
    getResearchSources,
    getResearcherHistory,
    getResearcherTrends,
    rejectResearchReviewItem,
    searchCoreMemory,
    searchResearchPapers,
    searchResearcher,
    syncResearchSources,
    type KnowledgeReviewItem,
    type IngestionStatusResponse,
    type MemorySearchResult,
    type ResearchHistoryEntry,
    type ResearchPaper,
    type ResearchResult,
    type ResearchTrendSource,
} from '../services/api'
import { navigateToHiveTab } from '../navigation'

type SearchMode = 'web' | 'arxiv'

type SearchState = {
    mode: SearchMode
    query: string
    domain: string
    synthesis: string
    results: ResearchResult[]
    papers: ResearchPaper[]
}

const INITIAL_SEARCH_STATE: SearchState = {
    mode: 'web',
    query: '',
    domain: 'tech',
    synthesis: '',
    results: [],
    papers: [],
}

function toDisplayString(value: unknown, fallback = '') {
    if (typeof value === 'string') {
        return value
    }
    if (typeof value === 'number' || typeof value === 'boolean') {
        return String(value)
    }
    if (value && typeof value === 'object') {
        try {
            return JSON.stringify(value)
        } catch {
            return fallback
        }
    }
    return fallback
}

function normalizeStringList(value: unknown) {
    if (Array.isArray(value)) {
        return value
            .map((entry) => toDisplayString(entry).trim())
            .filter(Boolean)
    }
    if (typeof value === 'string') {
        return value
            .split(',')
            .map((entry) => entry.trim())
            .filter(Boolean)
    }
    return []
}

function formatDate(value?: unknown) {
    if (!value) {
        return 'indisponible'
    }
    const date = new Date(toDisplayString(value))
    if (Number.isNaN(date.getTime())) {
        return 'indisponible'
    }
    return date.toLocaleString('fr-FR', {
        day: '2-digit',
        month: '2-digit',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
    })
}

function truncate(value: unknown, limit = 220) {
    const text = toDisplayString(value)
    if (!text) {
        return ''
    }
    return text.length > limit ? `${text.slice(0, limit - 3)}...` : text
}

function buildEntityKey(...parts: unknown[]) {
    return parts
        .map((part) => toDisplayString(part).trim())
        .filter(Boolean)
        .join('-')
}

function formatLogEntry(value: unknown) {
    if (typeof value === 'string') {
        return value
    }
    if (!value || typeof value !== 'object') {
        return toDisplayString(value)
    }
    const record = value as Record<string, unknown>
    const timestamp = toDisplayString(record.ts)
    const level = toDisplayString(record.level).toUpperCase()
    const message = toDisplayString(record.message)
    const chunks = [timestamp, level, message].filter(Boolean)
    if (chunks.length > 0) {
        return chunks.join(' | ')
    }
    return toDisplayString(record, 'Entree de log indisponible.')
}

function MetricTile({ label, value, tone = 'matrix' }: { label: string; value: string | number; tone?: 'matrix' | 'cyan' | 'amber' | 'pink' }) {
    const toneClass = {
        matrix: 'text-matrix',
        cyan: 'text-cyber-cyan',
        amber: 'text-cyber-amber',
        pink: 'text-cyber-pink',
    }[tone]

    return (
        <div className="cyber-panel p-4">
            <div className="text-[8px] uppercase tracking-[0.2em] text-white/20 mb-1">{label}</div>
            <div className={`text-xl font-bold ${toneClass}`}>{value}</div>
        </div>
    )
}

function dependencyTone(status: string) {
    if (status === 'ok') {
        return 'text-matrix border-matrix/20 bg-matrix/10'
    }
    if (status === 'error') {
        return 'text-cyber-pink border-cyber-pink/20 bg-cyber-pink/10'
    }
    return 'text-cyber-amber border-cyber-amber/20 bg-cyber-amber/10'
}

function SourceBadge({ source }: { source: ResearchTrendSource }) {
    const family = toDisplayString(source.family || source.source_type, 'source')
    const sourceName = toDisplayString(source.source_name, 'source')
    const url = toDisplayString(source.url)
    return (
        <div className="flex items-center justify-between gap-3 p-3 border border-white/[0.05] bg-white/[0.02]">
            <div>
                <div className="text-[10px] font-bold text-white/70">{sourceName}</div>
                <div className="text-[8px] text-white/25 uppercase tracking-[0.15em]">{family}</div>
            </div>
            <div className="text-right text-[8px] text-white/25">
                <div>Sync: {formatDate(source.last_sync)}</div>
                <div>Queue: {source.queued || 0} | Doublons: {source.duplicates || 0}</div>
                {url && <div className="truncate max-w-[180px]">{url}</div>}
            </div>
        </div>
    )
}

function ReviewCard({
    item,
    busy,
    onApprove,
    onReject,
}: {
    item: KnowledgeReviewItem
    busy: boolean
    onApprove: (itemId: string) => Promise<void>
    onReject: (itemId: string) => Promise<void>
}) {
    const tagList = normalizeStringList(item.tags).slice(0, 6)
    const itemFamily = toDisplayString(item.family, 'source')
    const sourceName = toDisplayString(item.source_name, 'source')
    const title = toDisplayString(item.title, 'Sans titre')
    const url = toDisplayString(item.url)
    return (
        <div className="cyber-panel hud-corners p-4 space-y-3">
            <div className="flex items-start justify-between gap-4">
                <div>
                    <div className="text-[8px] text-cyber-cyan/60 uppercase tracking-[0.2em]">
                        {itemFamily} | {sourceName}
                    </div>
                    <h4 className="text-[12px] font-bold text-white/80 mt-1">{title}</h4>
                </div>
                <div className="text-right text-[8px] text-white/25 shrink-0">
                    <div>Score {Math.round((item.confidence_score || 0) * 100)}%</div>
                    <div>{formatDate(item.published_at || item.collected_at)}</div>
                </div>
            </div>

            <p className="text-[10px] text-white/40 leading-relaxed">
                {truncate(item.summary_curated || item.summary_raw || 'Aucun resume disponible.', 280)}
            </p>

            <div className="flex flex-wrap gap-1.5">
                {tagList.map((tag) => (
                    <span key={buildEntityKey(item.id, tag)} className="px-2 py-0.5 text-[8px] border border-matrix/10 bg-matrix/[0.03] text-matrix/40 uppercase tracking-[0.15em]">
                        {tag}
                    </span>
                ))}
            </div>

            <div className="flex items-center justify-between gap-3 pt-2 border-t border-white/[0.04]">
                <a
                    href={url || '#'}
                    target="_blank"
                    rel="noreferrer"
                    className={`text-[9px] inline-flex items-center gap-1 ${url ? 'text-cyber-cyan/60 hover:text-cyber-cyan' : 'text-white/20 pointer-events-none'}`}
                >
                    <ExternalLink size={12} />
                    <span>Source</span>
                </a>
                <div className="flex gap-2">
                    <button
                        onClick={() => onReject(item.id)}
                        disabled={busy}
                        className="px-3 py-1.5 text-[9px] uppercase tracking-[0.15em] border border-cyber-pink/20 bg-cyber-pink/10 text-cyber-pink/70 disabled:opacity-40"
                    >
                        Rejeter
                    </button>
                    <button
                        onClick={() => onApprove(item.id)}
                        disabled={busy}
                        className="px-3 py-1.5 text-[9px] uppercase tracking-[0.15em] border border-matrix/20 bg-matrix/10 text-matrix disabled:opacity-40"
                    >
                        Valider
                    </button>
                </div>
            </div>
        </div>
    )
}

export default function KnowledgeVault() {
    const [ingestStatus, setIngestStatus] = useState<IngestionStatusResponse | null>(null)
    const [sources, setSources] = useState<ResearchTrendSource[]>([])
    const [pendingItems, setPendingItems] = useState<KnowledgeReviewItem[]>([])
    const [approvedItems, setApprovedItems] = useState<KnowledgeReviewItem[]>([])
    const [history, setHistory] = useState<ResearchHistoryEntry[]>([])
    const [trends, setTrends] = useState<ResearchTrendSource[]>([])
    const [memoryGraph, setMemoryGraph] = useState<{ nodes: Array<{ id: string }>; links: Array<{ source: string; target: string }> }>({ nodes: [], links: [] })
    const [memoryResults, setMemoryResults] = useState<MemorySearchResult[]>([])
    const [searchState, setSearchState] = useState<SearchState>(INITIAL_SEARCH_STATE)
    const [memoryQuery, setMemoryQuery] = useState('')
    const [busyAction, setBusyAction] = useState<string | null>(null)
    const [feedback, setFeedback] = useState<string>('')

    const loadKnowledgeState = useCallback(async () => {
        const [status, sourcePayload, pendingPayload, approvedPayload, historyPayload, trendPayload, graphPayload] = await Promise.all([
            getResearchIngestStatus(12),
            getResearchSources(),
            getResearchReviewQueue('pending', 12, 0),
            getApprovedKnowledge(12),
            getResearcherHistory(12),
            getResearcherTrends(searchState.domain),
            getMemoryGraph(24, 0.86),
        ])
        setIngestStatus(status)
        setSources(sourcePayload.sources || [])
        setPendingItems(pendingPayload.items || [])
        setApprovedItems(approvedPayload.items || [])
        setHistory(historyPayload.history || [])
        setTrends(trendPayload.ingest_sources || [])
        setMemoryGraph(graphPayload)
    }, [searchState.domain])

    useEffect(() => {
        void loadKnowledgeState()
        const interval = setInterval(() => {
            void loadKnowledgeState()
        }, 15000)
        return () => clearInterval(interval)
    }, [loadKnowledgeState])

    const handleSync = async () => {
        setBusyAction('sync')
        setFeedback('')
        const result = await syncResearchSources(true, true, 5)
        setFeedback(result.status === 'ok' ? 'Synchronisation des sources lancee.' : 'Synchronisation indisponible.')
        await loadKnowledgeState()
        setBusyAction(null)
    }

    const handleApprove = async (itemId: string) => {
        setBusyAction(`approve:${itemId}`)
        setFeedback('')
        const result = await approveResearchReviewItem(itemId)
        setFeedback(result.status === 'ok' ? `Candidat ${itemId} valide et ingere.` : `Validation impossible pour ${itemId}.`)
        await loadKnowledgeState()
        setBusyAction(null)
    }

    const handleReject = async (itemId: string) => {
        setBusyAction(`reject:${itemId}`)
        setFeedback('')
        const result = await rejectResearchReviewItem(itemId, 'Rejet manuel depuis Nexus')
        setFeedback(result.status === 'ok' ? `Candidat ${itemId} rejete.` : `Rejet impossible pour ${itemId}.`)
        await loadKnowledgeState()
        setBusyAction(null)
    }

    const handleResearchSearch = async () => {
        if (!searchState.query.trim()) {
            return
        }
        setBusyAction('search')
        setFeedback('')
        if (searchState.mode === 'arxiv') {
            const response = await searchResearchPapers(searchState.query, 'cs.AI', 5)
            setSearchState((current) => ({
                ...current,
                papers: response.papers,
                results: [],
                synthesis: `Papiers recuperes: ${response.total}. File de revue: ${response.review_queue?.queued || 0} ajouts.`,
            }))
        } else {
            const response = await searchResearcher(searchState.query, searchState.domain, 5)
            setSearchState((current) => ({
                ...current,
                results: response.results,
                papers: [],
                synthesis: response.synthesis,
            }))
        }
        await loadKnowledgeState()
        setBusyAction(null)
    }

    const handleMemorySearch = async () => {
        if (!memoryQuery.trim()) {
            setMemoryResults([])
            return
        }
        setBusyAction('memory')
        setMemoryResults(await searchCoreMemory(memoryQuery, 8))
        setBusyAction(null)
    }

    const topSources = useMemo(() => (Array.isArray(sources) ? sources : []).slice(0, 4), [sources])
    const trendSources = useMemo(() => (Array.isArray(trends) ? trends : []).slice(0, 4), [trends])
    const pipelineLogs = useMemo(
        () =>
            (Array.isArray(ingestStatus?.logs) ? ingestStatus.logs : [])
                .slice(-8)
                .map(formatLogEntry)
                .filter(Boolean),
        [ingestStatus],
    )

    return (
        <div className="h-full overflow-y-auto p-4 space-y-4 animate-fade-in">
            <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-3">
                <MetricTile label="En revue" value={ingestStatus?.counts?.pending ?? 0} tone="amber" />
                <MetricTile label="Valides" value={ingestStatus?.counts?.approved ?? 0} tone="matrix" />
                <MetricTile label="Rejetes" value={ingestStatus?.counts?.rejected ?? 0} tone="pink" />
                <MetricTile label="Ingeres" value={ingestStatus?.counts?.ingested ?? 0} tone="matrix" />
                <MetricTile label="Sources" value={sources.length} tone="cyan" />
                <MetricTile label="Doublons" value={`${Math.round((ingestStatus?.duplicate_rate || 0) * 100)}%`} tone="pink" />
            </div>

            <div className="grid grid-cols-1 xl:grid-cols-[1.15fr_0.85fr] gap-4">
                <section className="cyber-panel hud-corners p-4 lg:p-5 space-y-4">
                    <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
                        <div>
                            <div className="text-[9px] uppercase tracking-[0.2em] text-matrix/40">Controle connaissance</div>
                            <h3 className="font-display text-lg font-black tracking-[0.08em] text-white/80 mt-1">Researcher et Memory Bridge</h3>
                            <p className="text-[10px] text-white/25 mt-2 max-w-3xl">
                                La collecte academique et actualite passe par une revue obligatoire avant ingestion durable dans la memoire.
                            </p>
                        </div>
                        <button
                            onClick={handleSync}
                            disabled={busyAction !== null}
                            className="cyber-btn text-[9px] px-3 py-2 disabled:opacity-40"
                        >
                            <RefreshCw size={12} className={busyAction === 'sync' ? 'animate-spin' : ''} />
                            <span>Synchroniser les sources</span>
                        </button>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                        <button
                            onClick={() => navigateToHiveTab('memory')}
                            className="border border-white/[0.05] bg-white/[0.02] p-3 text-left transition-colors hover:border-matrix/20 hover:bg-matrix/[0.03]"
                        >
                            <div className="text-[9px] uppercase tracking-[0.18em] text-matrix/60">Voir en memoire</div>
                            <div className="mt-2 text-[10px] text-white/30 leading-relaxed">
                                Basculer vers Memory Store pour verifier les fragments, les scores RAG et le graphe de connaissance.
                            </div>
                        </button>
                        <button
                            onClick={() => navigateToHiveTab('graph')}
                            className="border border-white/[0.05] bg-white/[0.02] p-3 text-left transition-colors hover:border-cyber-cyan/20 hover:bg-cyber-cyan/[0.03]"
                        >
                            <div className="text-[9px] uppercase tracking-[0.18em] text-cyber-cyan/60">Voir dans le graphe</div>
                            <div className="mt-2 text-[10px] text-white/30 leading-relaxed">
                                Ouvrir Nexus Graph pour separer le graphe GNN de marche du graphe memoire de la ruche.
                            </div>
                        </button>
                        <div className="border border-white/[0.05] bg-white/[0.02] p-3">
                            <div className="flex items-center gap-2 text-[9px] uppercase tracking-[0.18em] text-cyber-amber/60">
                                <Workflow size={12} />
                                <span>Pipeline</span>
                            </div>
                            <div className="mt-2 text-[10px] text-white/30 leading-relaxed">
                                {'Recherche -> revue -> validation -> ingestion -> memoire vectorielle et graphe relationnel.'}
                            </div>
                        </div>
                    </div>

                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                        <div className="border border-white/[0.05] bg-black/30 p-4 space-y-3">
                            <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.2em] text-cyber-cyan/60">
                                <Globe2 size={14} />
                                <span>Recherche a la demande</span>
                            </div>
                            <div className="flex gap-2">
                                <button
                                    onClick={() => setSearchState((current) => ({ ...current, mode: 'web' }))}
                                    className={`px-3 py-1.5 text-[9px] uppercase tracking-[0.15em] border ${searchState.mode === 'web' ? 'border-cyber-cyan/30 bg-cyber-cyan/10 text-cyber-cyan' : 'border-white/10 text-white/30'}`}
                                >
                                    Web
                                </button>
                                <button
                                    onClick={() => setSearchState((current) => ({ ...current, mode: 'arxiv' }))}
                                    className={`px-3 py-1.5 text-[9px] uppercase tracking-[0.15em] border ${searchState.mode === 'arxiv' ? 'border-cyber-cyan/30 bg-cyber-cyan/10 text-cyber-cyan' : 'border-white/10 text-white/30'}`}
                                >
                                    arXiv
                                </button>
                            </div>
                            <input
                                value={searchState.query}
                                onChange={(event) => setSearchState((current) => ({ ...current, query: event.target.value }))}
                                onKeyDown={(event) => event.key === 'Enter' && void handleResearchSearch()}
                                placeholder={searchState.mode === 'web' ? 'Prompt de recherche ou sujet de veille...' : 'Sujet de paper, modele, methode...'}
                                className="w-full bg-black/50 border border-white/10 px-3 py-2 text-[12px] text-white/80 outline-none focus:border-cyber-cyan/40"
                            />
                            <div className="flex gap-2">
                                <select
                                    value={searchState.domain}
                                    onChange={(event) => setSearchState((current) => ({ ...current, domain: event.target.value }))}
                                    className="bg-black/50 border border-white/10 px-3 py-2 text-[11px] text-white/70 outline-none"
                                >
                                    <option value="tech">Tech</option>
                                    <option value="finance">Finance</option>
                                    <option value="science">Science</option>
                                    <option value="crypto">Crypto</option>
                                </select>
                                <button
                                    onClick={handleResearchSearch}
                                    disabled={busyAction !== null}
                                    className="cyber-btn text-[9px] px-3 py-2 disabled:opacity-40"
                                >
                                    <Search size={12} className={busyAction === 'search' ? 'animate-pulse' : ''} />
                                    <span>Explorer</span>
                                </button>
                            </div>
                            <div className="border border-white/[0.05] bg-white/[0.02] p-3 min-h-[120px]">
                                <div className="text-[8px] uppercase tracking-[0.2em] text-white/20 mb-2">Synthese / retour</div>
                                <div className="text-[10px] text-white/40 whitespace-pre-wrap leading-relaxed">
                                    {searchState.synthesis || 'Aucune recherche lancee pour le moment.'}
                                </div>
                            </div>
                            <div className="space-y-2 max-h-[260px] overflow-y-auto custom-scrollbar pr-1">
                                {searchState.mode === 'arxiv'
                                    ? searchState.papers.map((paper) => (
                                        <a key={buildEntityKey(paper.url, paper.title)} href={toDisplayString(paper.url, '#')} target="_blank" rel="noreferrer" className="block border border-white/[0.05] bg-white/[0.02] p-3 hover:border-cyber-cyan/20 transition-colors">
                                            <div className="text-[11px] font-bold text-white/70">{toDisplayString(paper.title, 'Sans titre')}</div>
                                            <div className="text-[8px] text-white/20 mt-1">{normalizeStringList(paper.authors).slice(0, 3).join(', ') || 'Auteur indisponible'}</div>
                                            <div className="text-[10px] text-white/35 mt-2 leading-relaxed">{truncate(paper.summary, 180)}</div>
                                        </a>
                                    ))
                                    : searchState.results.map((result) => (
                                        <a key={buildEntityKey(result.url, result.title)} href={toDisplayString(result.url, '#')} target="_blank" rel="noreferrer" className="block border border-white/[0.05] bg-white/[0.02] p-3 hover:border-cyber-cyan/20 transition-colors">
                                            <div className="flex items-center justify-between gap-3">
                                                <div className="text-[11px] font-bold text-white/70">{toDisplayString(result.title, 'Sans titre')}</div>
                                                <span className="text-[8px] text-cyber-cyan/50 uppercase tracking-[0.15em]">{toDisplayString(result.source, 'web')}</span>
                                            </div>
                                            <div className="text-[10px] text-white/35 mt-2 leading-relaxed">{truncate(result.summary, 180)}</div>
                                        </a>
                                    ))}
                            </div>
                        </div>

                        <div className="border border-white/[0.05] bg-black/30 p-4 space-y-3">
                            <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.2em] text-matrix/60">
                                <Brain size={14} />
                                <span>Memoire validee</span>
                            </div>
                            <div className="flex gap-2">
                                <input
                                    value={memoryQuery}
                                    onChange={(event) => setMemoryQuery(event.target.value)}
                                    onKeyDown={(event) => event.key === 'Enter' && void handleMemorySearch()}
                                    placeholder="Rechercher une idee, une strategie ou un auteur..."
                                    className="flex-1 bg-black/50 border border-white/10 px-3 py-2 text-[12px] text-white/80 outline-none focus:border-matrix/40"
                                />
                                <button onClick={handleMemorySearch} disabled={busyAction !== null} className="cyber-btn text-[9px] px-3 py-2 disabled:opacity-40">
                                    <Search size={12} className={busyAction === 'memory' ? 'animate-pulse' : ''} />
                                    <span>RAG</span>
                                </button>
                            </div>
                            <div className="grid grid-cols-2 gap-3">
                                <div className="border border-white/[0.05] bg-white/[0.02] p-3">
                                    <div className="text-[8px] uppercase tracking-[0.15em] text-white/20">Noeuds Qdrant/Neo4j</div>
                                    <div className="text-lg font-bold text-matrix mt-1">{memoryGraph.nodes.length}</div>
                                </div>
                                <div className="border border-white/[0.05] bg-white/[0.02] p-3">
                                    <div className="text-[8px] uppercase tracking-[0.15em] text-white/20">Relations</div>
                                    <div className="text-lg font-bold text-cyber-cyan mt-1">{memoryGraph.links.length}</div>
                                </div>
                            </div>
                            <div className="space-y-2 max-h-[312px] overflow-y-auto custom-scrollbar pr-1">
                                {memoryResults.length === 0 ? (
                                    <div className="border border-dashed border-white/10 p-4 text-[10px] text-white/25">
                                        Lance une recherche pour verifier ce qui est deja ingere dans la memoire durable.
                                    </div>
                                ) : (
                                    memoryResults.map((result) => (
                                        <div key={`${result.id}-${result.timestamp}`} className="border border-white/[0.05] bg-white/[0.02] p-3">
                                            <div className="flex items-center justify-between gap-3">
                                                <span className="text-[8px] uppercase tracking-[0.15em] text-matrix/50">{result.role || 'memoire'}</span>
                                                <span className="text-[8px] text-white/20">Score {Math.round(result.score * 100)}%</span>
                                            </div>
                                            <div className="text-[10px] text-white/40 mt-2 leading-relaxed">{truncate(result.content, 220)}</div>
                                            <div className="text-[8px] text-white/20 mt-2">{formatDate(result.timestamp)}</div>
                                        </div>
                                    ))
                                )}
                            </div>
                        </div>
                    </div>
                </section>

                <section className="space-y-4">
                    <div className="cyber-panel hud-corners p-4 space-y-3">
                        <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.2em] text-cyber-amber/60">
                            <Clock3 size={14} />
                            <span>File de revue</span>
                        </div>
                        <div className="text-[10px] text-white/25">
                            Run: {ingestStatus?.active_run?.status || 'idle'} | Derniere sync: {formatDate(String(((ingestStatus?.last_run as Record<string, unknown> | null)?.finished_at as string | undefined) || ingestStatus?.active_run?.updated_at || ''))}
                        </div>
                        <div className="space-y-3 max-h-[520px] overflow-y-auto custom-scrollbar pr-1">
                            {pendingItems.length === 0 ? (
                                <div className="border border-dashed border-white/10 p-4 text-[10px] text-white/25">
                                    Aucun candidat en attente de revue.
                                </div>
                            ) : (
                                pendingItems.map((item) => (
                                    <ReviewCard
                                        key={item.id}
                                        item={item}
                                        busy={busyAction === `approve:${item.id}` || busyAction === `reject:${item.id}`}
                                        onApprove={handleApprove}
                                        onReject={handleReject}
                                    />
                                ))
                            )}
                        </div>
                    </div>

                    <div className="cyber-panel hud-corners p-4 space-y-3">
                        <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.2em] text-white/50">
                            <ShieldCheck size={14} />
                            <span>Sources et provenance</span>
                        </div>
                            <div className="space-y-2">
                                {topSources.map((source) => (
                                    <SourceBadge key={buildEntityKey(source.source_name, source.url || source.key || 'source')} source={source} />
                                ))}
                            </div>
                        <div className="border border-white/[0.05] bg-white/[0.02] p-3">
                            <div className="text-[8px] uppercase tracking-[0.15em] text-white/20 mb-2">Veille par domaine</div>
                            <div className="space-y-1">
                                {trendSources.map((source) => (
                                    <div key={buildEntityKey(source.source_name, source.url || source.key || 'trend')} className="flex items-center justify-between text-[9px] text-white/35">
                                        <span>{toDisplayString(source.source_name, 'source')}</span>
                                        <span>{toDisplayString(source.family || source.source_type, 'source')}</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>

                    <div className="cyber-panel hud-corners p-4 space-y-3">
                        <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.2em] text-cyber-cyan/60">
                            <Workflow size={14} />
                            <span>Dependances et journal</span>
                        </div>
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                            {Object.keys(ingestStatus?.dependencies || {}).length === 0 ? (
                                <div className="md:col-span-3 border border-dashed border-white/10 p-4 text-[10px] text-white/25">
                                    Etat des dependances indisponible.
                                </div>
                            ) : (
                                Object.entries(ingestStatus?.dependencies || {}).map(([dependencyName, dependency]) => (
                                    <div key={dependencyName} className="border border-white/[0.05] bg-white/[0.02] p-3">
                                        <div className="flex items-center justify-between gap-3">
                                            <div className="text-[10px] font-bold text-white/65 uppercase tracking-[0.15em]">{dependencyName}</div>
                                            <span className={`px-2 py-0.5 text-[8px] uppercase tracking-[0.18em] border ${dependencyTone(dependency.status)}`}>
                                                {dependency.status}
                                            </span>
                                        </div>
                                        <div className="text-[9px] text-white/25 mt-2 break-words">{dependency.detail || 'Aucun detail'}</div>
                                    </div>
                                ))
                            )}
                        </div>
                        <div className="border border-white/[0.05] bg-black/30 p-3 space-y-2">
                            <div className="text-[8px] uppercase tracking-[0.15em] text-white/20">Logs de pipeline</div>
                            <div className="space-y-1 max-h-[160px] overflow-y-auto custom-scrollbar pr-1">
                                {pipelineLogs.length === 0 ? (
                                    <div className="text-[10px] text-white/25">Aucun log recent disponible.</div>
                                ) : (
                                    pipelineLogs.map((line, index) => (
                                        <div key={buildEntityKey(index, line.slice(0, 24))} className="text-[10px] text-white/35 font-mono break-words">
                                            {line}
                                        </div>
                                    ))
                                )}
                            </div>
                        </div>
                    </div>
                </section>
            </div>

            <div className="grid grid-cols-1 xl:grid-cols-[0.9fr_1.1fr] gap-4">
                <section className="cyber-panel hud-corners p-4 space-y-3">
                    <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.2em] text-cyber-cyan/60">
                        <Sparkles size={14} />
                        <span>Historique recherche</span>
                    </div>
                    <div className="space-y-2 max-h-[240px] overflow-y-auto custom-scrollbar pr-1">
                        {history.length === 0 ? (
                            <div className="border border-dashed border-white/10 p-4 text-[10px] text-white/25">
                                Aucun historique disponible.
                            </div>
                        ) : (
                            history.map((entry) => (
                                <div key={buildEntityKey(entry.query, entry.timestamp)} className="border border-white/[0.05] bg-white/[0.02] p-3">
                                    <div className="text-[11px] font-bold text-white/70">{toDisplayString(entry.query, 'Sans requete')}</div>
                                    <div className="text-[9px] text-white/25 mt-1">{entry.results_count} resultats | {formatDate(entry.timestamp)}</div>
                                </div>
                            ))
                        )}
                    </div>
                </section>

                <section className="cyber-panel hud-corners p-4 space-y-3">
                    <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.2em] text-matrix/60">
                        <CheckCircle2 size={14} />
                        <span>Connaissance approuvee</span>
                    </div>
                    <div className="space-y-2 max-h-[240px] overflow-y-auto custom-scrollbar pr-1">
                        {approvedItems.length === 0 ? (
                            <div className="border border-dashed border-white/10 p-4 text-[10px] text-white/25">
                                Rien d'ingere pour le moment.
                            </div>
                        ) : (
                            approvedItems.map((item) => (
                                <div key={item.id} className="border border-white/[0.05] bg-white/[0.02] p-3">
                                    <div className="flex items-center justify-between gap-3">
                                        <div className="text-[11px] font-bold text-white/70">{toDisplayString(item.title, 'Sans titre')}</div>
                                        <span className="text-[8px] uppercase tracking-[0.15em] text-matrix/50">{toDisplayString(item.family, 'source')}</span>
                                    </div>
                                    <div className="text-[9px] text-white/25 mt-1">{toDisplayString(item.source_name, 'source')} | {formatDate(item.ingested_at || item.reviewed_at || item.collected_at)}</div>
                                    <div className="text-[10px] text-white/35 mt-2 leading-relaxed">{truncate(item.summary_curated || item.summary_raw || '', 180)}</div>
                                </div>
                            ))
                        )}
                    </div>
                </section>
            </div>

            {feedback && (
                <div className="cyber-panel p-4 text-[10px] text-white/45 border border-white/[0.05]">
                    {feedback}
                </div>
            )}
        </div>
    )
}
