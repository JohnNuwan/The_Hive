import { useCallback, useEffect, useMemo, useState } from 'react'
import { Binary, Bot, ChevronRight, Database, Network, Search, Sparkles, User } from 'lucide-react'

import {
    getApprovedKnowledge,
    getMemoryFragments,
    getMemoryGraph,
    getResearchIngestStatus,
    getResearchSources,
    searchCoreMemory,
    type IngestionStatusResponse,
    type KnowledgeReviewItem,
    type MemoryFragment,
    type MemoryGraphLink,
    type MemoryGraphNode,
    type MemorySearchResult,
    type ResearchTrendSource,
} from '../services/api'
import { navigateToHiveTab } from '../navigation'

function formatDate(value?: string | null) {
    if (!value) {
        return 'indisponible'
    }
    const date = new Date(value)
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

function truncate(value: string, limit = 180) {
    if (!value) {
        return ''
    }
    return value.length > limit ? `${value.slice(0, limit - 3)}...` : value
}

function SummaryCard({ label, value, tone }: { label: string; value: string | number; tone: 'matrix' | 'cyan' | 'amber' }) {
    const toneClass = {
        matrix: 'text-matrix',
        cyan: 'text-cyber-cyan',
        amber: 'text-cyber-amber',
    }[tone]

    return (
        <div className="cyber-panel p-4">
            <div className="text-[8px] text-white/20 tracking-[0.2em] uppercase mb-1">{label}</div>
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

export default function MemoryExplorer() {
    const [fragments, setFragments] = useState<MemoryFragment[]>([])
    const [graphNodes, setGraphNodes] = useState<MemoryGraphNode[]>([])
    const [graphLinks, setGraphLinks] = useState<MemoryGraphLink[]>([])
    const [ingestStatus, setIngestStatus] = useState<IngestionStatusResponse | null>(null)
    const [approvedItems, setApprovedItems] = useState<KnowledgeReviewItem[]>([])
    const [sources, setSources] = useState<ResearchTrendSource[]>([])
    const [searchResults, setSearchResults] = useState<MemorySearchResult[]>([])
    const [isLoading, setIsLoading] = useState(true)
    const [searching, setSearching] = useState(false)
    const [searchTerm, setSearchTerm] = useState('')
    const [remoteQuery, setRemoteQuery] = useState('')

    const loadMemory = useCallback(async () => {
        setIsLoading(true)
        const [fragmentPayload, graphPayload, ingestPayload, approvedPayload, sourcePayload] = await Promise.all([
            getMemoryFragments(60),
            getMemoryGraph(30, 0.86),
            getResearchIngestStatus(8),
            getApprovedKnowledge(8),
            getResearchSources(),
        ])
        setFragments(fragmentPayload)
        setGraphNodes(graphPayload.nodes || [])
        setGraphLinks(graphPayload.links || [])
        setIngestStatus(ingestPayload)
        setApprovedItems(approvedPayload.items || [])
        setSources(sourcePayload.sources || [])
        setIsLoading(false)
    }, [])

    useEffect(() => {
        void loadMemory()
    }, [loadMemory])

    const filteredFragments = useMemo(() => {
        const needle = searchTerm.trim().toLowerCase()
        if (!needle) {
            return fragments
        }
        return fragments.filter((fragment) =>
            fragment.content.toLowerCase().includes(needle) || fragment.role.toLowerCase().includes(needle),
        )
    }, [fragments, searchTerm])

    const handleMemorySearch = async () => {
        if (!remoteQuery.trim()) {
            setSearchResults([])
            return
        }
        setSearching(true)
        setSearchResults(await searchCoreMemory(remoteQuery, 10))
        setSearching(false)
    }

    const topSources = useMemo(() => sources.slice(0, 4), [sources])
    const dependencyEntries = useMemo(() => Object.entries(ingestStatus?.dependencies || {}), [ingestStatus])
    const activeSourceLabel = ingestStatus?.active_run?.active
        ? ingestStatus?.active_run?.current_source || ingestStatus?.active_run?.status || 'actif'
        : 'aucun run actif'

    return (
        <div className="flex flex-col h-full gap-6 animate-fade-in">
            <div className="grid grid-cols-2 lg:grid-cols-6 gap-3">
                <SummaryCard label="Fragments" value={fragments.length} tone="matrix" />
                <SummaryCard label="Noeuds graphe" value={graphNodes.length} tone="cyan" />
                <SummaryCard label="Relations" value={graphLinks.length} tone="amber" />
                <SummaryCard label="Valides" value={ingestStatus?.counts?.approved ?? 0} tone="matrix" />
                <SummaryCard label="Ingeres" value={ingestStatus?.counts?.ingested ?? 0} tone="cyan" />
                <SummaryCard label="Resultats RAG" value={searchResults.length} tone="matrix" />
            </div>

            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 glass p-6 rounded-[2rem] border border-white/5">
                <div className="flex items-center gap-4">
                    <div className="w-12 h-12 bg-sky-500/10 border border-sky-500/20 rounded-2xl flex items-center justify-center">
                        <Database className="text-sky-400" size={24} />
                    </div>
                    <div>
                        <h2 className="text-lg font-black text-white uppercase tracking-tighter">Memory Explorer</h2>
                        <p className="text-[10px] text-slate-500 font-bold uppercase tracking-[0.2em]">Qdrant + Neo4j | memoire validee</p>
                        <p className="text-[9px] text-slate-500 mt-2 uppercase tracking-[0.18em]">
                            Pipeline connaissance: {activeSourceLabel}
                        </p>
                    </div>
                </div>

                <div className="relative group flex-grow max-w-md">
                    <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-500 group-focus-within:text-sky-400 transition-colors" size={18} />
                    <input
                        type="text"
                        placeholder="Filtrer les fragments locaux..."
                        value={searchTerm}
                        onChange={(event) => setSearchTerm(event.target.value)}
                        className="w-full bg-white/[0.03] border border-white/10 rounded-2xl py-3 pl-12 pr-4 text-sm outline-none focus:border-sky-500/50 transition-all font-medium"
                    />
                </div>
                <div className="flex gap-2">
                    <button
                        onClick={() => navigateToHiveTab('knowledge')}
                        className="px-4 py-3 text-[9px] uppercase tracking-[0.18em] border border-cyber-cyan/20 bg-cyber-cyan/[0.04] text-cyber-cyan/70 transition-colors hover:bg-cyber-cyan/[0.08]"
                    >
                        Voir en revue
                    </button>
                    <button
                        onClick={() => navigateToHiveTab('graph')}
                        className="px-4 py-3 text-[9px] uppercase tracking-[0.18em] border border-matrix/20 bg-matrix/[0.04] text-matrix/70 transition-colors hover:bg-matrix/[0.08]"
                    >
                        Voir dans le graphe
                    </button>
                </div>
            </div>

            <div className="grid grid-cols-1 xl:grid-cols-[1.1fr_0.9fr] gap-6 flex-grow min-h-0">
                <div className="flex-grow glass rounded-[2.5rem] border border-white/5 overflow-hidden flex flex-col min-h-0">
                    <div className="p-4 border-b border-white/5 bg-white/[0.02] flex items-center justify-between text-[10px] font-black text-slate-500 uppercase tracking-widest px-8">
                        <div className="flex items-center gap-3">
                            <Binary size={14} />
                            <span>Fragments recents</span>
                        </div>
                        <span>{filteredFragments.length} visibles</span>
                    </div>

                    <div className="flex-grow overflow-y-auto p-4 space-y-2 custom-scrollbar">
                        {isLoading ? (
                            <div className="flex items-center justify-center h-full opacity-30 animate-pulse">
                                <Binary size={48} className="text-sky-500" />
                            </div>
                        ) : filteredFragments.length === 0 ? (
                            <div className="flex flex-col items-center justify-center h-full opacity-20 py-20">
                                <Search size={64} className="mb-4" />
                                <p className="font-black uppercase tracking-widest text-xs">Aucun fragment visible</p>
                            </div>
                        ) : (
                            filteredFragments.map((fragment) => (
                                <div
                                    key={fragment.id}
                                    className="flex items-center justify-between p-4 rounded-2xl hover:bg-white/[0.03] border border-transparent hover:border-white/5 transition-all group cursor-pointer"
                                >
                                    <div className="flex items-center gap-8 overflow-hidden">
                                        <div className={`w-10 h-10 rounded-xl flex items-center justify-center shrink-0 ${fragment.role === 'user' ? 'bg-indigo-500/10 text-indigo-400' : 'bg-sky-500/10 text-sky-400'}`}>
                                            {fragment.role === 'user' ? <User size={18} /> : <Bot size={18} />}
                                        </div>
                                        <div className="overflow-hidden">
                                            <p className="text-sm text-slate-300 truncate group-hover:text-white transition-colors">{fragment.content}</p>
                                            <div className="flex items-center gap-3 mt-1">
                                                <span className="text-[9px] font-black text-slate-600 uppercase tracking-widest">ID: {String(fragment.id).slice(0, 8)}</span>
                                                <div className="w-1 h-1 rounded-full bg-slate-800" />
                                                <span className="text-[9px] font-black text-slate-600 uppercase tracking-widest">{fragment.role}</span>
                                            </div>
                                        </div>
                                    </div>
                                    <div className="flex items-center gap-6 shrink-0">
                                        <div className="text-right">
                                            <p className="text-[10px] font-bold text-slate-400">{formatDate(fragment.timestamp)}</p>
                                        </div>
                                        <ChevronRight size={16} className="text-slate-700 group-hover:text-sky-500 transition-colors" />
                                    </div>
                                </div>
                            ))
                        )}
                    </div>
                </div>

                <div className="flex flex-col gap-4 min-h-0">
                    <div className="cyber-panel hud-corners p-4 space-y-3">
                        <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.2em] text-cyber-cyan/60">
                            <Sparkles size={14} />
                            <span>Recherche semantique</span>
                        </div>
                        <div className="flex gap-2">
                            <input
                                type="text"
                                value={remoteQuery}
                                onChange={(event) => setRemoteQuery(event.target.value)}
                                onKeyDown={(event) => event.key === 'Enter' && void handleMemorySearch()}
                                placeholder="Strategie, auteur, pattern, these..."
                                className="flex-1 bg-black/50 border border-white/10 px-3 py-2 text-[12px] text-white/80 outline-none focus:border-cyber-cyan/40"
                            />
                            <button onClick={handleMemorySearch} className="cyber-btn text-[9px] px-3 py-2" disabled={searching}>
                                <Search size={12} className={searching ? 'animate-pulse' : ''} />
                                <span>Interroger</span>
                            </button>
                        </div>
                        <div className="space-y-2 max-h-[240px] overflow-y-auto custom-scrollbar pr-1">
                            {searchResults.length === 0 ? (
                                <div className="border border-dashed border-white/10 p-4 text-[10px] text-white/25">
                                    Aucun resultat distant. Lance une recherche semantique pour verifier la memoire validee.
                                </div>
                            ) : (
                                searchResults.map((result) => (
                                    <div key={`${result.id}-${result.timestamp}`} className="border border-white/[0.05] bg-white/[0.02] p-3">
                                        <div className="flex items-center justify-between gap-3">
                                            <span className="text-[8px] uppercase tracking-[0.15em] text-cyber-cyan/50">{result.role || 'memoire'}</span>
                                            <span className="text-[8px] text-white/20">Score {Math.round((result.score || 0) * 100)}%</span>
                                        </div>
                                        <div className="text-[10px] text-white/40 mt-2 leading-relaxed">{truncate(result.content, 220)}</div>
                                        <div className="text-[8px] text-white/20 mt-2">{formatDate(result.timestamp)}</div>
                                    </div>
                                ))
                            )}
                        </div>
                    </div>

                    <div className="cyber-panel hud-corners p-4 space-y-3 flex-grow min-h-0">
                        <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.2em] text-matrix/60">
                            <Network size={14} />
                            <span>Graphe de connaissance</span>
                        </div>
                        <div className="grid grid-cols-2 gap-3">
                            <div className="border border-white/[0.05] bg-white/[0.02] p-3">
                                <div className="text-[8px] uppercase tracking-[0.15em] text-white/20">Noeuds</div>
                                <div className="text-lg font-bold text-matrix mt-1">{graphNodes.length}</div>
                            </div>
                            <div className="border border-white/[0.05] bg-white/[0.02] p-3">
                                <div className="text-[8px] uppercase tracking-[0.15em] text-white/20">Liens</div>
                                <div className="text-lg font-bold text-cyber-cyan mt-1">{graphLinks.length}</div>
                            </div>
                        </div>
                        <div className="space-y-2 overflow-y-auto custom-scrollbar pr-1">
                            {graphNodes.slice(0, 10).map((node) => (
                                <div key={node.id} className="border border-white/[0.05] bg-white/[0.02] p-3">
                                    <div className="text-[10px] font-bold text-white/70">{node.label}</div>
                                    <div className="text-[8px] text-white/20 mt-1 uppercase tracking-[0.15em]">
                                        {node.role || 'fragment'} | {node.expert || 'core'}
                                    </div>
                                </div>
                            ))}
                            {graphNodes.length === 0 && (
                                <div className="border border-dashed border-white/10 p-4 text-[10px] text-white/25">
                                    Graphe indisponible ou encore vide.
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-1 xl:grid-cols-[0.95fr_1.05fr] gap-6">
                <div className="cyber-panel hud-corners p-4 space-y-4">
                    <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.2em] text-cyber-cyan/60">
                        <Sparkles size={14} />
                        <span>Source de verite memoire</span>
                    </div>

                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                        <div className="border border-white/[0.05] bg-white/[0.02] p-3">
                            <div className="text-[8px] uppercase tracking-[0.15em] text-white/20">Pending</div>
                            <div className="text-lg font-bold text-cyber-amber mt-1">{ingestStatus?.counts?.pending ?? 0}</div>
                        </div>
                        <div className="border border-white/[0.05] bg-white/[0.02] p-3">
                            <div className="text-[8px] uppercase tracking-[0.15em] text-white/20">Rejetes</div>
                            <div className="text-lg font-bold text-cyber-pink mt-1">{ingestStatus?.counts?.rejected ?? 0}</div>
                        </div>
                        <div className="border border-white/[0.05] bg-white/[0.02] p-3">
                            <div className="text-[8px] uppercase tracking-[0.15em] text-white/20">Sources</div>
                            <div className="text-lg font-bold text-cyber-cyan mt-1">{sources.length}</div>
                        </div>
                        <div className="border border-white/[0.05] bg-white/[0.02] p-3">
                            <div className="text-[8px] uppercase tracking-[0.15em] text-white/20">Doublons</div>
                            <div className="text-lg font-bold text-white/70 mt-1">{Math.round((ingestStatus?.duplicate_rate || 0) * 100)}%</div>
                        </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                        <div className="border border-white/[0.05] bg-black/30 p-3 space-y-2">
                            <div className="text-[8px] uppercase tracking-[0.15em] text-white/20">Dependances ingestion</div>
                            {dependencyEntries.length === 0 ? (
                                <div className="text-[10px] text-white/25">Etat indisponible.</div>
                            ) : (
                                dependencyEntries.map(([dependencyName, dependency]) => (
                                    <div key={dependencyName} className="flex items-center justify-between gap-3">
                                        <span className="text-[10px] text-white/35 uppercase tracking-[0.12em]">{dependencyName}</span>
                                        <span className={`px-2 py-0.5 text-[8px] uppercase tracking-[0.18em] border ${dependencyTone(dependency.status)}`}>
                                            {dependency.status}
                                        </span>
                                    </div>
                                ))
                            )}
                        </div>
                        <div className="border border-white/[0.05] bg-black/30 p-3 space-y-2">
                            <div className="text-[8px] uppercase tracking-[0.15em] text-white/20">Provenance active</div>
                            {topSources.length === 0 ? (
                                <div className="text-[10px] text-white/25">Aucune source visible.</div>
                            ) : (
                                topSources.map((source) => (
                                    <div key={`${source.source_name}-${source.url || source.key || source.source_name}`} className="flex items-center justify-between gap-3 text-[10px] text-white/35">
                                        <span>{source.source_name}</span>
                                        <span className="uppercase tracking-[0.12em] text-cyber-cyan/55">{source.family || source.source_type || 'source'}</span>
                                    </div>
                                ))
                            )}
                        </div>
                    </div>
                </div>

                <div className="cyber-panel hud-corners p-4 space-y-4">
                    <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.2em] text-matrix/60">
                        <Database size={14} />
                        <span>Connaissance approuvee</span>
                    </div>
                    <div className="space-y-2 max-h-[320px] overflow-y-auto custom-scrollbar pr-1">
                        {approvedItems.length === 0 ? (
                            <div className="border border-dashed border-white/10 p-4 text-[10px] text-white/25">
                                Aucun element approuve visible pour le moment.
                            </div>
                        ) : (
                            approvedItems.map((item) => (
                                <div key={item.id} className="border border-white/[0.05] bg-white/[0.02] p-3">
                                    <div className="flex items-center justify-between gap-3">
                                        <div className="text-[10px] font-bold text-white/70">{item.title}</div>
                                        <span className="text-[8px] uppercase tracking-[0.15em] text-matrix/50">{item.family}</span>
                                    </div>
                                    <div className="text-[8px] text-white/20 mt-1">
                                        {item.source_name} | {formatDate(item.ingested_at || item.reviewed_at || item.collected_at)}
                                    </div>
                                    <div className="text-[10px] text-white/35 mt-2 leading-relaxed">
                                        {truncate(item.summary_curated || item.summary_raw || 'Aucun resume disponible.', 180)}
                                    </div>
                                </div>
                            ))
                        )}
                    </div>
                </div>
            </div>
        </div>
    )
}
