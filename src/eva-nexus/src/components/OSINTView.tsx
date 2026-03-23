import { type ReactNode, useCallback, useEffect, useMemo, useState } from 'react'
import {
    Activity,
    AlertTriangle,
    BookOpen,
    Globe2,
    Radar,
    Shield,
    Siren,
} from 'lucide-react'

import {
    checkNodeHealth,
    getResearchPeaAnalysis,
    getResearcherTrends,
    getSentinelAlerts,
    getShadowAlerts,
    getShadowMonitors,
    getShadowThreatHistory,
    reconShadow,
    searchShadow,
    type NodeHealth,
    type ResearchResult,
    type ShadowAlert,
    type ShadowMonitor,
    type ShadowThreatAnalysis,
} from '../services/api'

interface OSINTAgent {
    id: string
    name: string
    codename: string
    expert: string
    icon: string
    description: string
    capabilities: string[]
    healthUrl: string
    port: number
    model: string
    phase: string
}

const OSINT_AGENTS: OSINTAgent[] = [
    {
        id: 'shadow',
        name: 'THE SHADOW',
        codename: 'Expert C',
        expert: 'OSINT et investigation',
        icon: 'SH',
        description: 'Agent OSINT et recherche web. Veille, recherche, recon et collecte de signaux ouverts.',
        capabilities: ['Recherche web', 'Reconnaissance', 'Threat intelligence', 'Veille', 'Personas'],
        healthUrl: '/api/shadow/health',
        port: 8002,
        model: 'Dolphin-Qwen',
        phase: 'ALPHA',
    },
    {
        id: 'sentinel',
        name: 'THE SENTINEL',
        codename: 'Expert F',
        expert: 'Cybersecurite active',
        icon: 'SN',
        description: 'Surveillance, alertes securite, integrite, audit et quarantaine.',
        capabilities: ['Monitoring', 'Detection', 'Integrite', 'Audit', 'Quarantaine'],
        healthUrl: '/api/sentinel/health',
        port: 8007,
        model: 'Cyber-Llama',
        phase: 'ALPHA',
    },
    {
        id: 'wraith',
        name: 'THE WRAITH',
        codename: 'Expert D',
        expert: 'Vision et analyse video',
        icon: 'WR',
        description: 'Agent vision pour micro-expressions, CCTV et analyse visuelle.',
        capabilities: ['Video', 'Tracking', 'Micro-expressions', 'CCTV', 'Reconnaissance objets'],
        healthUrl: '/api/wraith/health',
        port: 8012,
        model: 'V-JEPA',
        phase: 'SKELETON',
    },
    {
        id: 'researcher',
        name: 'THE RESEARCHER',
        codename: 'Expert I',
        expert: 'Recherche scientifique',
        icon: 'RS',
        description: 'Agent de veille academique, arXiv et synthese de recherche.',
        capabilities: ['arXiv', 'Synthese', 'Veille', 'Papiers', 'Recherche'],
        healthUrl: '/api/researcher/health',
        port: 8013,
        model: 'Galactica',
        phase: 'ALPHA',
    },
]

function formatDate(value?: string | null) {
    if (!value) {
        return 'indisponible'
    }
    const date = new Date(value)
    if (Number.isNaN(date.getTime())) {
        return 'indisponible'
    }
    return date.toLocaleString('fr-FR', {
        hour: '2-digit',
        minute: '2-digit',
        day: '2-digit',
        month: '2-digit',
    })
}

function cleanMessage(value: string) {
    return (value || '')
        .replace(/[^\x20-\x7E]+/g, ' ')
        .replace(/\s+/g, ' ')
        .trim()
}

function AgentCard({ agent, health }: { agent: OSINTAgent; health?: NodeHealth }) {
    const status = health?.status || 'offline'
    const isOnline = status === 'online'
    const latency = health?.latency ?? -1

    return (
        <div className="cyber-panel hud-corners p-4 flex flex-col gap-3">
            <div className="flex items-start justify-between">
                <div className="flex items-center gap-3">
                    <div className="text-sm font-bold text-white/60 w-7">{agent.icon}</div>
                    <div>
                        <h3 className="font-display text-[11px] font-bold tracking-[0.1em] text-white/70">{agent.name}</h3>
                        <div className="text-[8px] text-white/15 tracking-[0.2em]">{agent.codename} - {agent.expert}</div>
                    </div>
                </div>
                <div className="flex flex-col items-end gap-1">
                    <div className="flex items-center gap-2">
                        <div className={`w-2 h-2 rounded-full ${isOnline ? 'bg-matrix shadow-[0_0_6px_rgba(0,255,65,0.5)]' : 'bg-white/10'}`} />
                        <span className={`text-[8px] tracking-wider ${isOnline ? 'neon-text' : 'text-white/20'}`}>{isOnline ? 'ONLINE' : 'OFFLINE'}</span>
                    </div>
                    <span className="text-[7px] text-white/10 tracking-[0.15em]">{agent.phase} - :{agent.port}</span>
                </div>
            </div>

            <p className="text-[10px] text-white/25 leading-relaxed">{agent.description}</p>

            <div className="flex flex-wrap gap-1.5">
                {agent.capabilities.map((capability) => (
                    <span key={capability} className="px-2 py-0.5 text-[8px] border border-matrix/10 bg-matrix/[0.03] text-matrix/40 tracking-wider">
                        {capability}
                    </span>
                ))}
            </div>

            <div className="flex items-center justify-between pt-2 border-t border-white/[0.03]">
                <span className="text-[8px] text-white/15 tracking-wider">{isOnline ? `LATENCE ${latency}ms` : 'INDISPONIBLE'}</span>
                <span className="text-[8px] text-white/20 tracking-wider">{agent.model}</span>
            </div>
        </div>
    )
}

function DataPanel({ title, icon, children }: { title: string; icon: ReactNode; children: ReactNode }) {
    return (
        <div className="cyber-panel hud-corners p-4">
            <div className="flex items-center gap-2 text-[9px] uppercase tracking-[0.2em] text-matrix/40 mb-3">
                {icon}
                <span>{title}</span>
            </div>
            {children}
        </div>
    )
}

export default function OSINTView() {
    const [agentHealth, setAgentHealth] = useState<Record<string, NodeHealth>>({})
    const [searchResults, setSearchResults] = useState<ResearchResult[]>([])
    const [searching, setSearching] = useState(false)
    const [query, setQuery] = useState('')
    const [alerts, setAlerts] = useState<Array<ShadowAlert & { origin: string }>>([])
    const [threats, setThreats] = useState<ShadowThreatAnalysis[]>([])
    const [monitors, setMonitors] = useState<ShadowMonitor[]>([])
    const [peaData, setPeaData] = useState<Record<string, unknown> | null>(null)
    const [trends, setTrends] = useState<string[]>([])

    const fetchState = useCallback(async () => {
        const [healthResults, shadowAlertPayload, sentinelAlertPayload, threatPayload, monitorPayload, trendPayload] = await Promise.all([
            Promise.all(OSINT_AGENTS.map(async (agent) => ({ id: agent.id, health: await checkNodeHealth(agent.name, agent.healthUrl) }))),
            getShadowAlerts(8),
            getSentinelAlerts(8),
            getShadowThreatHistory(),
            getShadowMonitors(),
            getResearcherTrends('tech'),
        ])

        const nextHealth: Record<string, NodeHealth> = {}
        healthResults.forEach((result) => {
            nextHealth[result.id] = result.health
        })
        setAgentHealth(nextHealth)

        const mergedAlerts = [
            ...(shadowAlertPayload.alerts || []).map((alert) => ({ ...alert, origin: 'shadow' })),
            ...(sentinelAlertPayload.alerts || []).map((alert) => ({ ...alert, origin: 'sentinel' })),
        ]
            .sort((a, b) => String(b.timestamp || '').localeCompare(String(a.timestamp || '')))
            .slice(0, 10)
        setAlerts(mergedAlerts)
        setThreats((threatPayload.analyses || []).slice().reverse().slice(0, 8))
        setMonitors(monitorPayload.monitors || [])
        setTrends(trendPayload.sources || [])
    }, [])

    useEffect(() => {
        void fetchState()
        const interval = setInterval(() => {
            void fetchState()
        }, 10000)
        return () => clearInterval(interval)
    }, [fetchState])

    const handleSearch = async (isRecon: boolean) => {
        if (!query.trim()) {
            return
        }
        setSearching(true)
        if (isRecon) {
            const report = await reconShadow(query.trim())
            const findings = Array.isArray(report.web_findings) ? report.web_findings : []
            setSearchResults(findings as ResearchResult[])
        } else {
            const result = await searchShadow(query.trim(), 8)
            setSearchResults(result.results || [])
        }
        setSearching(false)
    }

    const handlePea = async () => {
        setPeaData(await getResearchPeaAnalysis())
    }

    const onlineCount = useMemo(
        () => OSINT_AGENTS.filter((agent) => agentHealth[agent.id]?.status === 'online').length,
        [agentHealth],
    )

    return (
        <div className="h-full overflow-y-auto p-4 space-y-4 animate-fade-in">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <div className="cyber-panel hud-corners p-4">
                    <div className="text-[8px] text-white/20 tracking-wider mb-1">AGENTS INTEL</div>
                    <div className="text-xl font-bold neon-text">{onlineCount}<span className="text-white/20 text-sm">/{OSINT_AGENTS.length}</span></div>
                </div>
                <div className="cyber-panel hud-corners p-4">
                    <div className="text-[8px] text-white/20 tracking-wider mb-1">ALERTES VIVES</div>
                    <div className="text-xl font-bold neon-text-cyan">{alerts.length}</div>
                </div>
                <div className="cyber-panel hud-corners p-4">
                    <div className="text-[8px] text-white/20 tracking-wider mb-1">THREATS TRACEES</div>
                    <div className="text-xl font-bold neon-text-amber">{threats.length}</div>
                </div>
                <div className="cyber-panel hud-corners p-4">
                    <div className="text-[8px] text-white/20 tracking-wider mb-1">MONITORS ACTIFS</div>
                    <div className="text-xl font-bold text-white/60">{monitors.filter((item) => item.status === 'active').length}</div>
                </div>
            </div>

            <div className="cyber-panel hud-corners p-4 space-y-3">
                <div className="text-[9px] uppercase tracking-[0.2em] text-matrix/40">Recherche OSINT - The Shadow</div>
                <div className="flex gap-2">
                    <input
                        type="text"
                        value={query}
                        onChange={(event) => setQuery(event.target.value)}
                        onKeyDown={(event) => event.key === 'Enter' && void handleSearch(false)}
                        placeholder="> cible, sujet, domaine, organisation..."
                        className="cyber-input flex-1"
                    />
                    <button onClick={() => void handleSearch(false)} className="cyber-btn-cyan shrink-0" disabled={searching}>CHERCHER</button>
                    <button onClick={() => void handleSearch(true)} className="cyber-btn shrink-0" disabled={searching}>RECON</button>
                </div>
                <div className="space-y-2 max-h-[220px] overflow-y-auto custom-scrollbar">
                    {searchResults.length === 0 ? (
                        <div className="text-[10px] text-white/20 border border-dashed border-white/10 p-4">
                            Aucune recherche en cours. Les resultats The Shadow apparaitront ici.
                        </div>
                    ) : (
                        searchResults.map((result, index) => (
                            <a key={`${result.title}-${index}`} href={result.url} target="_blank" rel="noreferrer" className="block p-3 border border-white/[0.03] bg-white/[0.01] hover:bg-white/[0.02]">
                                <div className="text-[10px] text-cyber-cyan/60 truncate">{result.title}</div>
                                <div className="text-[8px] text-matrix/25 truncate">{result.url}</div>
                                <div className="text-[9px] text-white/20 mt-1">{cleanMessage(result.summary)}</div>
                            </a>
                        ))
                    )}
                </div>
            </div>

            <div className="text-[9px] uppercase tracking-[0.2em] text-matrix/40 px-1">AGENTS INTELLIGENCE ({OSINT_AGENTS.length})</div>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
                {OSINT_AGENTS.map((agent) => (
                    <AgentCard key={agent.id} agent={agent} health={agentHealth[agent.id]} />
                ))}
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
                <DataPanel title="Alertes fusionnees" icon={<Siren size={14} />}>
                    <div className="space-y-2 max-h-[260px] overflow-y-auto custom-scrollbar">
                        {alerts.length === 0 ? (
                            <div className="text-[10px] text-white/20 border border-dashed border-white/10 p-4">Aucune alerte remontee.</div>
                        ) : (
                            alerts.map((alert, index) => (
                                <div key={`${alert.timestamp}-${index}`} className="flex items-start gap-3 p-2 border border-white/[0.03] bg-white/[0.01]">
                                    <AlertTriangle size={14} className="text-cyber-pink mt-0.5" />
                                    <div className="flex-1 min-w-0">
                                        <div className="text-[9px] text-white/40 uppercase tracking-[0.15em]">{alert.origin} | {alert.severity || alert.category || 'info'}</div>
                                        <div className="text-[10px] text-white/25 mt-1">{cleanMessage(alert.message)}</div>
                                    </div>
                                    <div className="text-[8px] text-white/10 shrink-0">{formatDate(alert.timestamp)}</div>
                                </div>
                            ))
                        )}
                    </div>
                </DataPanel>

                <DataPanel title="Renseignement menace" icon={<Radar size={14} />}>
                    <div className="space-y-2 max-h-[260px] overflow-y-auto custom-scrollbar">
                        {threats.length === 0 ? (
                            <div className="text-[10px] text-white/20 border border-dashed border-white/10 p-4">Aucune analyse d'IoC disponible.</div>
                        ) : (
                            threats.map((threat) => (
                                <div key={`${threat.indicator}-${threat.analyzed_at}`} className="flex items-center gap-3 p-2 border border-white/[0.03] bg-white/[0.01]">
                                    <Shield size={14} className="text-cyber-amber" />
                                    <div className="flex-1 min-w-0">
                                        <div className="text-[10px] text-white/45 font-bold truncate">{threat.indicator}</div>
                                        <div className="text-[8px] text-white/15">{threat.type} | score {threat.threat_score}</div>
                                    </div>
                                    <span className="text-[8px] text-cyber-amber/60 uppercase tracking-[0.15em]">{threat.severity}</span>
                                </div>
                            ))
                        )}
                    </div>
                </DataPanel>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
                <DataPanel title="Veille active" icon={<Activity size={14} />}>
                    <div className="space-y-2 max-h-[220px] overflow-y-auto custom-scrollbar">
                        {monitors.length === 0 ? (
                            <div className="text-[10px] text-white/20 border border-dashed border-white/10 p-4">Aucune cible de veille active.</div>
                        ) : (
                            monitors.map((monitor) => (
                                <div key={monitor.id} className="border border-white/[0.03] bg-white/[0.01] p-3">
                                    <div className="flex items-center justify-between gap-3">
                                        <div className="text-[10px] font-bold text-white/60">{monitor.keyword}</div>
                                        <span className="text-[8px] text-matrix/50 uppercase tracking-[0.15em]">{monitor.status}</span>
                                    </div>
                                    <div className="text-[8px] text-white/20 mt-1">{monitor.category} | intervalle {monitor.interval_minutes} min | hits {monitor.hits || 0}</div>
                                </div>
                            ))
                        )}
                    </div>
                </DataPanel>

                <DataPanel title="Veille Researcher" icon={<BookOpen size={14} />}>
                    <div className="space-y-2 mb-3">
                        {trends.slice(0, 5).map((source) => (
                            <div key={source} className="flex items-center gap-2 text-[10px] text-white/30">
                                <Globe2 size={12} className="text-cyber-cyan/60" />
                                <span>{source}</span>
                            </div>
                        ))}
                    </div>
                    <button onClick={handlePea} className="cyber-btn-cyan text-[9px] px-3 py-2 mb-3">ANALYSE PEA</button>
                    <div className="border border-white/[0.03] bg-white/[0.01] p-3 text-[10px] text-white/30 min-h-[120px]">
                        {peaData ? String(peaData.message || 'Analyse disponible.') : 'Le module PEA est accessible en lecture. Declenche une analyse pour verifier son etat courant.'}
                    </div>
                </DataPanel>
            </div>
        </div>
    )
}
