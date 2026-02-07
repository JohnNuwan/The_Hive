import { useState, useEffect, useCallback, useRef } from 'react'
import { checkNodeHealth, type NodeHealth } from '../services/api'

// ═══════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════

interface OSINTAgent {
    id: string
    name: string
    codename: string
    expert: string
    icon: string
    color: string
    description: string
    capabilities: string[]
    healthUrl: string
    port: number
    model: string
    phase: string
}

interface SearchResult {
    title: string
    url: string
    snippet: string
}

interface SecurityAlert {
    id: string
    type: string
    severity: 'critical' | 'high' | 'medium' | 'low' | 'info'
    message: string
    timestamp: string
    source?: string
}

interface ThreatIntel {
    target: string
    is_malicious: boolean
    reputation_score: number
    sources: string[]
}

// ═══════════════════════════════════════════════════════
// AGENT DEFINITIONS
// ═══════════════════════════════════════════════════════

const OSINT_AGENTS: OSINTAgent[] = [
    {
        id: 'shadow',
        name: 'THE SHADOW',
        codename: 'Expert C',
        expert: 'OSINT & Investigation',
        icon: '🌑',
        color: 'cyber-purple',
        description: "Agent OSINT et Recherche Web. Enquêteur et Threat Intel. Effectue des recherches approfondies, du scraping web et de l'analyse de menaces via DuckDuckGo, Brave Search, AlienVault et VirusTotal.",
        capabilities: ['Web Search', 'Entity Recon', 'Threat Intelligence', 'Dark Web Monitoring', 'Social Engineering OSINT'],
        healthUrl: '/api/shadow/health',
        port: 8002,
        model: 'Dolphin-Qwen',
        phase: 'ALPHA',
    },
    {
        id: 'sentinel',
        name: 'THE SENTINEL',
        codename: 'Expert F',
        expert: 'Cybersécurité Active',
        icon: '🛡️',
        color: 'cyber-cyan',
        description: 'Agent de Sécurité et Monitoring de la Ruche. Surveillance système temps réel, détection intrusions (Wazuh/Suricata), intégrité des fichiers et alertes de sécurité.',
        capabilities: ['System Monitoring', 'Intrusion Detection', 'Integrity Checks', 'Vulnerability Scanning', 'Incident Response'],
        healthUrl: '/api/sentinel/health',
        port: 8007,
        model: 'Cyber-Llama',
        phase: 'ALPHA',
    },
    {
        id: 'wraith',
        name: 'THE WRAITH',
        codename: 'Expert D',
        expert: 'Vision & Analyse Vidéo',
        icon: '👁️',
        color: 'cyber-amber',
        description: "Agent Vision. Compréhension sémantique vidéo, Skeleton Tracking via V-JEPA et Coral TPU. Analyse de micro-expressions et surveillance caméra intelligente.",
        capabilities: ['Video Analysis', 'Skeleton Tracking', 'Micro-expression Detection', 'CCTV Monitoring', 'Object Recognition'],
        healthUrl: '/api/wraith/health',
        port: 8012,
        model: 'V-JEPA (Vision)',
        phase: 'SKELETON',
    },
    {
        id: 'researcher',
        name: 'THE RESEARCHER',
        codename: 'Expert I',
        expert: 'Optimisation Algorithmique',
        icon: '🔬',
        color: 'matrix',
        description: 'Agent de Recherche Scientifique. Optimisation algorithmique, analyse de papers (ArXiv), méta-apprentissage. Utilise Galactica pour la compréhension de la littérature scientifique.',
        capabilities: ['ArXiv Analysis', 'Algorithm Optimization', 'Meta-Learning', 'Paper Summarization', 'Experiment Design'],
        healthUrl: '/api/researcher/health',
        port: 8013,
        model: 'Galactica',
        phase: 'SKELETON',
    },
]

// ═══════════════════════════════════════════════════════
// SIMULATED DATA
// ═══════════════════════════════════════════════════════

const MOCK_ALERTS: SecurityAlert[] = [
    { id: 'SA-001', type: 'INTEGRITY_CHECK', severity: 'info', message: 'Kernel binary hash verification: OK', timestamp: new Date(Date.now() - 120000).toISOString(), source: 'Sentinel' },
    { id: 'SA-002', type: 'PORT_SCAN', severity: 'low', message: 'External port scan detected from 185.220.101.x — blocked by firewall', timestamp: new Date(Date.now() - 300000).toISOString(), source: 'Suricata' },
    { id: 'SA-003', type: 'AUTH_ATTEMPT', severity: 'medium', message: 'Failed SSH login attempt (3x) from 45.33.32.156 — IP blacklisted', timestamp: new Date(Date.now() - 600000).toISOString(), source: 'Wazuh' },
    { id: 'SA-004', type: 'SYSTEM_UPDATE', severity: 'info', message: 'ZFS snapshot auto-rotation completed — 12 snapshots retained', timestamp: new Date(Date.now() - 900000).toISOString(), source: 'Phoenix' },
    { id: 'SA-005', type: 'NETWORK_ANOMALY', severity: 'low', message: 'Unusual outbound traffic pattern detected on port 8443 — monitoring', timestamp: new Date(Date.now() - 1200000).toISOString(), source: 'Suricata' },
]

const MOCK_THREAT_FEED: { indicator: string; type: string; risk: string; source: string }[] = [
    { indicator: '185.220.101.0/24', type: 'IP Range', risk: 'HIGH', source: 'AlienVault OTX' },
    { indicator: 'malware-c2.example.com', type: 'Domain', risk: 'CRITICAL', source: 'VirusTotal' },
    { indicator: 'CVE-2026-1234', type: 'Vulnerability', risk: 'MEDIUM', source: 'NVD' },
    { indicator: 'phishing-kit-v3.2', type: 'Malware', risk: 'HIGH', source: 'Hybrid Analysis' },
]

// ═══════════════════════════════════════════════════════
// COMPONENTS
// ═══════════════════════════════════════════════════════

function SeverityBadge({ severity }: { severity: string }) {
    const colors: Record<string, string> = {
        critical: 'bg-cyber-pink/20 border-cyber-pink/30 text-cyber-pink',
        high: 'bg-cyber-pink/10 border-cyber-pink/20 text-cyber-pink/70',
        medium: 'bg-cyber-amber/10 border-cyber-amber/20 text-cyber-amber/70',
        low: 'bg-cyber-cyan/10 border-cyber-cyan/20 text-cyber-cyan/50',
        info: 'bg-white/[0.03] border-white/10 text-white/30',
    }
    return (
        <span className={`px-2 py-0.5 text-[7px] tracking-[0.15em] border ${colors[severity] || colors.info}`}>
            {severity.toUpperCase()}
        </span>
    )
}

function AgentCard({ agent, health }: { agent: OSINTAgent; health?: NodeHealth }) {
    const status = health?.status || 'offline'
    const isOnline = status === 'online'
    const latency = health?.latency ?? -1

    return (
        <div className="cyber-panel hud-corners p-4 flex flex-col gap-3">
            {/* Header */}
            <div className="flex items-start justify-between">
                <div className="flex items-center gap-3">
                    <div className="text-2xl">{agent.icon}</div>
                    <div>
                        <h3 className="font-display text-[11px] font-bold tracking-[0.1em] text-white/70">{agent.name}</h3>
                        <div className="text-[8px] text-white/15 tracking-[0.2em]">{agent.codename} • {agent.expert}</div>
                    </div>
                </div>
                <div className="flex flex-col items-end gap-1">
                    <div className="flex items-center gap-2">
                        <div className={`w-2 h-2 rounded-full ${isOnline ? 'bg-matrix shadow-[0_0_6px_rgba(0,255,65,0.5)]' : 'bg-white/10'}`} />
                        <span className={`text-[8px] tracking-wider ${isOnline ? 'neon-text' : 'text-white/20'}`}>
                            {isOnline ? 'ONLINE' : 'OFFLINE'}
                        </span>
                    </div>
                    <span className="text-[7px] text-white/10 tracking-[0.15em]">{agent.phase} • :{agent.port}</span>
                </div>
            </div>

            {/* Description */}
            <p className="text-[10px] text-white/25 leading-relaxed">{agent.description}</p>

            {/* Model */}
            <div className="flex items-center gap-2">
                <span className="text-[8px] text-white/15 tracking-wider">MODEL:</span>
                <span className="px-2 py-0.5 text-[9px] border border-white/[0.05] bg-white/[0.01] text-white/40">{agent.model}</span>
            </div>

            {/* Capabilities */}
            <div className="flex flex-wrap gap-1.5">
                {agent.capabilities.map((cap, i) => (
                    <span key={i} className="px-2 py-0.5 text-[8px] border border-matrix/10 bg-matrix/[0.03] text-matrix/40 tracking-wider">
                        {cap}
                    </span>
                ))}
            </div>

            {/* Status */}
            <div className="flex items-center justify-between pt-2 border-t border-white/[0.03]">
                <span className="text-[8px] text-white/15 tracking-wider">
                    {isOnline ? `LATENCY: ${latency}ms` : 'AWAITING DEPLOYMENT'}
                </span>
                {isOnline && <span className="text-[8px] neon-text tracking-wider">OPERATIONAL</span>}
            </div>
        </div>
    )
}

function OSINTSearchBar({ onSearch }: { onSearch: (q: string) => void }) {
    const [query, setQuery] = useState('')
    const inputRef = useRef<HTMLInputElement>(null)

    const handleSubmit = () => {
        if (query.trim()) {
            onSearch(query.trim())
        }
    }

    return (
        <div className="cyber-panel hud-corners p-4">
            <div className="text-[9px] uppercase tracking-[0.2em] text-matrix/40 mb-3">🔍 OSINT SEARCH — THE SHADOW</div>
            <div className="flex gap-2">
                <input
                    ref={inputRef}
                    type="text"
                    value={query}
                    onChange={e => setQuery(e.target.value)}
                    onKeyDown={e => e.key === 'Enter' && handleSubmit()}
                    placeholder="> Enter search query or target for recon..."
                    className="cyber-input flex-1"
                />
                <button onClick={handleSubmit} className="cyber-btn-cyan shrink-0">
                    SEARCH
                </button>
                <button onClick={() => { if (query.trim()) onSearch(`recon:${query.trim()}`) }} className="cyber-btn shrink-0">
                    RECON
                </button>
            </div>
            <div className="flex gap-3 mt-2 text-[8px] text-white/10">
                <span>Engines: DuckDuckGo • Brave • Shodan</span>
                <span>|</span>
                <span>Threat Intel: AlienVault • VirusTotal • NVD</span>
            </div>
        </div>
    )
}

function SearchResultsPanel({ results, loading }: { results: SearchResult[]; loading: boolean }) {
    if (loading) {
        return (
            <div className="cyber-panel p-4 text-center">
                <div className="text-[10px] text-matrix/50 animate-pulse tracking-wider">SEARCHING DEEP WEB...</div>
            </div>
        )
    }

    if (results.length === 0) return null

    return (
        <div className="cyber-panel hud-corners p-4">
            <div className="text-[9px] uppercase tracking-[0.2em] text-cyber-cyan/40 mb-3">SEARCH RESULTS ({results.length})</div>
            <div className="space-y-2 max-h-[200px] overflow-y-auto">
                {results.map((r, i) => (
                    <div key={i} className="p-2 border border-white/[0.03] bg-white/[0.01] hover:bg-white/[0.02] transition-colors">
                        <div className="text-[10px] text-cyber-cyan/60 truncate">{r.title}</div>
                        <div className="text-[8px] text-matrix/25 truncate">{r.url}</div>
                        {r.snippet && <div className="text-[9px] text-white/20 mt-1">{r.snippet}</div>}
                    </div>
                ))}
            </div>
        </div>
    )
}

function SecurityAlertsPanel() {
    return (
        <div className="cyber-panel hud-corners p-4">
            <div className="flex items-center justify-between mb-3">
                <div className="text-[9px] uppercase tracking-[0.2em] text-cyber-pink/50">🚨 SECURITY ALERTS</div>
                <div className="text-[8px] text-white/15 tracking-wider">{MOCK_ALERTS.length} EVENTS</div>
            </div>
            <div className="space-y-2 max-h-[250px] overflow-y-auto">
                {MOCK_ALERTS.map(alert => (
                    <div key={alert.id} className="flex items-start gap-3 p-2 border border-white/[0.03] bg-white/[0.01]">
                        <SeverityBadge severity={alert.severity} />
                        <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2">
                                <span className="text-[9px] text-white/40">{alert.type}</span>
                                {alert.source && <span className="text-[7px] text-matrix/25 tracking-wider">{alert.source}</span>}
                            </div>
                            <div className="text-[10px] text-white/25 mt-0.5">{alert.message}</div>
                        </div>
                        <div className="text-[8px] text-white/10 shrink-0">
                            {new Date(alert.timestamp).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' })}
                        </div>
                    </div>
                ))}
            </div>
        </div>
    )
}

function ThreatFeedPanel() {
    return (
        <div className="cyber-panel hud-corners p-4">
            <div className="text-[9px] uppercase tracking-[0.2em] text-cyber-amber/50 mb-3">⚠️ THREAT INTELLIGENCE FEED</div>
            <div className="space-y-2">
                {MOCK_THREAT_FEED.map((threat, i) => (
                    <div key={i} className="flex items-center gap-3 p-2 border border-white/[0.03] bg-white/[0.01]">
                        <span className={`text-[8px] tracking-wider px-2 py-0.5 border shrink-0 ${
                            threat.risk === 'CRITICAL' ? 'border-cyber-pink/30 text-cyber-pink/70 bg-cyber-pink/10' :
                            threat.risk === 'HIGH' ? 'border-cyber-pink/20 text-cyber-pink/50 bg-cyber-pink/5' :
                            'border-cyber-amber/20 text-cyber-amber/50 bg-cyber-amber/5'
                        }`}>{threat.risk}</span>
                        <div className="flex-1 min-w-0">
                            <div className="text-[10px] text-white/40 truncate font-bold">{threat.indicator}</div>
                            <div className="text-[8px] text-white/15">{threat.type}</div>
                        </div>
                        <span className="text-[8px] text-white/10 shrink-0">{threat.source}</span>
                    </div>
                ))}
            </div>
        </div>
    )
}

function NetworkMapPanel() {
    return (
        <div className="cyber-panel hud-corners p-4">
            <div className="text-[9px] uppercase tracking-[0.2em] text-matrix/40 mb-3">🌐 NETWORK TOPOLOGY — THE HIVE</div>
            <div className="grid grid-cols-3 gap-3">
                {/* Bastion */}
                <div className="col-span-3 p-3 border border-cyber-pink/15 bg-cyber-pink/[0.02]">
                    <div className="flex items-center justify-between">
                        <div>
                            <div className="text-[9px] text-cyber-pink/50 font-bold tracking-wider">CT 400 — THE BASTION</div>
                            <div className="text-[8px] text-white/15 mt-0.5">Sécurité offensive/défensive • Wazuh • Suricata</div>
                        </div>
                        <div className="flex items-center gap-2">
                            <div className="w-1.5 h-1.5 rounded-full bg-matrix shadow-[0_0_4px_rgba(0,255,65,0.3)]" />
                            <span className="text-[7px] text-matrix/40">MONITORING</span>
                        </div>
                    </div>
                </div>

                {/* VMs */}
                <div className="p-3 border border-matrix/15 bg-matrix/[0.02]">
                    <div className="text-[9px] neon-text font-bold tracking-wider">VM 100</div>
                    <div className="text-[8px] text-white/20 mt-0.5">THE BRAIN</div>
                    <div className="text-[7px] text-white/10 mt-1">FastAPI Orchestrator</div>
                </div>
                <div className="p-3 border border-cyber-cyan/15 bg-cyber-cyan/[0.02]">
                    <div className="text-[9px] neon-text-cyan font-bold tracking-wider">VM 101</div>
                    <div className="text-[8px] text-white/20 mt-0.5">THE COUNCIL</div>
                    <div className="text-[7px] text-white/10 mt-1">GPU RTX 3090 Inference</div>
                </div>
                <div className="p-3 border border-cyber-amber/15 bg-cyber-amber/[0.02]">
                    <div className="text-[9px] neon-text-amber font-bold tracking-wider">VM 200</div>
                    <div className="text-[8px] text-white/20 mt-0.5">TRADING FLOOR</div>
                    <div className="text-[7px] text-white/10 mt-1">Hydra 20x MT5</div>
                </div>

                {/* Arena */}
                <div className="col-span-3 p-3 border border-cyber-cyan/10 bg-cyber-cyan/[0.01]">
                    <div className="flex items-center justify-between">
                        <div>
                            <div className="text-[9px] text-cyber-cyan/40 font-bold tracking-wider">CT 500 — THE ARENA</div>
                            <div className="text-[8px] text-white/15 mt-0.5">Sandbox isolé • Tests • Code non-vérifié</div>
                        </div>
                        <div className="flex items-center gap-2">
                            <div className="w-1.5 h-1.5 rounded-full bg-cyber-cyan shadow-[0_0_4px_rgba(0,212,255,0.3)]" />
                            <span className="text-[7px] text-cyber-cyan/40">ISOLATED</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}

// ═══════════════════════════════════════════════════════
// MAIN VIEW
// ═══════════════════════════════════════════════════════

export default function OSINTView() {
    const [agentHealth, setAgentHealth] = useState<Record<string, NodeHealth>>({})
    const [searchResults, setSearchResults] = useState<SearchResult[]>([])
    const [searching, setSearching] = useState(false)

    const fetchHealth = useCallback(async () => {
        const results = await Promise.all(
            OSINT_AGENTS.map(async (agent) => {
                const h = await checkNodeHealth(agent.name, agent.healthUrl)
                return { id: agent.id, health: h }
            })
        )
        const map: Record<string, NodeHealth> = {}
        results.forEach(r => { map[r.id] = r.health })
        setAgentHealth(map)
    }, [])

    useEffect(() => {
        fetchHealth()
        const interval = setInterval(fetchHealth, 8000)
        return () => clearInterval(interval)
    }, [fetchHealth])

    const handleSearch = async (query: string) => {
        setSearching(true)
        setSearchResults([])
        try {
            const isRecon = query.startsWith('recon:')
            const q = isRecon ? query.slice(6) : query
            const endpoint = isRecon ? `/api/shadow/recon?target=${encodeURIComponent(q)}` : `/api/shadow/search?q=${encodeURIComponent(q)}`
            
            const controller = new AbortController()
            const timeout = setTimeout(() => controller.abort(), 10000)
            const res = await fetch(endpoint, { signal: controller.signal })
            clearTimeout(timeout)

            if (res.ok) {
                const data = await res.json()
                const results = isRecon ? data.web_findings || [] : data.results || []
                setSearchResults(results)
            } else {
                setSearchResults([{ title: '⚠ Shadow agent offline', url: '', snippet: 'Unable to reach The Shadow. Deploy the agent first.' }])
            }
        } catch {
            setSearchResults([{ title: '⚠ Connection failed', url: '', snippet: 'Shadow agent is not reachable. Check deployment status.' }])
        } finally {
            setSearching(false)
        }
    }

    const onlineCount = OSINT_AGENTS.filter(a => agentHealth[a.id]?.status === 'online').length
    const threatLevel = onlineCount >= 2 ? 'GREEN' : onlineCount >= 1 ? 'YELLOW' : 'UNKNOWN'
    const threatColor = threatLevel === 'GREEN' ? 'neon-text' : threatLevel === 'YELLOW' ? 'neon-text-amber' : 'text-white/30'

    return (
        <div className="h-full overflow-y-auto p-4 space-y-4 animate-fade-in">
            {/* Header Stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <div className="cyber-panel hud-corners p-4">
                    <div className="text-[8px] text-white/20 tracking-wider mb-1">AGENTS INTEL ACTIFS</div>
                    <div className="text-xl font-bold neon-text">{onlineCount}<span className="text-white/20 text-sm">/{OSINT_AGENTS.length}</span></div>
                </div>
                <div className="cyber-panel hud-corners p-4">
                    <div className="text-[8px] text-white/20 tracking-wider mb-1">THREAT LEVEL</div>
                    <div className={`text-xl font-bold ${threatColor}`}>{threatLevel}</div>
                </div>
                <div className="cyber-panel hud-corners p-4">
                    <div className="text-[8px] text-white/20 tracking-wider mb-1">ALERTS (24H)</div>
                    <div className="text-xl font-bold neon-text-cyan">{MOCK_ALERTS.length}</div>
                </div>
                <div className="cyber-panel hud-corners p-4">
                    <div className="text-[8px] text-white/20 tracking-wider mb-1">THREAT FEEDS</div>
                    <div className="text-xl font-bold neon-text-amber">{MOCK_THREAT_FEED.length}</div>
                </div>
            </div>

            {/* Search */}
            <OSINTSearchBar onSearch={handleSearch} />
            <SearchResultsPanel results={searchResults} loading={searching} />

            {/* Agents Grid */}
            <div className="text-[9px] uppercase tracking-[0.2em] text-matrix/40 px-1">🕵️ INTELLIGENCE AGENTS ({OSINT_AGENTS.length})</div>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
                {OSINT_AGENTS.map(agent => (
                    <AgentCard
                        key={agent.id}
                        agent={agent}
                        health={agentHealth[agent.id]}
                    />
                ))}
            </div>

            {/* Security & Threat Intel */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
                <SecurityAlertsPanel />
                <ThreatFeedPanel />
            </div>

            {/* Network Map */}
            <NetworkMapPanel />
        </div>
    )
}
