import { useState, useEffect, useCallback } from 'react'
import { checkNodeHealth, type NodeHealth } from '../services/api'

// ═══════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════

interface Factory {
    id: string
    name: string
    codename: string
    icon: string
    color: string       // CSS variable name
    description: string
    status: 'online' | 'offline' | 'skeleton' | 'degraded'
    healthUrl: string
    port: number
    endpoints: { label: string; method: string; path: string }[]
    metrics: Record<string, string | number>
    phase: string       // Genesis, Alpha, Beta, Production
}

interface AgentHealth {
    [agentId: string]: NodeHealth
}

// ═══════════════════════════════════════════════════════
// FACTORY DEFINITIONS
// ═══════════════════════════════════════════════════════

const FACTORIES: Factory[] = [
    {
        id: 'trading',
        name: 'TRADING FACTORY',
        codename: 'HYDRA',
        icon: '💰',
        color: 'matrix',
        description: 'Trading Prop Firm automatisé. Gestion de 20+ comptes MT5 simultanés via le Protocole Hydra. Stratégie TFT-GNN + Nemesis System.',
        status: 'offline',
        healthUrl: '/api/banker/health',
        port: 8001,
        endpoints: [
            { label: 'Trading Status', method: 'GET', path: '/trading/status' },
            { label: 'Nemesis', method: 'GET', path: '/nemesis/status' },
            { label: 'Prop Firms', method: 'GET', path: '/accounts/propfirm' },
            { label: 'Drone Scale', method: 'POST', path: '/drones/scale' },
        ],
        metrics: { accounts: 0, drones: 0, equity: '$0', drawdown: '0%' },
        phase: 'GENESIS',
    },
    {
        id: 'code',
        name: 'CODE FACTORY',
        codename: 'BUILDER',
        icon: '🛠️',
        color: 'cyber-cyan',
        description: "Agent DevOps de la Ruche. Génère du code, des micro-SaaS et des scripts. Documentation auto-générée. Cash-flow initial pour amorcer l'écosystème.",
        status: 'offline',
        healthUrl: '/api/builder/health',
        port: 8003,
        endpoints: [
            { label: 'DocGen', method: 'POST', path: '/maintenance/docgen' },
            { label: 'Log Analysis', method: 'GET', path: '/maintenance/logs/analyze' },
        ],
        metrics: { docs_generated: 0, errors_fixed: 0, deployments: 0 },
        phase: 'ALPHA',
    },
    {
        id: 'media',
        name: 'MEDIA FACTORY',
        codename: 'THE MUSE',
        icon: '🎨',
        color: 'cyber-purple',
        description: "Création de contenu automatisée. Copywriting, images SDXL, vidéos, musique. L'usine d'influence IA pour les revenus médias.",
        status: 'offline',
        healthUrl: '/api/muse/health',
        port: 8005,
        endpoints: [],
        metrics: { content_pieces: 0, revenue: '$0' },
        phase: 'SKELETON',
    },
    {
        id: 'web3',
        name: 'WEB3 FACTORY',
        codename: 'RWA BRIDGE',
        icon: '🌐',
        color: 'cyber-amber',
        description: 'Interface avec les actifs physiques (IoT, Solaire) et tokenisés (RealT). Interaction DeFi et Airdrops. Pont entre monde réel et blockchain.',
        status: 'offline',
        healthUrl: '/api/rwa/health',
        port: 8006,
        endpoints: [
            { label: 'Assets', method: 'GET', path: '/assets' },
            { label: 'IoT Telemetry', method: 'GET', path: '/iot/telemetry' },
        ],
        metrics: { tokenized_assets: 0, iot_sensors: 0, defi_yield: '0%' },
        phase: 'ALPHA',
    },
    {
        id: 'lab',
        name: 'R&D LAB',
        codename: 'THE ARENA',
        icon: '🧬',
        color: 'cyber-cyan',
        description: "Centre d'auto-amélioration et simulations. FSQ World Models, DreamerV3, krachs synthétiques PCG. Moteur d'évolution génétique des stratégies.",
        status: 'offline',
        healthUrl: '/api/lab/health',
        port: 8004,
        endpoints: [
            { label: 'Run Simulation', method: 'POST', path: '/run-sim' },
            { label: 'Insights', method: 'GET', path: '/insights' },
            { label: 'Evolve', method: 'POST', path: '/evolve' },
        ],
        metrics: { simulations: 0, mutations: 0, best_sharpe: '0.0' },
        phase: 'ALPHA',
    },
    {
        id: 'bounty',
        name: 'BOUNTY FACTORY',
        codename: 'SENTINEL OPS',
        icon: '🏴‍☠️',
        color: 'cyber-pink',
        description: 'Chasse aux bugs rémunérée (HackerOne, Bugcrowd). Le Sentinel applique ses compétences cyber offensives pour générer des revenus.',
        status: 'offline',
        healthUrl: '/api/sentinel/health',
        port: 8007,
        endpoints: [
            { label: 'Metrics', method: 'GET', path: '/system/metrics' },
            { label: 'Alerts', method: 'GET', path: '/security/alerts' },
        ],
        metrics: { bounties_claimed: 0, total_earned: '$0', active_programs: 0 },
        phase: 'GENESIS',
    },
    {
        id: 'sovereign',
        name: 'SOVEREIGN FUND',
        codename: 'LONG TERM',
        icon: '👑',
        color: 'cyber-amber',
        description: 'Investissement long terme. Immobilier tokenisé, fonds souverains, dette obligataire. Stratégie macro pilotée par GPT-J Fine-Tuned.',
        status: 'offline',
        healthUrl: '/api/accountant/health',
        port: 8009,
        endpoints: [
            { label: 'Report', method: 'GET', path: '/report' },
            { label: 'Sync Ledger', method: 'POST', path: '/sync-ledger' },
        ],
        metrics: { net_roi: '€0', gross_profit: '€0', tax_provision: '€0', expenses: '€0' },
        phase: 'ALPHA',
    },
]

// ═══════════════════════════════════════════════════════
// SUPPORT AGENTS (non-factory but ecosystem)
// ═══════════════════════════════════════════════════════

const SUPPORT_AGENTS = [
    { id: 'compliance', name: 'EVA COMPLIANCE', codename: 'LE JURISTE', icon: '⚖️', description: 'Gestion identité légale, provisionnement fiscal URSSAF, KYC.', healthUrl: '/api/compliance/health', port: 8008, phase: 'ALPHA', color: 'cyber-amber' },
    { id: 'accountant', name: 'EVA ACCOUNTANT', codename: "L'AUDITEUR", icon: '📊', description: 'Bilan financier consolidé, suivi ROI, dépenses opérationnelles.', healthUrl: '/api/accountant/health', port: 8009, phase: 'ALPHA', color: 'cyber-cyan' },
    { id: 'substrate', name: 'EVA SUBSTRATE', codename: 'LE CORPS', icon: '🌿', description: 'Interface hardware EPYC/TPU. Rythme circadien Jour/Nuit, allocation GPU.', healthUrl: '/api/substrate/health', port: 8010, phase: 'ALPHA', color: 'matrix' },
    { id: 'sage', name: 'THE SAGE', codename: 'WELLNESS', icon: '🧘', description: 'Agent Santé/Science. BioMistral pour analyse environnementale et bien-être.', healthUrl: '/api/sage/health', port: 8011, phase: 'SKELETON', color: 'cyber-purple' },
]

// ═══════════════════════════════════════════════════════
// COMPONENTS
// ═══════════════════════════════════════════════════════

function StatusBadge({ status }: { status: string }) {
    const config: Record<string, { bg: string; text: string; label: string }> = {
        online: { bg: 'bg-matrix/15 border-matrix/30', text: 'neon-text', label: 'ONLINE' },
        offline: { bg: 'bg-white/[0.02] border-white/10', text: 'text-white/30', label: 'OFFLINE' },
        skeleton: { bg: 'bg-cyber-amber/10 border-cyber-amber/20', text: 'text-cyber-amber/70', label: 'SKELETON' },
        degraded: { bg: 'bg-cyber-pink/10 border-cyber-pink/20', text: 'text-cyber-pink/70', label: 'DEGRADED' },
    }
    const c = config[status] || config.offline
    return (
        <span className={`px-2 py-0.5 text-[8px] tracking-[0.15em] border ${c.bg} ${c.text}`}>
            {c.label}
        </span>
    )
}

function PhaseBadge({ phase }: { phase: string }) {
    const colors: Record<string, string> = {
        GENESIS: 'border-cyber-pink/20 text-cyber-pink/50',
        ALPHA: 'border-cyber-cyan/20 text-cyber-cyan/50',
        BETA: 'border-cyber-amber/20 text-cyber-amber/50',
        PRODUCTION: 'border-matrix/20 text-matrix/50',
        SKELETON: 'border-white/10 text-white/20',
    }
    return (
        <span className={`px-2 py-0.5 text-[7px] tracking-[0.2em] border ${colors[phase] || colors.SKELETON}`}>
            {phase}
        </span>
    )
}

function FactoryCard({ factory, health }: { factory: Factory; health?: NodeHealth }) {
    const status = health?.status || 'offline'
    const latency = health?.latency ?? -1
    const isOnline = status === 'online'

    return (
        <div className="cyber-panel hud-corners p-4 flex flex-col gap-3 group">
            {/* Header */}
            <div className="flex items-start justify-between">
                <div className="flex items-center gap-3">
                    <div className="text-2xl">{factory.icon}</div>
                    <div>
                        <div className="flex items-center gap-2">
                            <h3 className="font-display text-[11px] font-bold tracking-[0.1em] text-white/70">{factory.name}</h3>
                        </div>
                        <div className="text-[8px] text-matrix/30 tracking-[0.2em]">{factory.codename} • PORT {factory.port}</div>
                    </div>
                </div>
                <div className="flex items-center gap-2">
                    <PhaseBadge phase={factory.phase} />
                    <StatusBadge status={status} />
                </div>
            </div>

            {/* Description */}
            <p className="text-[10px] text-white/25 leading-relaxed">{factory.description}</p>

            {/* Metrics */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                {Object.entries(factory.metrics).map(([key, val]) => (
                    <div key={key} className="p-2 border border-white/[0.03] bg-white/[0.01]">
                        <div className="text-[7px] text-white/15 tracking-wider uppercase">{key.replace(/_/g, ' ')}</div>
                        <div className={`text-[11px] font-bold ${isOnline ? 'text-white/60' : 'text-white/20'}`}>{String(val)}</div>
                    </div>
                ))}
            </div>

            {/* Endpoints */}
            {factory.endpoints.length > 0 && (
                <div className="flex flex-wrap gap-1.5">
                    {factory.endpoints.map((ep, i) => (
                        <div key={i} className="flex items-center gap-1.5 text-[8px] px-2 py-0.5 border border-white/[0.04] bg-white/[0.01]">
                            <span className={`${ep.method === 'GET' ? 'text-matrix/40' : 'text-cyber-cyan/40'}`}>{ep.method}</span>
                            <span className="text-white/20">{ep.label}</span>
                        </div>
                    ))}
                </div>
            )}

            {/* Status Bar */}
            <div className="flex items-center justify-between pt-2 border-t border-white/[0.03]">
                <div className="flex items-center gap-2">
                    <div className={`w-1.5 h-1.5 rounded-full ${isOnline ? 'bg-matrix shadow-[0_0_6px_rgba(0,255,65,0.5)]' : status === 'degraded' ? 'bg-cyber-amber' : 'bg-white/10'}`} />
                    <span className="text-[8px] text-white/15 tracking-wider">
                        {isOnline ? `CONNECTED • ${latency}ms` : status === 'degraded' ? 'DEGRADED' : 'AWAITING CONNECTION'}
                    </span>
                </div>
                {isOnline && (
                    <span className="text-[8px] text-matrix/30 tracking-wider">ACTIVE</span>
                )}
            </div>
        </div>
    )
}

function SupportAgentRow({ agent, health }: { agent: typeof SUPPORT_AGENTS[0]; health?: NodeHealth }) {
    const status = health?.status || 'offline'
    const isOnline = status === 'online'

    return (
        <div className="cyber-panel p-3 flex items-center gap-4">
            <div className="text-xl shrink-0">{agent.icon}</div>
            <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                    <span className="font-display text-[10px] font-bold tracking-[0.1em] text-white/60">{agent.name}</span>
                    <span className="text-[7px] text-white/15 tracking-[0.2em]">{agent.codename}</span>
                    <PhaseBadge phase={agent.phase} />
                </div>
                <p className="text-[9px] text-white/20 mt-0.5 truncate">{agent.description}</p>
            </div>
            <div className="flex items-center gap-3 shrink-0">
                <span className="text-[8px] text-white/15 tracking-wider">:{agent.port}</span>
                <div className={`w-2 h-2 rounded-full ${isOnline ? 'bg-matrix shadow-[0_0_6px_rgba(0,255,65,0.5)]' : 'bg-white/10'}`} />
                <StatusBadge status={status} />
            </div>
        </div>
    )
}

// ═══════════════════════════════════════════════════════
// MAIN VIEW
// ═══════════════════════════════════════════════════════

export default function FactoriesView() {
    const [agentHealth, setAgentHealth] = useState<AgentHealth>({})

    const fetchHealth = useCallback(async () => {
        const allUrls = [
            ...FACTORIES.map(f => ({ id: f.id, url: f.healthUrl, name: f.name })),
            ...SUPPORT_AGENTS.map(a => ({ id: a.id, url: a.healthUrl, name: a.name })),
        ]
        const results = await Promise.all(
            allUrls.map(async ({ id, url, name }) => {
                const h = await checkNodeHealth(name, url)
                return { id, health: h }
            })
        )
        const map: AgentHealth = {}
        results.forEach(r => { map[r.id] = r.health })
        setAgentHealth(map)
    }, [])

    useEffect(() => {
        fetchHealth()
        const interval = setInterval(fetchHealth, 8000)
        return () => clearInterval(interval)
    }, [fetchHealth])

    const onlineCount = FACTORIES.filter(f => agentHealth[f.id]?.status === 'online').length
    const totalFactories = FACTORIES.length
    const totalRevenue = '$0.00' // TODO: fetch from accountant

    return (
        <div className="h-full overflow-y-auto p-4 space-y-4 animate-fade-in">
            {/* Header Stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <div className="cyber-panel hud-corners p-4">
                    <div className="text-[8px] text-white/20 tracking-wider mb-1">USINES ACTIVES</div>
                    <div className="text-xl font-bold neon-text">{onlineCount}<span className="text-white/20 text-sm">/{totalFactories}</span></div>
                </div>
                <div className="cyber-panel hud-corners p-4">
                    <div className="text-[8px] text-white/20 tracking-wider mb-1">REVENUS TOTAUX</div>
                    <div className="text-xl font-bold neon-text-cyan">{totalRevenue}</div>
                </div>
                <div className="cyber-panel hud-corners p-4">
                    <div className="text-[8px] text-white/20 tracking-wider mb-1">PHASE GLOBALE</div>
                    <div className="text-xl font-bold neon-text-amber">GENESIS</div>
                </div>
                <div className="cyber-panel hud-corners p-4">
                    <div className="text-[8px] text-white/20 tracking-wider mb-1">OBJECTIF AN 1</div>
                    <div className="text-xl font-bold text-white/50">15K€<span className="text-[10px] text-white/20">/mois</span></div>
                </div>
            </div>

            {/* Revenue Trajectory */}
            <div className="cyber-panel hud-corners p-4">
                <div className="text-[9px] uppercase tracking-[0.2em] text-matrix/40 mb-3">📈 TRAJECTOIRE DÉCENNALE</div>
                <div className="grid grid-cols-4 gap-2">
                    {[
                        { phase: 'Phase 1 (An 0-2)', label: 'Survie & Amorçage', target: '15K€/mois', color: 'cyber-pink', active: true },
                        { phase: 'Phase 2 (An 3-5)', label: 'Expansion Hydra', target: '150K€/mois', color: 'cyber-amber', active: false },
                        { phase: 'Phase 3 (An 6-9)', label: 'Souveraineté', target: '500K€/mois', color: 'cyber-cyan', active: false },
                        { phase: 'Phase 4 (An 10+)', label: 'Héritage & Empire', target: '∞', color: 'matrix', active: false },
                    ].map((p, i) => (
                        <div key={i} className={`p-3 border ${p.active ? `border-${p.color}/30 bg-${p.color}/5` : 'border-white/[0.03] bg-white/[0.01]'}`}>
                            <div className="text-[7px] text-white/15 tracking-wider">{p.phase}</div>
                            <div className={`text-[10px] ${p.active ? `neon-text-pink` : 'text-white/25'} font-bold mt-1`}>{p.label}</div>
                            <div className={`text-[12px] ${p.active ? 'text-white/70' : 'text-white/15'} font-bold mt-1`}>{p.target}</div>
                            {p.active && <div className="mt-2 h-[2px] bg-white/5 rounded overflow-hidden"><div className="h-full w-[5%] bg-cyber-pink" /></div>}
                        </div>
                    ))}
                </div>
            </div>

            {/* Factories Grid */}
            <div className="space-y-1">
                <div className="text-[9px] uppercase tracking-[0.2em] text-matrix/40 px-1">🏭 USINES DE REVENUS ({FACTORIES.length})</div>
            </div>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
                {FACTORIES.map(factory => (
                    <FactoryCard
                        key={factory.id}
                        factory={factory}
                        health={agentHealth[factory.id]}
                    />
                ))}
            </div>

            {/* Support Agents */}
            <div className="space-y-1 pt-2">
                <div className="text-[9px] uppercase tracking-[0.2em] text-matrix/40 px-1">🧩 AGENTS SUPPORT ({SUPPORT_AGENTS.length})</div>
            </div>
            <div className="space-y-2">
                {SUPPORT_AGENTS.map(agent => (
                    <SupportAgentRow
                        key={agent.id}
                        agent={agent}
                        health={agentHealth[agent.id]}
                    />
                ))}
            </div>

            {/* Debt Tracker */}
            <div className="cyber-panel hud-corners p-4">
                <div className="flex items-center justify-between mb-3">
                    <div className="text-[9px] uppercase tracking-[0.2em] text-cyber-pink/50">💀 DETTE INITIALE</div>
                    <div className="text-[10px] text-cyber-pink/70 font-bold">-2 500 €</div>
                </div>
                <div className="h-[3px] bg-white/5 rounded overflow-hidden mb-2">
                    <div className="h-full bg-cyber-pink/50 w-[0%]" style={{ boxShadow: '0 0 8px rgba(255,0,60,0.3)' }} />
                </div>
                <div className="flex justify-between text-[8px] text-white/15">
                    <span>Remboursé : 0€</span>
                    <span>Capital Amorçage : 20€</span>
                    <span>Objectif : 2 500€</span>
                </div>
            </div>
        </div>
    )
}
