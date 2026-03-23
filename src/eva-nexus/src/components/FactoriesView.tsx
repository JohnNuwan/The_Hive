import { type ReactNode, useCallback, useEffect, useMemo, useState } from 'react'

import {
    checkNodeHealth,
    getAccountantDashboard,
    getBuilderHealth,
    getComplianceAlerts,
    getComplianceUrssafReport,
    getMuseStats,
    getRwaHealth,
    getRwaPortfolio,
    getRwaRecommendations,
    getRwaStrategy,
    type AccountantDashboard,
    type BuilderHealth,
    type ComplianceAlert,
    type ComplianceUrssafReport,
    type MuseStats,
    type NodeHealth,
    type RwaHealth,
} from '../services/api'
import BuilderWorkbench from './BuilderWorkbench'

interface Factory {
    id: string
    name: string
    codename: string
    icon: string
    description: string
    healthUrl: string
    port: number
    phase: string
}

const FACTORIES: Factory[] = [
    {
        id: 'code',
        name: 'CODE FACTORY',
        codename: 'BUILDER',
        icon: 'CD',
        description: 'Generation de code, builds, deploiements et pipeline BMAD.',
        healthUrl: '/api/builder/health',
        port: 8003,
        phase: 'ALPHA',
    },
    {
        id: 'media',
        name: 'MEDIA FACTORY',
        codename: 'THE MUSE',
        icon: 'MD',
        description: 'Generation de contenu, niches, video et viralisation.',
        healthUrl: '/api/muse/health',
        port: 8005,
        phase: 'ALPHA',
    },
    {
        id: 'web3',
        name: 'RWA FACTORY',
        codename: 'SOVEREIGN BRIDGE',
        icon: 'W3',
        description: 'Portefeuille d actifs reels, strategie et telemetrie IoT.',
        healthUrl: '/api/rwa/health',
        port: 8006,
        phase: 'ALPHA',
    },
    {
        id: 'sovereign',
        name: 'ACCOUNTING FACTORY',
        codename: 'AUDIT LAYER',
        icon: 'AC',
        description: 'Synthese financiere, depenses, ROI et projections.',
        healthUrl: '/api/accountant/health',
        port: 8009,
        phase: 'ALPHA',
    },
    {
        id: 'compliance',
        name: 'COMPLIANCE FACTORY',
        codename: 'KEEPER',
        icon: 'CP',
        description: 'Ledger fiscal, rapports URSSAF et alertes declaratives.',
        healthUrl: '/api/compliance/health',
        port: 8008,
        phase: 'ALPHA',
    },
    {
        id: 'substrate',
        name: 'SUBSTRATE OPS',
        codename: 'THE BODY',
        icon: 'SB',
        description: 'Corps hardware, metrics energie et orchestration physique.',
        healthUrl: '/api/substrate/health',
        port: 8010,
        phase: 'ALPHA',
    },
]

const SUPPORT_AGENTS = [
    { id: 'sentinel', name: 'EVA SENTINEL', codename: 'SECOPS', description: 'Supervision securite et audit.', healthUrl: '/api/sentinel/health', port: 8007, phase: 'ALPHA' },
    { id: 'shadow', name: 'THE SHADOW', codename: 'OSINT', description: 'Recherche, recon et veille.', healthUrl: '/api/shadow/health', port: 8002, phase: 'ALPHA' },
    { id: 'researcher', name: 'THE RESEARCHER', codename: 'KNOWLEDGE', description: 'Veille academique et ingestion.', healthUrl: '/api/researcher/health', port: 8013, phase: 'ALPHA' },
] as const

function formatCurrency(value: number | undefined, currency = 'EUR') {
    return new Intl.NumberFormat('fr-FR', { style: 'currency', currency, maximumFractionDigits: 0 }).format(value || 0)
}

function formatNumber(value: number | undefined) {
    return new Intl.NumberFormat('fr-FR', { maximumFractionDigits: 2 }).format(value || 0)
}

function HealthPill({ status }: { status: string }) {
    const config: Record<string, string> = {
        online: 'border-matrix/30 bg-matrix/10 text-matrix',
        degraded: 'border-cyber-amber/20 bg-cyber-amber/10 text-cyber-amber',
        offline: 'border-white/10 bg-white/[0.02] text-white/25',
    }
    return <span className={`px-2 py-0.5 text-[8px] uppercase tracking-[0.2em] border ${config[status] || config.offline}`}>{status}</span>
}

function PhasePill({ value }: { value: string }) {
    return <span className="px-2 py-0.5 text-[7px] uppercase tracking-[0.2em] border border-cyber-cyan/20 bg-cyber-cyan/10 text-cyber-cyan/60">{value}</span>
}

function Section({ title, children }: { title: string; children: ReactNode }) {
    return (
        <div className="cyber-panel hud-corners p-4 space-y-3">
            <div className="text-[9px] uppercase tracking-[0.2em] text-matrix/40">{title}</div>
            {children}
        </div>
    )
}

function SummaryTile({ label, value, tone = 'matrix' }: { label: string; value: string; tone?: 'matrix' | 'cyan' | 'amber' | 'pink' }) {
    const toneClass = {
        matrix: 'text-matrix',
        cyan: 'text-cyber-cyan',
        amber: 'text-cyber-amber',
        pink: 'text-cyber-pink',
    }[tone]
    return (
        <div className="cyber-panel hud-corners p-4">
            <div className="text-[8px] text-white/20 tracking-wider mb-1">{label}</div>
            <div className={`text-xl font-bold ${toneClass}`}>{value}</div>
        </div>
    )
}

function FactoryCard({ factory, health, metrics }: { factory: Factory; health?: NodeHealth; metrics: Array<{ label: string; value: string }> }) {
    const status = health?.status || 'offline'
    return (
        <div className="cyber-panel hud-corners p-4 flex flex-col gap-3">
            <div className="flex items-start justify-between gap-3">
                <div className="flex items-center gap-3">
                    <div className="text-lg font-bold text-white/60 w-8">{factory.icon}</div>
                    <div>
                        <div className="text-[11px] font-bold tracking-[0.1em] text-white/70">{factory.name}</div>
                        <div className="text-[8px] text-white/15 tracking-[0.2em]">{factory.codename} - :{factory.port}</div>
                    </div>
                </div>
                <div className="flex items-center gap-2">
                    <PhasePill value={factory.phase} />
                    <HealthPill status={status} />
                </div>
            </div>
            <p className="text-[10px] text-white/25 leading-relaxed">{factory.description}</p>
            <div className="grid grid-cols-2 gap-2">
                {metrics.map((metric) => (
                    <div key={metric.label} className="p-2 border border-white/[0.03] bg-white/[0.01]">
                        <div className="text-[7px] text-white/15 tracking-wider uppercase">{metric.label}</div>
                        <div className="text-[11px] font-bold text-white/60">{metric.value}</div>
                    </div>
                ))}
            </div>
        </div>
    )
}

export default function FactoriesView() {
    const [agentHealth, setAgentHealth] = useState<Record<string, NodeHealth>>({})
    const [builderHealth, setBuilderHealth] = useState<BuilderHealth | null>(null)
    const [accountantDashboard, setAccountantDashboard] = useState<AccountantDashboard | null>(null)
    const [complianceReport, setComplianceReport] = useState<ComplianceUrssafReport | null>(null)
    const [complianceAlerts, setComplianceAlerts] = useState<ComplianceAlert[]>([])
    const [rwaHealth, setRwaHealth] = useState<RwaHealth | null>(null)
    const [rwaPortfolio, setRwaPortfolio] = useState<Record<string, unknown> | null>(null)
    const [rwaStrategy, setRwaStrategy] = useState<Record<string, unknown> | null>(null)
    const [rwaRecommendations, setRwaRecommendations] = useState<Record<string, unknown> | null>(null)
    const [museStats, setMuseStats] = useState<MuseStats | null>(null)

    const fetchState = useCallback(async () => {
        const [healthResults, builder, accountant, urssaf, alerts, rwaHealthPayload, portfolio, strategy, recommendations, muse] = await Promise.all([
            Promise.all(
                [...FACTORIES, ...SUPPORT_AGENTS].map(async (service) => ({
                    id: service.id,
                    health: await checkNodeHealth(service.name, service.healthUrl),
                })),
            ),
            getBuilderHealth(),
            getAccountantDashboard(),
            getComplianceUrssafReport(),
            getComplianceAlerts(8),
            getRwaHealth(),
            getRwaPortfolio(),
            getRwaStrategy(),
            getRwaRecommendations(),
            getMuseStats(),
        ])

        const nextHealth: Record<string, NodeHealth> = {}
        healthResults.forEach((result) => {
            nextHealth[result.id] = result.health
        })
        setAgentHealth(nextHealth)
        setBuilderHealth(builder)
        setAccountantDashboard(accountant)
        setComplianceReport(urssaf)
        setComplianceAlerts(alerts?.alerts || [])
        setRwaHealth(rwaHealthPayload)
        setRwaPortfolio(portfolio)
        setRwaStrategy(strategy)
        setRwaRecommendations(recommendations)
        setMuseStats(muse)
    }, [])

    useEffect(() => {
        void fetchState()
        const interval = setInterval(() => {
            void fetchState()
        }, 12000)
        return () => clearInterval(interval)
    }, [fetchState])

    const onlineCount = useMemo(
        () => FACTORIES.filter((factory) => agentHealth[factory.id]?.status === 'online').length,
        [agentHealth],
    )

    const netRoi = accountantDashboard?.summary?.net_roi || 0
    const factoryMetrics = useMemo(() => {
        const totalValuation = Number(rwaPortfolio?.total_valuation || rwaHealth?.total_valuation || 0)
        const museGenerations = museStats?.total_generations || 0
        const net = accountantDashboard?.summary?.net_roi || 0
        const provisions = complianceReport?.total_provisions || 0
        return {
            code: [
                { label: 'builds', value: String(builderHealth?.builds_completed || 0) },
                { label: 'pipelines', value: String(builderHealth?.active_pipelines || 0) },
                { label: 'catalogue', value: String(builderHealth?.public_api_entries || 0) },
                { label: 'forge', value: String(builderHealth?.forge_runs || 0) },
            ],
            media: [
                { label: 'generations', value: String(museGenerations) },
                { label: 'templates', value: String(museStats?.available_templates || 0) },
                { label: 'mode', value: museStats?.mode || 'indisponible' },
                { label: 'modele', value: museStats?.model || 'indisponible' },
            ],
            web3: [
                { label: 'assets', value: String(rwaHealth?.total_assets || 0) },
                { label: 'valuation', value: formatCurrency(totalValuation) },
                { label: 'yield', value: `${formatNumber(Number(rwaPortfolio?.weighted_yield || 0))}%` },
                { label: 'phase', value: String(rwaPortfolio?.phase || 'indisponible') },
            ],
            sovereign: [
                { label: 'net roi', value: formatCurrency(net) },
                { label: 'brut', value: formatCurrency(accountantDashboard?.summary?.gross_profit || 0) },
                { label: 'taxes', value: formatCurrency(accountantDashboard?.summary?.total_taxes || 0) },
                { label: 'depenses', value: formatCurrency(accountantDashboard?.summary?.total_expenses || 0) },
            ],
            compliance: [
                { label: 'periode', value: complianceReport?.period || 'indisponible' },
                { label: 'urssaf', value: formatCurrency(complianceReport?.cotisations_urssaf || 0) },
                { label: 'provisions', value: formatCurrency(provisions) },
                { label: 'alertes', value: String(complianceAlerts.length) },
            ],
            substrate: [
                { label: 'status', value: agentHealth.substrate?.status || 'offline' },
                { label: 'latence', value: agentHealth.substrate?.latency >= 0 ? `${agentHealth.substrate.latency}ms` : 'n/a' },
                { label: 'mode', value: 'observabilite' },
                { label: 'scope', value: 'non trading' },
            ],
        }
    }, [accountantDashboard, agentHealth, builderHealth, complianceAlerts.length, complianceReport, museStats, rwaHealth, rwaPortfolio])

    return (
        <div className="h-full overflow-y-auto p-4 space-y-4 animate-fade-in">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <SummaryTile label="Usines online" value={`${onlineCount}/${FACTORIES.length}`} tone="matrix" />
                <SummaryTile label="Net ROI" value={formatCurrency(netRoi)} tone="cyan" />
                <SummaryTile label="Provisions URSSAF" value={formatCurrency(complianceReport?.total_provisions || 0)} tone="amber" />
                <SummaryTile label="Valorisation RWA" value={formatCurrency(Number(rwaHealth?.total_valuation || 0))} tone="pink" />
            </div>

            <div className="space-y-1">
                <div className="text-[9px] uppercase tracking-[0.2em] text-matrix/40 px-1">USINES BUSINESS ({FACTORIES.length})</div>
            </div>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
                {FACTORIES.map((factory) => (
                    <FactoryCard key={factory.id} factory={factory} health={agentHealth[factory.id]} metrics={factoryMetrics[factory.id as keyof typeof factoryMetrics] || []} />
                ))}
            </div>

            <BuilderWorkbench />

            <div className="grid grid-cols-1 xl:grid-cols-[1fr_1fr_1fr] gap-4">
                <Section title="Cockpit financier">
                    <div className="grid grid-cols-2 gap-2">
                        <div className="border border-white/[0.03] bg-white/[0.01] p-3">
                            <div className="text-[8px] text-white/20 uppercase tracking-[0.15em]">Profit brut</div>
                            <div className="text-lg font-bold text-matrix mt-1">{formatCurrency(accountantDashboard?.summary?.gross_profit || 0)}</div>
                        </div>
                        <div className="border border-white/[0.03] bg-white/[0.01] p-3">
                            <div className="text-[8px] text-white/20 uppercase tracking-[0.15em]">Depenses</div>
                            <div className="text-lg font-bold text-cyber-pink mt-1">{formatCurrency(accountantDashboard?.summary?.total_expenses || 0)}</div>
                        </div>
                    </div>
                    <div className="space-y-2">
                        {Object.entries(accountantDashboard?.expenses_by_category || {}).slice(0, 5).map(([category, amount]) => (
                            <div key={category} className="flex items-center justify-between text-[10px] text-white/35 border-b border-white/[0.03] pb-2">
                                <span>{category}</span>
                                <span>{formatCurrency(Number(amount || 0))}</span>
                            </div>
                        ))}
                        {Object.keys(accountantDashboard?.expenses_by_category || {}).length === 0 && (
                            <div className="text-[10px] text-white/25 border border-dashed border-white/10 p-3">Aucune repartition de depenses disponible.</div>
                        )}
                    </div>
                </Section>

                <Section title="Conformite et obligations">
                    <div className="space-y-2 text-[10px] text-white/35">
                        <div className="flex items-center justify-between"><span>Periode</span><span>{complianceReport?.period || 'indisponible'}</span></div>
                        <div className="flex items-center justify-between"><span>CA trimestre</span><span>{formatCurrency(complianceReport?.gross_revenue_quarter || 0)}</span></div>
                        <div className="flex items-center justify-between"><span>URSSAF</span><span>{formatCurrency(complianceReport?.cotisations_urssaf || 0)}</span></div>
                        <div className="flex items-center justify-between"><span>Provisions</span><span>{formatCurrency(complianceReport?.total_provisions || 0)}</span></div>
                    </div>
                    <div className="space-y-2 pt-2 border-t border-white/[0.04]">
                        {complianceAlerts.length === 0 ? (
                            <div className="text-[10px] text-white/25 border border-dashed border-white/10 p-3">Aucune alerte compliance active.</div>
                        ) : (
                            complianceAlerts.slice(0, 5).map((alert, index) => (
                                <div key={`${alert.timestamp}-${index}`} className="border border-white/[0.03] bg-white/[0.01] p-3">
                                    <div className="text-[8px] uppercase tracking-[0.15em] text-cyber-amber/60">{alert.severity || alert.category || 'info'}</div>
                                    <div className="text-[10px] text-white/30 mt-1">{String(alert.message || '').replace(/[^\x20-\x7E]+/g, ' ').trim()}</div>
                                </div>
                            ))
                        )}
                    </div>
                </Section>

                <Section title="Portefeuille souverain et media">
                    <div className="space-y-2 text-[10px] text-white/35">
                        <div className="flex items-center justify-between"><span>Phase RWA</span><span>{String(rwaPortfolio?.phase || 'indisponible')}</span></div>
                        <div className="flex items-center justify-between"><span>Actifs</span><span>{String(rwaHealth?.total_assets || 0)}</span></div>
                        <div className="flex items-center justify-between"><span>Yield pond. </span><span>{`${formatNumber(Number(rwaPortfolio?.weighted_yield || 0))}%`}</span></div>
                        <div className="flex items-center justify-between"><span>Generations Muse</span><span>{String(museStats?.total_generations || 0)}</span></div>
                    </div>
                    <div className="pt-2 border-t border-white/[0.04] space-y-2">
                        <div className="text-[8px] uppercase tracking-[0.15em] text-cyber-cyan/60">Strategie</div>
                        <div className="text-[10px] text-white/30">{String((rwaStrategy?.current_phase as Record<string, unknown> | undefined)?.name || 'Strategie indisponible.')}</div>
                        <div className="text-[8px] uppercase tracking-[0.15em] text-cyber-cyan/60 pt-2">Recommandations</div>
                        {Array.isArray(rwaRecommendations?.recommendations) && rwaRecommendations.recommendations.length > 0 ? (
                            (rwaRecommendations.recommendations as Array<Record<string, unknown>>).slice(0, 3).map((item, index) => (
                                <div key={`${item.category || 'rec'}-${index}`} className="border border-white/[0.03] bg-white/[0.01] p-3">
                                    <div className="text-[10px] font-bold text-white/60">{String(item.category || 'diversification')}</div>
                                    <div className="text-[9px] text-white/25 mt-1">{String(item.reason || 'Aucune raison fournie.')}</div>
                                </div>
                            ))
                        ) : (
                            <div className="text-[10px] text-white/25 border border-dashed border-white/10 p-3">Aucune recommandation disponible.</div>
                        )}
                    </div>
                </Section>
            </div>

            <div className="space-y-1 pt-2">
                <div className="text-[9px] uppercase tracking-[0.2em] text-matrix/40 px-1">AGENTS SUPPORT ({SUPPORT_AGENTS.length})</div>
            </div>
            <div className="space-y-2">
                {SUPPORT_AGENTS.map((agent) => (
                    <div key={agent.id} className="cyber-panel p-3 flex items-center justify-between gap-4">
                        <div>
                            <div className="text-[10px] font-bold text-white/60 tracking-[0.1em]">{agent.name}</div>
                            <div className="text-[9px] text-white/20 mt-1">{agent.description}</div>
                        </div>
                        <div className="flex items-center gap-2 shrink-0">
                            <span className="text-[8px] text-white/15 tracking-[0.15em]">{agent.codename}</span>
                            <PhasePill value={agent.phase} />
                            <HealthPill status={agentHealth[agent.id]?.status || 'offline'} />
                        </div>
                    </div>
                ))}
            </div>
        </div>
    )
}
