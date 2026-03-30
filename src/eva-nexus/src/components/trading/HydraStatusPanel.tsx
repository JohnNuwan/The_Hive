import { Network, Smartphone, TimerReset, Waypoints } from 'lucide-react'
import type {
    HydraAccountSnapshot,
    HydraAggregateResponse,
    HydraHealthSnapshot,
    HydraJobSnapshot,
    TradingNetworkStatus,
} from '../../services/api'
import { MetricPill, MetricTile, PanelShell, StatusBadge, compactList, formatDateLabel } from './TradingShared'

function resolveAccountHealth(
    account: HydraAccountSnapshot,
    health: HydraHealthSnapshot | undefined,
) {
    if (!account.active || !account.copy_enabled) {
        return { label: 'pause', tone: 'amber' as const }
    }
    if (account.quarantined_until) {
        return { label: 'quarantaine', tone: 'rose' as const }
    }
    if (!health?.process_alive) {
        return { label: 'process down', tone: 'rose' as const }
    }
    if (!health?.mt5_connected) {
        return { label: 'mt5 offline', tone: 'amber' as const }
    }
    if (!health?.autotrading_enabled) {
        return { label: 'autotrading off', tone: 'amber' as const }
    }
    return { label: 'copie active', tone: 'emerald' as const }
}

function summarizeJobs(jobs: HydraJobSnapshot[]) {
    const executed = jobs.filter((job) => job.status === 'executed').length
    const failed = jobs.filter((job) => job.status === 'failed' || job.status === 'rejected').length
    const pending = jobs.filter((job) => job.status === 'pending' || job.status === 'dispatched').length
    return { executed, failed, pending }
}

export default function HydraStatusPanel({
    hydra,
    network,
}: {
    hydra: HydraAggregateResponse | null
    network: TradingNetworkStatus | null | undefined
}) {
    if (!hydra || !hydra.enabled) {
        return (
            <PanelShell title="Hydra" subtitle="Copy trading master/slaves" accent="violet">
                <div className="rounded-2xl border border-white/5 bg-white/[0.02] p-5 text-sm text-slate-400">
                    Hydra n&apos;est pas encore configuree sur ce noeud.
                </div>
            </PanelShell>
        )
    }

    const accounts = hydra.registry?.accounts || []
    const slaves = accounts.filter((account) => account.role === 'slave')
    const healthByAccount = new Map((hydra.health || []).map((item) => [item.account_id, item]))
    const metricsByAccount = new Map((hydra.metrics || []).map((item) => [String(item.account_id || ''), item]))
    const recentJobs = hydra.jobs || []
    const recentSummary = summarizeJobs(recentJobs)

    return (
        <PanelShell
            title="Hydra"
            subtitle="Master transitoire, slaves MT5 et replication fill-confirmed"
            accent="violet"
            aside={
                <div className="flex flex-wrap justify-end gap-2 max-w-[22rem]">
                    <StatusBadge label={hydra.mode || 'mode inconnu'} tone="violet" />
                    <StatusBadge
                        label={network?.wireguard_enabled ? 'WireGuard actif' : 'WireGuard a configurer'}
                        tone={network?.wireguard_enabled ? 'emerald' : 'amber'}
                    />
                    <StatusBadge
                        label={hydra.master?.mt5_connected ? 'Master connecte' : 'Master offline'}
                        tone={hydra.master?.mt5_connected ? 'emerald' : 'rose'}
                    />
                </div>
            }
        >
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
                <MetricTile
                    label="Master"
                    value={hydra.master?.login ? String(hydra.master.login) : 'local'}
                    meta={`Source ${hydra.master_source_id} | Serveur ${String(hydra.master?.server || '--')}`}
                    accent="violet"
                />
                <MetricTile
                    label="Slaves"
                    value={String(slaves.length)}
                    meta={`${hydra.registry.quarantined_accounts} en quarantaine | ${hydra.registry.masters} master`}
                    accent="sky"
                />
                <MetricTile
                    label="Latency"
                    value={hydra.summary?.average_latency_ms != null ? `${hydra.summary.average_latency_ms} ms` : '--'}
                    meta={`Executed ${hydra.summary.executed_jobs} | Failed ${hydra.summary.failed_jobs}`}
                    accent={hydra.summary.failed_jobs > 0 ? 'amber' : 'emerald'}
                />
                <MetricTile
                    label="VPN"
                    value={network?.wireguard_enabled ? 'WireGuard' : 'A configurer'}
                    meta={`Sous-reseau ${String(network?.private_subnet || '--')} | Endpoint ${String(network?.public_endpoint || '--')}`}
                    accent={network?.wireguard_enabled ? 'emerald' : 'amber'}
                />
            </div>

            <div className="mt-4 grid grid-cols-1 xl:grid-cols-[minmax(0,1.2fr)_minmax(300px,0.8fr)] gap-4">
                <div className="space-y-3">
                    <div className="rounded-2xl border border-white/5 bg-black/20 p-4">
                        <div className="flex items-center gap-2 text-[9px] text-slate-500 uppercase font-black tracking-[0.2em]">
                            <Waypoints size={14} className="text-violet-300" />
                            Comptes slaves
                        </div>
                        <div className="mt-3 space-y-3 max-h-[20rem] overflow-y-auto pr-2 custom-scrollbar">
                            {slaves.length === 0 ? (
                                <div className="rounded-xl border border-white/5 bg-white/[0.02] p-4 text-[11px] text-slate-400">
                                    Aucun compte esclave enregistre.
                                </div>
                            ) : slaves.map((account) => {
                                const health = healthByAccount.get(account.id)
                                const metrics = metricsByAccount.get(account.id)
                                const status = resolveAccountHealth(account, health)
                                return (
                                    <div key={account.id} className="rounded-2xl border border-white/5 bg-white/[0.02] p-4">
                                        <div className="flex items-start justify-between gap-3">
                                            <div>
                                                <div className="text-sm font-black text-white/90">{account.name}</div>
                                                <div className="mt-1 text-[10px] text-slate-500 uppercase font-black tracking-[0.18em]">
                                                    {account.broker} | login {account.login} | {account.scaling_mode} x{account.scaling_factor}
                                                </div>
                                            </div>
                                            <StatusBadge label={status.label} tone={status.tone} />
                                        </div>
                                        <div className="mt-3 grid grid-cols-2 gap-2">
                                            <MetricPill label="Executeur" value={account.executor_url || '--'} />
                                            <MetricPill label="Dernier statut" value={String(metrics?.last_status || '--')} />
                                            <MetricPill label="Latence" value={metrics?.last_latency_ms != null ? `${metrics.last_latency_ms} ms` : '--'} />
                                            <MetricPill label="Symboles" value={compactList(health?.symbols_available || [], 4)} />
                                        </div>
                                        {account.quarantined_until ? (
                                            <div className="mt-3 text-[11px] text-amber-200">
                                                Quarantaine jusqu&apos;a {formatDateLabel(account.quarantined_until)} | {account.quarantine_reason || 'motif non renseigne'}
                                            </div>
                                        ) : null}
                                    </div>
                                )
                            })}
                        </div>
                    </div>
                </div>

                <div className="space-y-4">
                    <div className="rounded-2xl border border-white/5 bg-black/20 p-4">
                        <div className="flex items-center gap-2 text-[9px] text-slate-500 uppercase font-black tracking-[0.2em]">
                            <TimerReset size={14} className="text-sky-300" />
                            Replication recente
                        </div>
                        <div className="mt-3 grid grid-cols-3 gap-2">
                            <MetricPill label="Executed" value={String(recentSummary.executed)} />
                            <MetricPill label="Failed" value={String(recentSummary.failed)} />
                            <MetricPill label="Pending" value={String(recentSummary.pending)} />
                        </div>
                        <div className="mt-3 space-y-2 max-h-[14rem] overflow-y-auto pr-2 custom-scrollbar">
                            {recentJobs.length === 0 ? (
                                <div className="rounded-xl border border-white/5 bg-white/[0.02] p-3 text-[11px] text-slate-400">
                                    Aucun job Hydra disponible.
                                </div>
                            ) : recentJobs.slice(0, 8).map((job) => (
                                <div key={job.id} className="rounded-xl border border-white/5 bg-white/[0.02] px-3 py-2">
                                    <div className="flex items-center justify-between gap-2">
                                        <div className="text-[11px] font-black text-white/85">{job.symbol} | {job.action}</div>
                                        <StatusBadge
                                            label={job.status}
                                            tone={job.status === 'executed' ? 'emerald' : job.status === 'failed' || job.status === 'rejected' ? 'rose' : 'amber'}
                                        />
                                    </div>
                                    <div className="mt-1 text-[10px] text-slate-500">
                                        ticket maitre {job.source_ticket} | volume {job.volume} | latence {job.latency_ms != null ? `${job.latency_ms} ms` : '--'}
                                    </div>
                                    {job.error_message ? (
                                        <div className="mt-1 text-[10px] text-rose-300">{job.error_message}</div>
                                    ) : null}
                                </div>
                            ))}
                        </div>
                    </div>

                    <div className="rounded-2xl border border-white/5 bg-black/20 p-4">
                        <div className="flex items-center gap-2 text-[9px] text-slate-500 uppercase font-black tracking-[0.2em]">
                            <Network size={14} className="text-emerald-300" />
                            Mobile et acces prive
                        </div>
                        <div className="mt-3 text-[11px] text-slate-300 leading-relaxed">
                            Nexus PWA reste le cockpit V1. Le telephone passe par WireGuard puis lit Hydra, risque, drawdown et jobs recents sans exposer les APIs sensibles sur Internet.
                        </div>
                        <div className="mt-3 flex items-center gap-2 text-[10px] text-slate-400">
                            <Smartphone size={14} className="text-violet-300" />
                            Endpoint prive: {String(network?.public_endpoint || '--')}
                        </div>
                    </div>
                </div>
            </div>
        </PanelShell>
    )
}
