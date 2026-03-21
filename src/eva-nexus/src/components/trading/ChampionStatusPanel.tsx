import type { HorizonChampionStatus, LabChampionStatus } from '../../services/api'
import { MetricPill, PanelShell, StatusBadge } from './TradingShared'

interface ChampionOfflineMetrics {
    win_rate?: number
    return_pct?: number
    profit_factor?: number
    expectancy_pct?: number
    max_drawdown_pct?: number
    positive_episode_rate?: number
    total_trades?: number
}

function selectionTone(selection?: string) {
    if (selection === 'champion' || selection === 'legacy_champion') return 'emerald'
    if (selection?.startsWith('blocked')) return 'rose'
    if (selection === 'latest' || selection === 'checkpoint_preview') return 'amber'
    return 'slate'
}

function outcomeLabel(status?: HorizonChampionStatus) {
    const report = status?.arena_report as Record<string, unknown> | null | undefined
    const battle = (report?.battle_report || {}) as Record<string, unknown>
    return String(battle.outcome || 'INCONNU')
}

function metricValue(value?: number, suffix = '') {
    if (typeof value !== 'number') return '--'
    return `${value.toFixed(2)}${suffix}`
}

export default function ChampionStatusPanel({ championStatus }: { championStatus: LabChampionStatus | null }) {
    const horizons = ['scalp', 'intraday', 'swing']
    const engineEntries = championStatus?.engines
        ? Object.entries(championStatus.engines).flatMap(([engine, engineHorizons]) =>
            horizons.map((horizon) => ({
                engine,
                horizon,
                status: engineHorizons?.[horizon],
            }))
        )
        : horizons.map((horizon) => ({
            engine: 'muzero',
            horizon,
            status: championStatus?.horizons?.[horizon],
        }))

    return (
        <PanelShell
            title="Champions"
            subtitle="Candidat, champion live et gate de promotion par moteur"
            accent="cyan"
            aside={<StatusBadge label={String(championStatus?.selection_policy || 'champion_only')} tone="cyan" />}
        >
            <div className="space-y-4">
                {engineEntries.map(({ engine, horizon, status }) => {
                    const gate = status?.promotion_gate
                    const metrics = (gate?.metrics || {}) as ChampionOfflineMetrics
                    const liveUniverse = status?.live_universe
                    return (
                        <div key={`${engine}-${horizon}`} className="rounded-2xl border border-white/5 bg-black/20 p-4">
                            <div className="flex items-start justify-between gap-3">
                                <div>
                                    <div className="text-[9px] text-slate-500 uppercase font-black tracking-[0.2em]">{engine} | {horizon}</div>
                                    <div className="mt-1 text-sm font-black text-white/90">Live: {status?.live_champion_id || 'Aucun'}</div>
                                    <div className="mt-1 text-[10px] text-slate-500">Candidat: {status?.candidate_id || 'Aucun'}</div>
                                    <div className="mt-1 text-[10px] text-slate-500">Registre: {status?.registry_champion_id || 'Aucun'}</div>
                                    <div className="mt-1 text-[10px] text-slate-500">Famille: {status?.family || '--'}</div>
                                </div>
                                <div className="flex flex-col items-end gap-2">
                                    <StatusBadge label={String(status?.selection || 'aucun')} tone={selectionTone(status?.selection) as any} />
                                    <StatusBadge label={gate?.allowed ? 'deploy ok' : `bloque: ${String(gate?.reason || 'inconnu')}`} tone={gate?.allowed ? 'emerald' : 'rose'} />
                                </div>
                            </div>

                            <div className="mt-3 grid grid-cols-2 gap-2">
                                <MetricPill label="Win rate offline" value={metricValue(metrics.win_rate, '%')} />
                                <MetricPill label="Return offline" value={metricValue(metrics.return_pct, '%')} />
                                <MetricPill label="Profit factor" value={metricValue(metrics.profit_factor)} />
                                <MetricPill label="Expectancy" value={metricValue(metrics.expectancy_pct, '%')} />
                                <MetricPill label="Drawdown offline" value={metricValue(metrics.max_drawdown_pct, '%')} />
                                <MetricPill label="Episode positif" value={metricValue(metrics.positive_episode_rate, '%')} />
                                <MetricPill label="Trades evalues" value={typeof metrics.total_trades === 'number' ? String(metrics.total_trades) : '--'} />
                                <MetricPill label="Arena" value={outcomeLabel(status)} />
                                <MetricPill label="Univers live" value={liveUniverse?.count ? `${liveUniverse.count} | ${liveUniverse.source}` : '--'} />
                                <MetricPill label="Fichier live" value={status?.live_checkpoint?.path ? String(status.live_checkpoint.path).split(/[/\\]/).pop() || '--' : '--'} />
                            </div>
                        </div>
                    )
                })}
            </div>
        </PanelShell>
    )
}
