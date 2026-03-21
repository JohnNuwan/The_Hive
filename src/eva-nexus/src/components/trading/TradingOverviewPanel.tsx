import { AlertTriangle, Cpu, ShieldAlert, Wallet } from 'lucide-react'
import type { NemesisStatus, TradingAccountStatus, TradingRiskStatus, TradingPosition } from '../../services/api'
import { MetricTile, PanelShell, StatusBadge, formatPercent, formatUsd } from './TradingShared'

function vllmBadgeLabel(vllmState: string) {
    if (vllmState === 'stopped_for_training') return 'vLLM arrete pour entrainement'
    if (vllmState === 'online') return 'vLLM online'
    if (vllmState === 'running') return 'vLLM running'
    if (vllmState === 'offline') return 'vLLM offline'
    return `vLLM ${String(vllmState || 'unknown')}`
}

export default function TradingOverviewPanel({
    bankerStatus,
    labStatus,
    trainerState,
    vllmState,
    account,
    risk,
    positions,
    realizedPnl,
    openPnl,
    netPnl,
    closedTrades,
    winRate,
    nemesis,
}: {
    bankerStatus: 'online' | 'offline' | 'unknown'
    labStatus: 'online' | 'offline' | 'degraded'
    trainerState: string
    vllmState: string
    account: TradingAccountStatus
    risk: TradingRiskStatus
    positions: TradingPosition[]
    realizedPnl: number
    openPnl: number
    netPnl: number
    closedTrades: number
    winRate: number
    nemesis: NemesisStatus | null
}) {
    const riskBlocked = !risk.trading_allowed
    const nemesisActive = Boolean(nemesis?.trading_blocked)

    return (
        <PanelShell
            title="Live Trading"
            subtitle="Compte live, etat banker, protection du capital"
            accent="sky"
            aside={
                <div className="flex flex-wrap justify-end gap-2 max-w-[20rem]">
                    <StatusBadge label={`Banker ${bankerStatus}`} tone={bankerStatus === 'online' ? 'emerald' : 'rose'} />
                    <StatusBadge label={`Lab ${labStatus}`} tone={labStatus === 'online' ? 'emerald' : 'amber'} />
                    <StatusBadge label={vllmBadgeLabel(vllmState)} tone={vllmState === 'online' ? 'emerald' : vllmState === 'stopped_for_training' ? 'amber' : 'rose'} />
                    <StatusBadge label={`Trainer ${String(trainerState || 'idle')}`} tone={trainerState === 'running' ? 'violet' : 'slate'} />
                    {riskBlocked ? <StatusBadge label="Banker en pause risque" tone="amber" /> : null}
                    {nemesisActive ? <StatusBadge label="Nemesis active" tone="rose" /> : null}
                </div>
            }
        >
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
                <MetricTile
                    label="Equity"
                    value={`${Number(account.equity || 0).toLocaleString('fr-FR', { maximumFractionDigits: 2 })} ${account.currency || 'USD'}`}
                    meta={`Balance ${formatUsd(account.balance || 0)} | Marge libre ${formatUsd(account.free_margin || 0)}`}
                    accent="sky"
                />
                <MetricTile
                    label="PnL reel 7j"
                    value={formatUsd(realizedPnl)}
                    meta={`Trades fermes ${closedTrades} | Win rate reel ${formatPercent(winRate, 1)}`}
                    accent={realizedPnl >= 0 ? 'emerald' : 'rose'}
                />
                <MetricTile
                    label="PnL latent"
                    value={formatUsd(openPnl)}
                    meta={`Net combine ${formatUsd(netPnl)} | Positions ouvertes ${positions.length}`}
                    accent={openPnl >= 0 ? 'emerald' : 'amber'}
                />
                <MetricTile
                    label="Kill-switch"
                    value={riskBlocked ? 'BLOQUE' : 'SECURISE'}
                    meta={`Drawdown journalier ${formatPercent(risk.daily_drawdown_percent || 0)} | Limite active ${riskBlocked ? 'oui' : 'non'}`}
                    accent={riskBlocked ? 'rose' : 'emerald'}
                />
            </div>

            <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="rounded-2xl border border-white/5 bg-black/20 p-4 flex items-center gap-3">
                    <Wallet className="text-sky-400" size={16} />
                    <div>
                        <div className="text-[9px] text-slate-500 uppercase font-black tracking-[0.2em]">Compte</div>
                        <div className="text-sm font-black text-white/90">Levier {account.leverage || 0} | {account.currency || 'USD'}</div>
                    </div>
                </div>
                <div className="rounded-2xl border border-white/5 bg-black/20 p-4 flex items-center gap-3">
                    <Cpu className="text-violet-400" size={16} />
                    <div>
                        <div className="text-[9px] text-slate-500 uppercase font-black tracking-[0.2em]">Trainer</div>
                        <div className="text-sm font-black text-white/90">{String(trainerState || 'idle').toUpperCase()}</div>
                    </div>
                </div>
                <div className="rounded-2xl border border-white/5 bg-black/20 p-4 flex items-center gap-3">
                    {nemesisActive ? <AlertTriangle className="text-rose-400" size={16} /> : <ShieldAlert className="text-emerald-400" size={16} />}
                    <div>
                        <div className="text-[9px] text-slate-500 uppercase font-black tracking-[0.2em]">Protection</div>
                        <div className="text-sm font-black text-white/90">
                            {nemesisActive ? 'Nemesis bloque les nouvelles entrees' : riskBlocked ? 'Pause risque active' : 'Entrees autorisees'}
                        </div>
                    </div>
                </div>
            </div>
        </PanelShell>
    )
}

