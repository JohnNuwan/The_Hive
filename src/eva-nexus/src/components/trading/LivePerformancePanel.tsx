import type { ModelPerformanceReport } from '../../services/api'
import { MetricPill, MetricTile, PanelShell, formatPercent, formatUsd } from './TradingShared'

export default function LivePerformancePanel({
    modelPerformance,
    openPnl,
}: {
    modelPerformance: ModelPerformanceReport | null
    openPnl: number
}) {
    const summary = modelPerformance?.summary
    const realizedPnl = Number(summary?.realized_pnl ?? summary?.net_profit ?? 0)
    const netPnl = realizedPnl + Number(openPnl || 0)
    const recentTrades = modelPerformance?.recent_trades || []

    return (
        <PanelShell
            title="Performance reelle"
            subtitle={`Fenetre ${modelPerformance?.window_days || 7} jours | trades fermes MT5`}
            accent="emerald"
        >
            <div className="grid grid-cols-2 gap-3 mb-4">
                <MetricTile label="Reel" value={formatUsd(realizedPnl)} accent={realizedPnl >= 0 ? 'emerald' : 'rose'} />
                <MetricTile label="Latent" value={formatUsd(openPnl)} accent={openPnl >= 0 ? 'emerald' : 'amber'} />
                <MetricTile label="Net" value={formatUsd(netPnl)} accent={netPnl >= 0 ? 'emerald' : 'rose'} />
                <MetricTile label="Win rate reel" value={formatPercent(summary?.win_rate || 0, 1)} accent={Number(summary?.win_rate || 0) >= 50 ? 'emerald' : 'amber'} />
            </div>

            <div className="grid grid-cols-2 gap-2 mb-4">
                <MetricPill label="Trades" value={String(summary?.closed_trades || 0)} />
                <MetricPill label="Periodes" value={`${summary?.from || '--'} -> ${summary?.to || '--'}`} />
            </div>

            <div>
                <div className="text-[9px] text-slate-500 uppercase font-black tracking-[0.2em] mb-2">Derniers trades fermes</div>
                <div className="space-y-2 max-h-52 overflow-y-auto pr-2 custom-scrollbar">
                    {recentTrades.length === 0 ? (
                        <div className="rounded-xl border border-white/5 bg-white/[0.02] px-3 py-2 text-[10px] text-slate-400">
                            Aucun trade ferme sur la fenetre analysee.
                        </div>
                    ) : recentTrades.slice(0, 8).map((trade) => (
                        <div key={`${trade.position_id}-${trade.close_time || trade.symbol}`} className="rounded-xl border border-white/5 bg-white/[0.02] px-3 py-2">
                            <div className="flex items-center justify-between gap-3">
                                <div>
                                    <div className="text-[11px] font-black text-white/90">{trade.symbol} | {trade.label}</div>
                                    <div className="text-[10px] text-slate-500">{trade.family} | vol {trade.volume} | ferme {trade.close_time || '--'}</div>
                                </div>
                                <div className={`text-[11px] font-black ${trade.net_profit >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                                    {formatUsd(trade.net_profit)}
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </PanelShell>
    )
}
