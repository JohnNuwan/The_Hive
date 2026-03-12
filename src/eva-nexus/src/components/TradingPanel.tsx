import { useState, useEffect } from 'react'
import { Wallet, ArrowUpCircle, ShieldAlert, Activity } from 'lucide-react'
import {
    checkNodeHealth,
    createOrder,
    closePosition,
    getLabChampionStatus,
    getModelPerformance,
    getTradingStatus,
    toggleAutoTrading,
    type LabChampionStatus,
    type ModelPerformanceReport,
} from '../services/api'

export default function TradingPanel() {
    const [bankerStatus, setBankerStatus] = useState<'online' | 'offline' | 'unknown'>('unknown')
    const [tradingData, setTradingData] = useState<any>(null)
    const [championStatus, setChampionStatus] = useState<LabChampionStatus | null>(null)
    const [modelPerformance, setModelPerformance] = useState<ModelPerformanceReport | null>(null)

    useEffect(() => {
        const fetchData = async () => {
            try {
                const bankerHealth = await checkNodeHealth('Banker', '/api/banker/health')
                const currentStatus = bankerHealth.status === 'online' ? 'online' : 'offline'
                const [tradingResp, championResp, performanceResp] = await Promise.all([
                    currentStatus === 'online'
                        ? getTradingStatus()
                        : Promise.resolve({ account: {}, positions: [], risk: { daily_drawdown_percent: 0, trading_allowed: false }, decisions: {} }),
                    getLabChampionStatus(),
                    currentStatus === 'online'
                        ? getModelPerformance(7, 5)
                        : Promise.resolve(null),
                ])
                setBankerStatus(currentStatus)
                setTradingData(tradingResp)
                setChampionStatus(championResp)
                setModelPerformance(performanceResp)
            } catch (e) {
                console.error("Error fetching trading data:", e)
                setBankerStatus('offline')
            }
        }

        fetchData()
        const interval = setInterval(fetchData, 5000) // Refresh every 5s for trading
        return () => clearInterval(interval)
    }, [])

    const account = tradingData?.account || {}
    const positions = tradingData?.positions || []
    const risk = tradingData?.risk || {}


    // Calculation de P&L total latent
    const totalProfit = positions.reduce((acc: number, pos: any) => acc + parseFloat(pos.profit || 0), 0)
    const realizedProfit = Number(modelPerformance?.summary?.net_profit || 0)
    const totalPnl = realizedProfit + totalProfit
    const pnlWindowDays = Number(modelPerformance?.window_days || 7)

    const [orderForm, setOrderForm] = useState({ symbol: 'XAUUSD', volume: 0.01, sl: 2000.0, tp: 2100.0 })
    const [actionLoading, setActionLoading] = useState(false)
    const [autoTrading, setAutoTrading] = useState(true)

    const handleAutoToggle = async () => {
        const newState = !autoTrading
        setAutoTrading(newState)
        await toggleAutoTrading(newState)
    }

    const handleOrder = async (action: 'BUY' | 'SELL') => {
        setActionLoading(true)
        await createOrder({
            symbol: orderForm.symbol,
            action,
            volume: orderForm.volume,
            stop_loss: orderForm.sl,
            take_profit: orderForm.tp
        })
        setActionLoading(false)
        // Refresh data immediately
        if (bankerStatus === 'online') {
            setTradingData(await getTradingStatus())
        }
    }

    const handleClose = async (ticket: string) => {
        if (!confirm('Close Position #' + ticket + '?')) return
        await closePosition(ticket)
        // Refresh
        if (bankerStatus === 'online') {
            setTradingData(await getTradingStatus())
        }
    }

    const handleHold = async () => {
        // HOLD Action = Close ALL positions for this symbol (Flatten)
        if (!confirm(`HOLD COMMAND: Close ALL ${orderForm.symbol} positions?`)) return
        setActionLoading(true)
        const targetPositions = positions.filter((p: any) => p.symbol === orderForm.symbol)
        await Promise.all(targetPositions.map((p: any) => closePosition(p.ticket)))
        setActionLoading(false)
    }

    return (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-full animate-fade-in">
            {/* Portfolio Overview */}
            <div className="lg:col-span-2 flex flex-col gap-6">
                {/* ... existing StatCards ... */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <StatCard
                        title="Equity Total"
                        value={`${parseFloat(account.equity || 0).toLocaleString()} ${account.currency || '$'}`}
                        icon={<Wallet className="text-sky-400" />}
                        glow="sky"
                    />
                    <StatCard
                        title="PnL 7j + latent"
                        value={`${totalPnl >= 0 ? '+' : ''}${totalPnl.toFixed(2)} $`}
                        icon={<ArrowUpCircle className={totalPnl >= 0 ? "text-emerald-400" : "text-red-400"} />}
                        trend={account.balance ? `${((totalPnl / parseFloat(account.balance)) * 100).toFixed(2)}%` : "0%"}
                        meta={`${pnlWindowDays}j realise ${formatUsd(realizedProfit)} | Ouvert ${formatUsd(totalProfit)}`}
                        glow={totalPnl >= 0 ? "emerald" : "amber"}
                    />
                    {/* Dynamic Drawdown Gauge */}
                    <DrawdownGauge
                        drawdown={parseFloat(risk.daily_drawdown_percent || 0)}
                        limit={4.0}
                        isLocked={!risk.trading_allowed}
                    />
                </div>

                {/* MANUAL TRADING TERMINAL */}
                <div className="glass rounded-[2rem] p-6 border border-white/5 shadow-2xl relative overflow-hidden">
                    <div className="absolute top-0 right-0 w-32 h-32 bg-indigo-500/10 blur-3xl rounded-full" />

                    <div className="flex justify-between items-center mb-4 relative z-10">
                        <h4 className="font-black text-sm text-indigo-300 uppercase tracking-[0.3em]">Manual Override Terminal</h4>
                        <div className="flex items-center gap-3 bg-black/20 px-3 py-1.5 rounded-full border border-white/5">
                            <span className={`text-[9px] font-black uppercase tracking-widest transition-colors ${autoTrading ? 'text-emerald-400 drop-shadow-[0_0_8px_rgba(52,211,153,0.5)]' : 'text-slate-500'}`}>
                                {autoTrading ? 'Auto-Pilot Active' : 'Auto-Pilot Off'}
                            </span>
                            <button
                                onClick={handleAutoToggle}
                                className={`w-8 h-4 rounded-full p-0.5 transition-all duration-300 ${autoTrading ? 'bg-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.3)]' : 'bg-slate-700'}`}
                            >
                                <div className={`w-3 h-3 bg-white rounded-full shadow-md transform transition-transform duration-300 ${autoTrading ? 'translate-x-4' : 'translate-x-0'}`} />
                            </button>
                        </div>
                    </div>

                    <div className="flex flex-wrap items-end gap-4">
                        <label className="text-[9px] text-white/40 font-bold uppercase tracking-wider block mb-1">Symbol</label>
                        <select
                            value={orderForm.symbol}
                            onChange={e => setOrderForm({ ...orderForm, symbol: e.target.value })}
                            className="bg-black/40 border border-white/10 rounded-lg px-3 py-2 text-xs font-mono font-bold w-24 focus:border-indigo-500 outline-none appearance-none cursor-pointer"
                        >
                            {tradingData?.decisions && Object.keys(tradingData.decisions).length > 0 ? (
                                Object.keys(tradingData.decisions).map(sym => (
                                    <option key={sym} value={sym}>{sym}</option>
                                ))
                            ) : (
                                <>
                                    <option value="XAUUSD">XAUUSD</option>
                                    <option value="EURUSD">EURUSD</option>
                                    <option value="BTCUSD">BTCUSD</option>
                                    <option value="US30">US30</option>
                                </>
                            )}
                        </select>
                        <div>
                            <label className="text-[9px] text-white/40 font-bold uppercase tracking-wider block mb-1">Volume</label>
                            <input
                                type="number"
                                step="0.01"
                                value={orderForm.volume}
                                onChange={e => setOrderForm({ ...orderForm, volume: parseFloat(e.target.value) })}
                                className="bg-black/40 border border-white/10 rounded-lg px-3 py-2 text-xs font-mono font-bold w-20 focus:border-indigo-500 outline-none"
                            />
                        </div>
                        <div>
                            <label className="text-[9px] text-white/40 font-bold uppercase tracking-wider block mb-1">Stop Loss</label>
                            <input
                                type="number"
                                value={orderForm.sl}
                                onChange={e => setOrderForm({ ...orderForm, sl: parseFloat(e.target.value) })}
                                className="bg-black/40 border border-white/10 rounded-lg px-3 py-2 text-xs font-mono font-bold w-24 focus:border-indigo-500 outline-none"
                            />
                        </div>

                        <div className="flex gap-2 ml-auto">
                            <button
                                onClick={() => handleOrder('BUY')}
                                disabled={actionLoading}
                                className="bg-emerald-500/10 border border-emerald-500/40 text-emerald-400 px-6 py-2 rounded-xl text-xs font-black uppercase tracking-widest hover:bg-emerald-500 hover:text-black transition-all shadow-[0_0_15px_rgba(16,185,129,0.2)]"
                            >
                                BUY
                            </button>
                            <button
                                onClick={() => handleOrder('SELL')}
                                disabled={actionLoading}
                                className="bg-red-500/10 border border-red-500/40 text-red-400 px-6 py-2 rounded-xl text-xs font-black uppercase tracking-widest hover:bg-red-500 hover:text-black transition-all shadow-[0_0_15px_rgba(239,68,68,0.2)]"
                            >
                                SELL
                            </button>
                            <button
                                onClick={handleHold}
                                disabled={actionLoading}
                                className="bg-amber-500/10 border border-amber-500/40 text-amber-400 px-6 py-2 rounded-xl text-xs font-black uppercase tracking-widest hover:bg-amber-500 hover:text-black transition-all shadow-[0_0_15px_rgba(245,158,11,0.2)]"
                            >
                                HOLD
                            </button>
                        </div>
                    </div>
                </div>

                <div className="flex-grow glass rounded-[2.5rem] p-8 overflow-hidden flex flex-col shadow-2xl relative">
                    <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-sky-500/30 to-transparent" />
                    <div className="flex items-center justify-between mb-8">
                        <div>
                            <h4 className="font-black text-xl tracking-tight text-white/90">Positions de la Ruche</h4>
                            <p className="text-[10px] text-slate-500 uppercase font-bold tracking-[0.2em] mt-1">Surveillance Temps RÃ©el</p>
                        </div>
                        <div className="flex items-center gap-4">
                            <div className="flex items-center gap-2 px-4 py-2 bg-white/[0.03] rounded-xl border border-white/10 shadow-inner">
                                <Activity size={14} className={bankerStatus === 'online' ? 'text-emerald-500 animate-pulse' : 'text-red-500'} />
                                <span className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Banker Status: {bankerStatus}</span>
                            </div>
                        </div>
                    </div>

                    <div className="space-y-4 overflow-y-auto pr-2 custom-scrollbar">
                        {positions.length === 0 ? (
                            <div className="flex flex-col items-center justify-center h-48 opacity-20">
                                <Activity size={48} className="mb-4" />
                                <p className="text-xs font-black uppercase tracking-widest">Aucune mission active</p>
                            </div>
                        ) : positions.map((pos: any) => (
                            <div key={pos.ticket} className="p-5 rounded-[1.5rem] bg-white/[0.02] border border-white/5 flex items-center justify-between hover:bg-white/[0.04] hover:border-sky-500/30 transition-all duration-300 group">
                                <div className="flex items-center gap-5">
                                    <div className={`w-12 h-12 rounded-2xl flex items-center justify-center font-black text-xs shadow-lg transition-transform group-hover:scale-110 ${pos.action === 'BUY' ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' : 'bg-red-500/10 text-red-400 border border-red-500/20'}`}>
                                        {pos.action}
                                    </div>
                                    <div>
                                        <p className="font-black text-base tracking-tight text-white/90">{pos.symbol}</p>
                                        <p className="text-[10px] text-slate-500 font-bold uppercase tracking-widest mt-1">Volume: {pos.volume} â€¢ Ticket #{pos.ticket}</p>
                                    </div>
                                </div>
                                <div className="text-right">
                                    <p className={`font-black text-lg tracking-tight ${parseFloat(pos.profit) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                        {parseFloat(pos.profit) >= 0 ? '+' : ''}{pos.profit} $
                                    </p>
                                    <button onClick={() => handleClose(pos.ticket)} className="text-[9px] text-slate-600 hover:text-red-400 mt-2 uppercase font-black tracking-[0.2em] transition-colors cursor-pointer">Terminer Mission</button>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* Account Info & Hydra */}
            <div className="flex flex-col gap-6">
                <ChampionPanel championStatus={championStatus} />
                <PerformancePanel modelPerformance={modelPerformance} />
                <div className="glass rounded-[2rem] p-8 border-l-[6px] border-l-indigo-600 shadow-xl">
                    <h4 className="text-[10px] font-black text-indigo-400 uppercase mb-6 tracking-[0.3em]">Flux Prop-Firm Hydra</h4>
                    <div className="space-y-5">
                        <AccountRow label="FTMO Performance" status="Optimum" color="emerald" />
                        <AccountRow label="FundedPlus Alpha" status="Optimum" color="emerald" />
                        <AccountRow label="Apex Prime Omega" status="Synced" color="sky" />
                    </div>
                </div>

                <div className="glass rounded-[2rem] p-8 grow shadow-xl relative overflow-hidden">
                    <div className="absolute top-0 right-0 w-24 h-24 bg-sky-500/5 blur-3xl rounded-full" />
                    <h4 className="font-black text-sm text-white/80 mb-6 uppercase tracking-widest">Score de SincÃ©ritÃ© Cognitive</h4>
                    <div className="mt-4">
                        <SincerityGauge score={tradingData?.last_sincerity_score || 0.85} />
                    </div>
                    <div className="mt-8 space-y-6">
                        {tradingData?.decisions ? (
                            Object.entries(tradingData.decisions).map(([symbol, data]: [string, any]) => (
                                <SentimentItem
                                    key={symbol}
                                    symbol={symbol}
                                    price={data.price}
                                    action={data.action}
                                    rsi={data.rsi}
                                    vwap={data.vwap}
                                    adx={data.adx}
                                    cortex_bias={data.cortex_bias}
                                    gnn_bias={data.gnn_bias}
                                    comment={data.comment}
                                />
                            ))
                        ) : (
                            <div className="text-center text-xs text-slate-500 font-bold uppercase tracking-widest">
                                En attente du Cerveau...
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    )
}


function ChampionPanel({ championStatus }: { championStatus: LabChampionStatus | null }) {
    const horizons = ['intraday', 'scalp', 'swing']
    const nightlyStatus = String(championStatus?.nightly_summary?.status || 'unknown').toUpperCase()
    const selectionPolicy = championStatus?.selection_policy || 'champion_only'
    const liveEngine = championStatus?.horizons?.intraday?.engine_label || championStatus?.dreamer_gate?.engine || 'RSI Heuristic'
    const activeIntradaySelection = championStatus?.horizons?.intraday?.selection || 'none'

    return (
        <div className="glass rounded-[2rem] p-6 border border-cyan-500/20 shadow-xl relative overflow-hidden">
            <div className="absolute top-0 right-0 w-24 h-24 bg-cyan-500/10 blur-3xl rounded-full" />
            <div className="flex items-start justify-between gap-4 mb-5 relative z-10">
                <div>
                    <h4 className="text-[10px] font-black text-cyan-300 uppercase tracking-[0.3em]">Model Champions</h4>
                    <p className="text-[10px] text-slate-500 uppercase font-bold tracking-[0.18em] mt-2">
                        Politique live: {selectionPolicy}
                    </p>
                </div>
                <div className="text-right">
                    <div className="text-[9px] text-slate-500 uppercase font-black tracking-[0.2em]">Moteur live</div>
                    <div className="text-xs font-black text-cyan-300 mt-1">{liveEngine}</div>
                </div>
            </div>

            <div className="grid grid-cols-2 gap-3 mb-5 relative z-10">
                <ChampionStat
                    label="Inference"
                    value={activeIntradaySelection === 'champion' || activeIntradaySelection === 'legacy_champion' ? 'Champion' : 'Fallback'}
                    accent={activeIntradaySelection === 'champion' || activeIntradaySelection === 'legacy_champion' ? 'emerald' : 'amber'}
                />
                <ChampionStat
                    label="Nightly"
                    value={nightlyStatus}
                    accent={nightlyStatus === 'OK' || nightlyStatus === 'RUNNING' ? 'emerald' : 'amber'}
                />
            </div>

            <div className="space-y-3 relative z-10">
                {horizons.map((horizon) => {
                    const status = championStatus?.horizons?.[horizon]
                    const perf = championStatus?.performance_summary?.[horizon]
                    const gate = status?.promotion_gate
                    const liveUniverse = status?.live_universe
                    const arenaReport = status?.arena_report as Record<string, any> | undefined
                    const battle = arenaReport?.battle_report as Record<string, any> | undefined
                    const outcome = String(battle?.outcome || 'UNKNOWN')
                    const checkpointName = formatArtifactName(status?.live_checkpoint?.path)
                    const winRateValue = typeof gate?.metrics?.win_rate === 'number'
                        ? `${gate.metrics.win_rate.toFixed(1)}%`
                        : (typeof perf?.win_rate === 'number' ? `${perf.win_rate.toFixed(1)}%` : '--')
                    const returnValue = typeof gate?.metrics?.return_pct === 'number'
                        ? `${gate.metrics.return_pct.toFixed(2)}%`
                        : (typeof perf?.return_pct === 'number' ? `${perf.return_pct.toFixed(2)}%` : '--')
                    const profitFactorValue = typeof gate?.metrics?.profit_factor === 'number'
                        ? gate.metrics.profit_factor.toFixed(2)
                        : '--'
                    const gateValue = gate?.allowed ? 'DEPLOY OK' : 'BLOQUE'
                    const tradesValue = typeof gate?.metrics?.total_trades === 'number'
                        ? String(gate.metrics.total_trades)
                        : '--'
                    const universeValue = liveUniverse?.count
                        ? `${liveUniverse.count} | ${liveUniverse.source}`
                        : '--'
                    const gateReasonValue = gate?.allowed ? 'aucun' : String(gate?.reason || 'inconnu')
                    const liveChampionValue = status?.live_champion_id || 'Aucun live'
                    const arenaChampionValue = status?.registry_champion_id || status?.champion_id || 'Aucun arena'
                    const candidateValue = status?.candidate_id || 'Aucun challenger'
                    return (
                        <div key={horizon} className="rounded-2xl bg-black/30 border border-white/5 p-4">
                            <div className="flex items-center justify-between gap-3">
                                <div>
                                    <div className="text-[10px] text-slate-500 uppercase font-black tracking-[0.2em]">{horizon}</div>
                                    <div className="text-sm font-black text-white/90 mt-1">
                                        {liveChampionValue}
                                    </div>
                                    <div className="text-[10px] text-slate-500 mt-2">
                                        Arena: {arenaChampionValue}
                                    </div>
                                    <div className="text-[10px] text-slate-500 mt-1">
                                        Challenger: {candidateValue}
                                    </div>
                                </div>
                                <span className={`text-[9px] px-2 py-1 rounded-lg border font-black uppercase tracking-[0.2em] ${selectionBadgeClass(status?.selection)}`}>
                                    {status?.selection || 'none'}
                                </span>
                            </div>

                            <div className="mt-3 grid grid-cols-2 gap-2 text-[10px]">
                                <MetricPill label="Win Rate" value={winRateValue} />
                                <MetricPill label="Return" value={returnValue} />
                                <MetricPill label="Profit Factor" value={profitFactorValue} />
                                <MetricPill label="Deploy" value={gateValue} />
                                <MetricPill label="Trades" value={tradesValue} />
                                <MetricPill label="Univers" value={universeValue} />
                                <MetricPill label="Blocage" value={gateReasonValue} />
                                <MetricPill label="Arena" value={outcome} />
                                <MetricPill label="Live File" value={checkpointName} />
                            </div>
                        </div>
                    )
                })}
            </div>
        </div>
    )
}

function PerformancePanel({ modelPerformance }: { modelPerformance: ModelPerformanceReport | null }) {
    const summary = modelPerformance?.summary
    const topModel = modelPerformance?.by_model?.[0]
    const familyRows = modelPerformance?.by_family || []

    return (
        <div className="glass rounded-[2rem] p-6 border border-emerald-500/20 shadow-xl relative overflow-hidden">
            <div className="absolute top-0 right-0 w-24 h-24 bg-emerald-500/10 blur-3xl rounded-full" />
            <div className="flex items-start justify-between gap-4 mb-5 relative z-10">
                <div>
                    <h4 className="text-[10px] font-black text-emerald-300 uppercase tracking-[0.3em]">Performance Live</h4>
                    <p className="text-[10px] text-slate-500 uppercase font-bold tracking-[0.18em] mt-2">
                        Trades fermes sur {modelPerformance?.window_days || 7} jours
                    </p>
                </div>
                <div className={`text-right ${summary && summary.net_profit >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                    <div className="text-[9px] text-slate-500 uppercase font-black tracking-[0.2em]">PnL realise</div>
                    <div className="text-lg font-black mt-1">{formatUsd(summary?.net_profit || 0)}</div>
                </div>
            </div>

            <div className="grid grid-cols-3 gap-3 mb-5 relative z-10">
                <ChampionStat
                    label="Trades"
                    value={String(summary?.closed_trades || 0)}
                    accent={(summary?.closed_trades || 0) > 0 ? 'emerald' : 'amber'}
                />
                <ChampionStat
                    label="Win Rate"
                    value={`${Number(summary?.win_rate || 0).toFixed(1)}%`}
                    accent={Number(summary?.win_rate || 0) >= 50 ? 'emerald' : 'amber'}
                />
                <ChampionStat
                    label="Top Modele"
                    value={topModel?.label || 'Aucun'}
                    accent={topModel && topModel.net_profit >= 0 ? 'emerald' : 'amber'}
                />
            </div>

            <div className="space-y-3 relative z-10">
                {familyRows.length === 0 ? (
                    <div className="rounded-2xl bg-black/30 border border-white/5 p-4 text-[11px] font-black text-slate-500 uppercase tracking-[0.18em]">
                        Aucun trade ferme sur la fenetre analysee
                    </div>
                ) : familyRows.map((row) => (
                    <div key={row.label} className="rounded-2xl bg-black/30 border border-white/5 p-4">
                        <div className="flex items-center justify-between gap-3">
                            <div>
                                <div className="text-sm font-black text-white/90">{row.label}</div>
                                <div className="text-[10px] text-slate-500 uppercase font-black tracking-[0.18em] mt-1">
                                    {row.closed_trades} trades | {row.symbols.join(', ') || '--'}
                                </div>
                            </div>
                            <div className={`text-right ${row.net_profit >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                <div className="text-sm font-black">{formatUsd(row.net_profit)}</div>
                                <div className="text-[10px] font-black mt-1">{row.win_rate.toFixed(1)}% win</div>
                            </div>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    )
}

function ChampionStat({ label, value, accent }: { label: string; value: string; accent: 'emerald' | 'amber' }) {
    const accentClass = accent === 'emerald' ? 'text-emerald-400 border-emerald-500/20 bg-emerald-500/5' : 'text-amber-400 border-amber-500/20 bg-amber-500/5'
    return (
        <div className={`rounded-xl border p-3 ${accentClass}`}>
            <div className="text-[9px] uppercase tracking-[0.2em] font-black opacity-70">{label}</div>
            <div className="text-sm font-black mt-1">{value}</div>
        </div>
    )
}

function MetricPill({ label, value }: { label: string; value: string }) {
    return (
        <div className="rounded-xl border border-white/5 bg-white/[0.02] px-3 py-2">
            <div className="text-[8px] text-slate-500 uppercase font-black tracking-[0.18em]">{label}</div>
            <div className="text-[11px] font-black text-white/80 mt-1 truncate">{value}</div>
        </div>
    )
}

function selectionBadgeClass(selection?: string) {
    if (selection === 'blocked_champion' || selection === 'blocked_legacy_champion') {
        return 'text-red-400 border-red-500/20 bg-red-500/10'
    }
    if (selection === 'champion' || selection === 'legacy_champion') {
        return 'text-emerald-400 border-emerald-500/20 bg-emerald-500/10'
    }
    if (selection === 'latest' || selection === 'checkpoint_preview') {
        return 'text-amber-400 border-amber-500/20 bg-amber-500/10'
    }
    return 'text-slate-400 border-white/10 bg-white/[0.03]'
}

function formatArtifactName(path?: string | null) {
    if (!path) return '--'
    const parts = path.split(/[/\\]/)
    return parts[parts.length - 1] || path
}

function formatUsd(value: number) {
    const amount = Number(value || 0)
    const prefix = amount > 0 ? '+' : ''
    return `${prefix}${amount.toFixed(2)}$`
}

function StatCard({ title, value, icon, trend, glow, meta }: { title: string, value: string, icon: any, trend?: string, glow: string, meta?: string }) {
    const glowClass = glow === 'sky' ? 'shadow-sky-500/20' : glow === 'emerald' ? 'shadow-emerald-500/20' : 'shadow-amber-500/20'
    const borderClass = glow === 'sky' ? 'border-sky-500/20' : glow === 'emerald' ? 'border-emerald-500/20' : 'border-amber-500/20'

    return (
        <div className={`glass p-7 rounded-[2rem] border ${borderClass} shadow-2xl relative overflow-hidden group hover:scale-[1.02] transition-transform duration-300`}>
            <div className="absolute top-0 right-0 w-16 h-16 bg-gradient-to-br from-white/5 to-transparent rounded-bl-[2rem]" />
            <div className="flex items-center justify-between mb-4">
                <p className="text-[10px] font-black text-slate-500 uppercase tracking-[0.2em]">{title}</p>
                <div className={`p-3 bg-white/[0.03] rounded-2xl shadow-inner ${glowClass}`}>{icon}</div>
            </div>
            <div className="flex items-baseline gap-3">
                <p className="text-2xl font-black text-white tracking-tight">{value}</p>
                {trend && (
                    <span className={`text-[10px] px-2 py-0.5 rounded-lg font-black tracking-widest ${trend.includes('+') ? 'bg-emerald-500/10 text-emerald-400' : trend === 'SECURE' ? 'bg-sky-500/10 text-sky-400' : 'bg-red-500/10 text-red-400'}`}>
                        {trend}
                    </span>
                )}
            </div>
            {meta && (
                <p className="mt-3 text-[9px] font-black text-slate-500 uppercase tracking-[0.16em]">
                    {meta}
                </p>
            )}
        </div>
    )
}

function AccountRow({ label, status, color }: { label: string, status: string, color: string }) {
    const glowClass = color === 'emerald' ? 'bg-emerald-500 shadow-emerald-500/50' : 'bg-sky-500 shadow-sky-500/50'
    return (
        <div className="flex items-center justify-between p-4 rounded-2xl bg-white/[0.02] border border-white/5 hover:bg-white/[0.04] transition-colors group">
            <span className="text-xs font-bold text-slate-300 group-hover:text-white transition-colors">{label}</span>
            <div className="flex items-center gap-3">
                <div className={`w-2 h-2 rounded-full ${glowClass} shadow-[0_0_8px] animate-pulse`}></div>
                <span className="text-[9px] font-black uppercase text-slate-500 tracking-widest">{status}</span>
            </div>
        </div>
    )
}

function SincerityGauge({ score }: { score: number }) {
    const color = score > 0.8 ? 'text-emerald-400' : score > 0.6 ? 'text-sky-400' : 'text-red-400'
    const barColor = score > 0.8 ? 'bg-emerald-500' : score > 0.6 ? 'bg-sky-500' : 'bg-red-500'

    return (
        <div className="flex flex-col items-center">
            <div className={`text-4xl font-black ${color} mb-2`}>{(score * 100).toFixed(0)}%</div>
            <div className="w-full h-2 bg-white/5 rounded-full overflow-hidden">
                <div
                    className={`h-full ${barColor} transition-all duration-1000 shadow-[0_0_15px] shadow-current`}
                    style={{ width: `${score * 100}%` }}
                />
            </div>
            <p className="text-[9px] text-slate-500 font-bold uppercase tracking-widest mt-3">Validation Neuronale Active</p>
        </div>
    )
}

function SentimentItem({ symbol, price, action, rsi, vwap, adx, cortex_bias, gnn_bias, comment }: { symbol: string, price: number, action: string, rsi: number, vwap?: number, adx?: number, cortex_bias?: string, gnn_bias?: string, comment: string }) {
    const isBull = action.includes('BUY')
    const isBear = action.includes('SELL')

    const barColor = isBull ? 'bg-gradient-to-r from-emerald-600 to-emerald-400' : isBear ? 'bg-gradient-to-r from-red-600 to-red-400' : 'bg-gradient-to-r from-slate-600 to-slate-400'
    const textColor = isBull ? 'text-emerald-400' : isBear ? 'text-red-400' : 'text-slate-400'

    // RSI visual position (0-100)
    const width = Math.min(Math.max(rsi, 0), 100)

    // Bias Badge Coloring
    const getBiasColor = (bias?: string) => {
        if (!bias) return 'text-slate-500 bg-slate-500/10 border-slate-500/20'
        if (bias.includes('BULL')) return 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20'
        if (bias.includes('BEAR')) return 'text-red-400 bg-red-500/10 border-red-500/20'
        return 'text-sky-400 bg-sky-500/10 border-sky-500/20'
    }

    return (
        <div className="space-y-4 group p-4 rounded-3xl bg-black/40 hover:bg-white/[0.04] transition-all border border-white/5 hover:border-white/10 shadow-lg relative overflow-hidden">
            {/* Background Glow */}
            <div className={`absolute top-0 right-0 w-32 h-32 blur-[50px] rounded-full opacity-20 pointer-events-none ${isBull ? 'bg-emerald-500' : isBear ? 'bg-red-500' : 'bg-slate-500'}`} />

            <div className="flex justify-between items-start relative z-10">
                <div className="flex flex-col gap-1">
                    <div className="flex items-center gap-2">
                        <span className="text-white font-black text-lg tracking-tight">{symbol}</span>
                        <span className="text-slate-400 font-mono text-xs">{price.toFixed(2)} $</span>
                    </div>

                    {/* Multi-Agent Judgement Matrix */}
                    <div className="flex items-center gap-2 mt-2">
                        {cortex_bias && (
                            <div className={`px-2 py-0.5 rounded-md border text-[8px] font-black uppercase tracking-widest ${getBiasColor(cortex_bias)}`}>
                                CORTEX: {cortex_bias}
                            </div>
                        )}
                        {gnn_bias && (
                            <div className={`px-2 py-0.5 rounded-md border text-[8px] font-black uppercase tracking-widest ${getBiasColor(gnn_bias)}`}>
                                GNN: {gnn_bias}
                            </div>
                        )}
                    </div>
                </div>

                <div className="text-right flex flex-col items-end">
                    <div className={`${textColor} bg-black/60 px-4 py-1.5 rounded-xl border border-white/10 shadow-[0_0_15px_rgba(0,0,0,0.5)] font-black uppercase tracking-widest text-xs flex items-center gap-2`}>
                        {action !== 'WAIT' && <div className={`w-1.5 h-1.5 rounded-full animate-pulse ${isBull ? 'bg-emerald-500 shadow-[0_0_8px_#10b981]' : 'bg-red-500 shadow-[0_0_8px_#ef4444]'}`} />}
                        {action}
                    </div>
                    <span className="text-[9px] text-slate-500 font-bold uppercase tracking-widest mt-2 block max-w-[120px] truncate" title={comment}>
                        {comment}
                    </span>
                </div>
            </div>

            {/* Advanced Metrics Dash */}
            <div className="grid grid-cols-2 gap-3 relative z-10">
                <div className="bg-white/[0.02] border border-white/5 rounded-xl p-2 flex flex-col">
                    <span className="text-[8px] text-slate-500 uppercase font-black tracking-widest">VWAP (Institutional Level)</span>
                    <span className="text-xs font-mono font-bold text-sky-400 mt-0.5">{vwap ? vwap.toFixed(2) : '---'} $</span>
                </div>
                <div className="bg-white/[0.02] border border-white/5 rounded-xl p-2 flex flex-col">
                    <span className="text-[8px] text-slate-500 uppercase font-black tracking-widest">ADX (Trend Strength)</span>
                    <span className={`text-xs font-mono font-bold mt-0.5 ${adx && adx > 25 ? 'text-amber-400 drop-shadow-[0_0_5px_rgba(251,191,36,0.5)]' : 'text-slate-400'}`}>
                        {adx ? adx.toFixed(1) : '---'} {adx && adx > 25 ? 'ðŸ”¥' : ''}
                    </span>
                </div>
            </div>

            <div className="relative pt-2 z-10">
                <div className="flex justify-between text-[8px] font-bold text-slate-500 uppercase mb-1 px-1">
                    <span>Oversold (30)</span>
                    <span className="text-slate-300">RSI {rsi.toFixed(1)}</span>
                    <span>Overbought (70)</span>
                </div>
                <div className="h-1.5 bg-black/40 rounded-full overflow-hidden border border-white/5 shadow-inner relative">
                    {/* Zones safe */}
                    <div className="absolute left-[30%] right-[30%] top-0 bottom-0 bg-white/[0.04]" />
                    <div
                        className={`h-full rounded-full transition-all duration-1500 ease-out shadow-[0_0_10px_currentColor] ${barColor}`}
                        style={{ width: `${width}%` }}
                    />
                </div>
            </div>
        </div>
    )
}

function DrawdownGauge({ drawdown, limit, isLocked }: { drawdown: number, limit: number, isLocked: boolean }) {
    const dangerZone = drawdown > (limit * 0.8) // 80% du drawdown max = Danger
    const fatalZone = drawdown >= limit || isLocked

    const borderClass = fatalZone ? 'border-red-500/60' : dangerZone ? 'border-amber-500/40' : 'border-sky-500/20'
    const barColor = fatalZone ? 'bg-red-500' : dangerZone ? 'bg-gradient-to-r from-amber-500 to-red-500' : 'bg-gradient-to-r from-sky-500 to-sky-400'
    const iconColor = fatalZone ? 'text-red-500' : dangerZone ? 'text-amber-400' : 'text-sky-400'

    const percentage = Math.min((drawdown / limit) * 100, 100)

    return (
        <div className={`glass p-7 rounded-[2rem] border ${borderClass} shadow-2xl relative overflow-hidden group transition-all duration-300 ${fatalZone ? 'animate-pulse' : ''}`}>
            <div className={`absolute top-0 right-0 w-32 h-32 blur-3xl rounded-full opacity-20 pointer-events-none ${fatalZone ? 'bg-red-600' : dangerZone ? 'bg-amber-600' : 'bg-sky-600'}`} />

            <div className="flex items-center justify-between mb-4 relative z-10">
                <p className="text-[10px] font-black text-slate-500 uppercase tracking-[0.2em] flex items-center gap-2">
                    <ShieldAlert size={12} className={iconColor} />
                    Accountant Kill-Switch
                </p>
                <div className={`px-3 py-1 bg-black/40 rounded-xl border ${borderClass} shadow-inner`}>
                    <span className={`text-[9px] font-black uppercase text-slate-500 tracking-widest ${iconColor}`}>
                        {fatalZone ? 'HALTED' : dangerZone ? 'WARNING' : 'SECURE'}
                    </span>
                </div>
            </div>

            <div className="flex items-end justify-between mb-2 relative z-10">
                <div className="flex items-baseline gap-2">
                    <p className={`text-3xl font-black tracking-tight ${fatalZone ? 'text-red-500 drop-shadow-[0_0_15px_rgba(239,68,68,0.8)]' : dangerZone ? 'text-amber-400 text-shadow-glow' : 'text-white'}`}>
                        -{drawdown.toFixed(2)}%
                    </p>
                </div>
                <p className="text-[10px] text-slate-500 font-bold uppercase tracking-widest mb-1">Max: -{limit.toFixed(1)}%</p>
            </div>

            {/* Gauge */}
            <div className="relative pt-2 z-10">
                <div className="h-2 bg-black/60 rounded-full overflow-hidden border border-white/5 shadow-inner">
                    <div
                        className={`h-full rounded-full transition-all duration-1000 ease-out shadow-[0_0_15px_currentColor] ${barColor}`}
                        style={{ width: `${Math.max(percentage, 2)}%` }}
                    />
                </div>
            </div>
        </div>
    )
}



