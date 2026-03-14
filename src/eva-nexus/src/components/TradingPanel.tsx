import { useEffect, useMemo, useState } from 'react'
import { Activity, BrainCircuit, ShieldAlert } from 'lucide-react'
import {
    checkNodeHealth,
    closePosition,
    createOrder,
    getLabChampionStatus,
    getLabTrainingStatus,
    getModelPerformance,
    getNemesisStatus,
    getTradingStatus,
    toggleAutoTrading,
    type LabChampionStatus,
    type ModelPerformanceReport,
    type NemesisStatus,
    type TradingDecisionStatus,
    type TradingPosition,
    type TradingStatusResponse,
    type TrainingRunStatus,
} from '../services/api'
import ChampionStatusPanel from './trading/ChampionStatusPanel'
import LivePerformancePanel from './trading/LivePerformancePanel'
import TradingOverviewPanel from './trading/TradingOverviewPanel'
import TrainingRunPanel from './trading/TrainingRunPanel'
import UniverseSummaryPanel from './trading/UniverseSummaryPanel'
import { MetricPill, PanelShell, StatusBadge, formatDateLabel, formatUsd } from './trading/TradingShared'

type BankerNodeStatus = 'online' | 'offline' | 'unknown'

interface OrderFormState {
    symbol: string
    volume: number
    sl: number
    tp: number
}

const DEFAULT_TRADING_STATUS: TradingStatusResponse = {
    status: 'offline',
    connection: { mt5_connected: false, mock_mode: false },
    account: { equity: 0, balance: 0, margin: 0, free_margin: 0, currency: 'USD', leverage: 0 },
    positions: [],
    risk: {
        daily_drawdown_percent: 0,
        trading_allowed: false,
        open_positions: 0,
        anti_tilt_active: false,
        news_filter_active: false,
    },
    decisions: {},
    universe: {
        dynamic: false,
        symbols_total: 0,
        batch_size: 0,
        lab_live: null,
    },
}

function resolveLabStatus(trainingStatus: TrainingRunStatus | null): 'online' | 'offline' | 'degraded' {
    if (!trainingStatus) return 'offline'
    if (trainingStatus.status === 'offline') return 'offline'
    if (trainingStatus.run?.status === 'error') return 'degraded'
    return 'online'
}

function detectBlockers(params: {
    bankerStatus: BankerNodeStatus
    tradingData: TradingStatusResponse | null
    trainingStatus: TrainingRunStatus | null
    nemesisStatus: NemesisStatus | null
}) {
    const blockers: string[] = []
    if (params.bankerStatus !== 'online') {
        blockers.push('Banker indisponible')
    }
    if (params.trainingStatus?.dependencies?.vllm?.state === 'stopped_for_training') {
        blockers.push('vLLM arrete pour entrainement')
    }
    if (params.nemesisStatus?.trading_blocked) {
        blockers.push(`Nemesis active jusqu a ${formatDateLabel(params.nemesisStatus.blocked_until)}`)
    }
    if (params.tradingData && !params.tradingData.risk.trading_allowed) {
        blockers.push('Entrées bloquees par le risque')
    }
    return blockers
}

export default function TradingPanel() {
    const [bankerStatus, setBankerStatus] = useState<BankerNodeStatus>('unknown')
    const [tradingData, setTradingData] = useState<TradingStatusResponse>(DEFAULT_TRADING_STATUS)
    const [championStatus, setChampionStatus] = useState<LabChampionStatus | null>(null)
    const [modelPerformance, setModelPerformance] = useState<ModelPerformanceReport | null>(null)
    const [trainingStatus, setTrainingStatus] = useState<TrainingRunStatus | null>(null)
    const [nemesisStatus, setNemesisStatus] = useState<NemesisStatus | null>(null)
    const [actionLoading, setActionLoading] = useState(false)
    const [autoTrading, setAutoTrading] = useState(true)
    const [orderForm, setOrderForm] = useState<OrderFormState>({
        symbol: 'XAUUSD',
        volume: 0.01,
        sl: 2000,
        tp: 2100,
    })

    useEffect(() => {
        const fetchData = async () => {
            try {
                const bankerHealth = await checkNodeHealth('Banker', '/api/banker/health')
                const currentStatus: BankerNodeStatus = bankerHealth.status === 'online' ? 'online' : 'offline'
                const [tradingResp, championResp, performanceResp, trainingResp, nemesisResp] = await Promise.all([
                    currentStatus === 'online' ? getTradingStatus() : Promise.resolve(DEFAULT_TRADING_STATUS),
                    getLabChampionStatus(),
                    currentStatus === 'online' ? getModelPerformance(7, 6) : Promise.resolve(null),
                    getLabTrainingStatus(),
                    currentStatus === 'online' ? getNemesisStatus() : Promise.resolve(null),
                ])
                setBankerStatus(currentStatus)
                setTradingData(tradingResp)
                setChampionStatus(championResp)
                setModelPerformance(performanceResp)
                setTrainingStatus(trainingResp)
                setNemesisStatus(nemesisResp)
            } catch (error) {
                console.error('Erreur de chargement du Trading Floor:', error)
                setBankerStatus('offline')
            }
        }

        fetchData()
        const interval = setInterval(fetchData, 5000)
        return () => clearInterval(interval)
    }, [])

    const positions = tradingData.positions || []
    const account = tradingData.account || DEFAULT_TRADING_STATUS.account
    const risk = tradingData.risk || DEFAULT_TRADING_STATUS.risk
    const decisions = tradingData.decisions || {}
    const openPnl = positions.reduce((sum, position) => sum + Number(position.profit || 0), 0)
    const realizedPnl = Number(modelPerformance?.summary?.realized_pnl ?? modelPerformance?.summary?.net_profit ?? 0)
    const netPnl = realizedPnl + openPnl
    const closedTrades = Number(modelPerformance?.summary?.closed_trades || 0)
    const winRate = Number(modelPerformance?.summary?.win_rate || 0)
    const labStatus = resolveLabStatus(trainingStatus)
    const trainerState = String(trainingStatus?.dependencies?.trainer?.state || 'idle')
    const vllmState = String(trainingStatus?.dependencies?.vllm?.state || 'unknown')
    const blockers = detectBlockers({ bankerStatus, tradingData, trainingStatus, nemesisStatus })

    const sortedDecisions = useMemo(() => {
        return Object.entries(decisions)
            .map(([symbol, decision]) => ({ symbol, ...(decision as TradingDecisionStatus) }))
            .sort((left, right) => {
                const leftWeight = left.action === 'BUY' || left.action === 'SELL' ? 0 : 1
                const rightWeight = right.action === 'BUY' || right.action === 'SELL' ? 0 : 1
                if (leftWeight !== rightWeight) return leftWeight - rightWeight
                return Math.abs(Number(right.rsi || 0) - 50) - Math.abs(Number(left.rsi || 0) - 50)
            })
            .slice(0, 8)
    }, [decisions])

    const handleAutoToggle = async () => {
        const nextState = !autoTrading
        setAutoTrading(nextState)
        await toggleAutoTrading(nextState)
    }

    const refreshTrading = async () => {
        if (bankerStatus !== 'online') return
        setTradingData(await getTradingStatus())
        setNemesisStatus(await getNemesisStatus())
    }

    const handleOrder = async (action: 'BUY' | 'SELL') => {
        setActionLoading(true)
        await createOrder({
            symbol: orderForm.symbol,
            action,
            volume: orderForm.volume,
            stop_loss: orderForm.sl,
            take_profit: orderForm.tp,
        })
        setActionLoading(false)
        await refreshTrading()
    }

    const handleClose = async (ticket: string) => {
        if (!confirm(`Fermer la position #${ticket} ?`)) return
        await closePosition(ticket)
        await refreshTrading()
    }

    const handleHold = async () => {
        if (!confirm(`Fermer toutes les positions ${orderForm.symbol} ?`)) return
        setActionLoading(true)
        const targetPositions = positions.filter((position) => position.symbol === orderForm.symbol)
        await Promise.all(targetPositions.map((position) => closePosition(position.ticket)))
        setActionLoading(false)
        await refreshTrading()
    }

    return (
        <div className="grid grid-cols-1 2xl:grid-cols-[minmax(0,1.65fr)_minmax(360px,0.95fr)] gap-6 h-full animate-fade-in">
            <div className="flex flex-col gap-6 min-w-0">
                <TradingOverviewPanel
                    bankerStatus={bankerStatus}
                    labStatus={labStatus}
                    trainerState={trainerState}
                    vllmState={vllmState}
                    account={account}
                    risk={risk}
                    positions={positions}
                    realizedPnl={realizedPnl}
                    openPnl={openPnl}
                    netPnl={netPnl}
                    closedTrades={closedTrades}
                    winRate={winRate}
                    nemesis={nemesisStatus}
                />

                <ManualExecutionPanel
                    orderForm={orderForm}
                    setOrderForm={setOrderForm}
                    actionLoading={actionLoading}
                    autoTrading={autoTrading}
                    onToggleAuto={handleAutoToggle}
                    onBuy={() => handleOrder('BUY')}
                    onSell={() => handleOrder('SELL')}
                    onHold={handleHold}
                    availableSymbols={Object.keys(decisions)}
                />

                <PositionsPanel positions={positions} onClose={handleClose} />
                <DecisionFeedPanel decisions={sortedDecisions} blockers={blockers} nemesisStatus={nemesisStatus} />
            </div>

            <div className="flex flex-col gap-6 min-w-0">
                <TrainingRunPanel trainingStatus={trainingStatus} />
                <ChampionStatusPanel championStatus={championStatus} />
                <LivePerformancePanel modelPerformance={modelPerformance} openPnl={openPnl} />
                <UniverseSummaryPanel
                    trainingStatus={trainingStatus}
                    championStatus={championStatus}
                    tradingData={tradingData}
                />
            </div>
        </div>
    )
}

function ManualExecutionPanel({
    orderForm,
    setOrderForm,
    actionLoading,
    autoTrading,
    onToggleAuto,
    onBuy,
    onSell,
    onHold,
    availableSymbols,
}: {
    orderForm: OrderFormState
    setOrderForm: (value: OrderFormState) => void
    actionLoading: boolean
    autoTrading: boolean
    onToggleAuto: () => Promise<void>
    onBuy: () => Promise<void>
    onSell: () => Promise<void>
    onHold: () => Promise<void>
    availableSymbols: string[]
}) {
    const symbols = availableSymbols.length > 0 ? availableSymbols : ['XAUUSD', 'EURUSD', 'BTCUSD', 'US30.cash']

    return (
        <PanelShell
            title="Execution live"
            subtitle="Override manuel et pilotage des ordres"
            accent="sky"
            aside={<StatusBadge label={autoTrading ? 'Auto-pilot actif' : 'Auto-pilot coupe'} tone={autoTrading ? 'emerald' : 'amber'} />}
        >
            <div className="flex items-center justify-between gap-4 mb-4">
                <div className="text-[11px] text-slate-400">Le terminal manuel reste disponible meme sans role admin.</div>
                <button
                    onClick={onToggleAuto}
                    className={`rounded-full px-4 py-2 text-[10px] font-black uppercase tracking-[0.2em] transition-colors ${autoTrading ? 'bg-emerald-500/10 text-emerald-300 border border-emerald-500/20' : 'bg-amber-500/10 text-amber-300 border border-amber-500/20'}`}
                >
                    {autoTrading ? 'Desactiver auto' : 'Activer auto'}
                </button>
            </div>

            <div className="grid grid-cols-2 xl:grid-cols-5 gap-3 items-end">
                <label className="block">
                    <span className="text-[9px] text-slate-500 uppercase font-black tracking-[0.2em]">Symbole</span>
                    <select
                        value={orderForm.symbol}
                        onChange={(event) => setOrderForm({ ...orderForm, symbol: event.target.value })}
                        className="mt-2 w-full rounded-xl border border-white/10 bg-black/40 px-3 py-2 text-xs font-mono font-bold outline-none"
                    >
                        {symbols.map((symbol) => (
                            <option key={symbol} value={symbol}>{symbol}</option>
                        ))}
                    </select>
                </label>
                <label className="block">
                    <span className="text-[9px] text-slate-500 uppercase font-black tracking-[0.2em]">Volume</span>
                    <input
                        type="number"
                        step="0.01"
                        value={orderForm.volume}
                        onChange={(event) => setOrderForm({ ...orderForm, volume: Number(event.target.value) })}
                        className="mt-2 w-full rounded-xl border border-white/10 bg-black/40 px-3 py-2 text-xs font-mono font-bold outline-none"
                    />
                </label>
                <label className="block">
                    <span className="text-[9px] text-slate-500 uppercase font-black tracking-[0.2em]">Stop loss</span>
                    <input
                        type="number"
                        value={orderForm.sl}
                        onChange={(event) => setOrderForm({ ...orderForm, sl: Number(event.target.value) })}
                        className="mt-2 w-full rounded-xl border border-white/10 bg-black/40 px-3 py-2 text-xs font-mono font-bold outline-none"
                    />
                </label>
                <label className="block">
                    <span className="text-[9px] text-slate-500 uppercase font-black tracking-[0.2em]">Take profit</span>
                    <input
                        type="number"
                        value={orderForm.tp}
                        onChange={(event) => setOrderForm({ ...orderForm, tp: Number(event.target.value) })}
                        className="mt-2 w-full rounded-xl border border-white/10 bg-black/40 px-3 py-2 text-xs font-mono font-bold outline-none"
                    />
                </label>
                <div className="grid grid-cols-3 gap-2">
                    <ActionButton label="BUY" tone="emerald" onClick={onBuy} disabled={actionLoading} />
                    <ActionButton label="SELL" tone="rose" onClick={onSell} disabled={actionLoading} />
                    <ActionButton label="HOLD" tone="amber" onClick={onHold} disabled={actionLoading} />
                </div>
            </div>
        </PanelShell>
    )
}

function PositionsPanel({ positions, onClose }: { positions: TradingPosition[]; onClose: (ticket: string) => Promise<void> }) {
    return (
        <PanelShell title="Positions ouvertes" subtitle="Etat live des ordres en cours" accent="emerald">
            <div className="space-y-3 max-h-[26rem] overflow-y-auto pr-2 custom-scrollbar">
                {positions.length === 0 ? (
                    <div className="rounded-2xl border border-white/5 bg-white/[0.02] p-6 text-center text-sm font-black text-slate-500 uppercase tracking-[0.18em]">
                        Aucune position ouverte
                    </div>
                ) : positions.map((position) => (
                    <div key={position.ticket} className="rounded-2xl border border-white/5 bg-black/20 p-4 flex items-center justify-between gap-4">
                        <div>
                            <div className="text-sm font-black text-white/90">{position.symbol} | {position.action}</div>
                            <div className="mt-1 text-[10px] text-slate-500 uppercase font-black tracking-[0.18em]">
                                ticket #{position.ticket} | volume {position.volume} | entree {position.open_price}
                            </div>
                        </div>
                        <div className="text-right">
                            <div className={`text-sm font-black ${Number(position.profit) >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                                {formatUsd(Number(position.profit || 0))}
                            </div>
                            <button
                                onClick={() => onClose(position.ticket)}
                                className="mt-2 text-[9px] font-black uppercase tracking-[0.18em] text-slate-500 hover:text-rose-400"
                            >
                                Fermer
                            </button>
                        </div>
                    </div>
                ))}
            </div>
        </PanelShell>
    )
}

function DecisionFeedPanel({
    decisions,
    blockers,
    nemesisStatus,
}: {
    decisions: Array<{ symbol: string } & TradingDecisionStatus>
    blockers: string[]
    nemesisStatus: NemesisStatus | null
}) {
    return (
        <PanelShell title="Signal live" subtitle="Decisions en direct et messages de blocage" accent="rose">
            <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1.2fr)_minmax(280px,0.8fr)] gap-4">
                <div className="space-y-3 max-h-[24rem] overflow-y-auto pr-2 custom-scrollbar">
                    {decisions.length === 0 ? (
                        <div className="rounded-2xl border border-white/5 bg-white/[0.02] p-4 text-sm text-slate-400">
                            Aucune decision disponible.
                        </div>
                    ) : decisions.map((decision) => {
                        const actionTone = decision.action === 'BUY' ? 'text-emerald-400' : decision.action === 'SELL' ? 'text-rose-400' : 'text-slate-300'
                        return (
                            <div key={`${decision.symbol}-${decision.action}-${decision.rsi}`} className="rounded-2xl border border-white/5 bg-black/20 p-4">
                                <div className="flex items-start justify-between gap-3">
                                    <div>
                                        <div className="text-sm font-black text-white/90">{decision.symbol}</div>
                                        <div className="mt-1 text-[10px] text-slate-500 uppercase font-black tracking-[0.18em]">
                                            RSI {Number(decision.rsi || 0).toFixed(1)} | ADX {Number(decision.adx || 0).toFixed(1)} | VWAP {Number(decision.vwap || 0).toFixed(2)}
                                        </div>
                                    </div>
                                    <div className={`text-xs font-black uppercase tracking-[0.2em] ${actionTone}`}>{decision.action}</div>
                                </div>
                                <div className="mt-3 grid grid-cols-2 gap-2">
                                    <MetricPill label="Cortex" value={String(decision.cortex_bias || 'indisponible')} />
                                    <MetricPill label="GNN" value={String(decision.gnn_bias || 'indisponible')} />
                                </div>
                                <div className="mt-3 text-[11px] text-slate-300 leading-relaxed">{decision.comment || 'Aucun commentaire de decision.'}</div>
                            </div>
                        )
                    })}
                </div>

                <div className="space-y-3">
                    <div className="rounded-2xl border border-white/5 bg-black/20 p-4">
                        <div className="flex items-center gap-2 text-[9px] text-slate-500 uppercase font-black tracking-[0.2em]">
                            <ShieldAlert size={14} className="text-amber-400" />
                            Blocages actifs
                        </div>
                        <div className="mt-3 space-y-2">
                            {blockers.length === 0 ? (
                                <div className="text-[11px] text-emerald-300 font-bold">Aucun blocage critique.</div>
                            ) : blockers.map((blocker) => (
                                <div key={blocker} className="rounded-xl border border-amber-500/20 bg-amber-500/10 px-3 py-2 text-[11px] text-amber-200">
                                    {blocker}
                                </div>
                            ))}
                        </div>
                    </div>

                    <div className="rounded-2xl border border-white/5 bg-black/20 p-4">
                        <div className="flex items-center gap-2 text-[9px] text-slate-500 uppercase font-black tracking-[0.2em]">
                            <BrainCircuit size={14} className="text-rose-400" />
                            Nemesis
                        </div>
                        <div className="mt-3 space-y-2 text-[11px] text-slate-300">
                            <div>Defaites totales: {nemesisStatus?.total_defeats || 0}</div>
                            <div>Trading bloque: {nemesisStatus?.trading_blocked ? 'oui' : 'non'}</div>
                            <div>Deblocage: {formatDateLabel(nemesisStatus?.blocked_until || null)}</div>
                        </div>
                    </div>

                    <div className="rounded-2xl border border-white/5 bg-black/20 p-4">
                        <div className="flex items-center gap-2 text-[9px] text-slate-500 uppercase font-black tracking-[0.2em]">
                            <Activity size={14} className="text-sky-400" />
                            Lecture rapide
                        </div>
                        <div className="mt-3 text-[11px] text-slate-300 leading-relaxed">
                            HOLD = pas d ordre. Rejected = veto. Ordre valide + EXEC = ordre parti. Fermer manuellement ne doit plus etre necessaire apres stabilisation du Shepherd.
                        </div>
                    </div>
                </div>
            </div>
        </PanelShell>
    )
}

function ActionButton({
    label,
    tone,
    onClick,
    disabled,
}: {
    label: string
    tone: 'emerald' | 'rose' | 'amber'
    onClick: () => Promise<void>
    disabled: boolean
}) {
    const className = tone === 'emerald'
        ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-300 hover:bg-emerald-500 hover:text-black'
        : tone === 'rose'
            ? 'border-rose-500/30 bg-rose-500/10 text-rose-300 hover:bg-rose-500 hover:text-black'
            : 'border-amber-500/30 bg-amber-500/10 text-amber-300 hover:bg-amber-500 hover:text-black'

    return (
        <button
            onClick={onClick}
            disabled={disabled}
            className={`rounded-xl border px-3 py-2 text-[10px] font-black uppercase tracking-[0.2em] transition-colors ${className}`}
        >
            {label}
        </button>
    )
}
