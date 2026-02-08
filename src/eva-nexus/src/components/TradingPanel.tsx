import { useState, useEffect } from 'react'
import { Wallet, ArrowUpCircle, ArrowDownCircle, ShieldAlert, Activity } from 'lucide-react'
import { getStatus, getTradingStatus } from '../services/api'

export default function TradingPanel() {
    const [bankerStatus, setBankerStatus] = useState<'online' | 'offline' | 'unknown'>('unknown')
    const [tradingData, setTradingData] = useState<any>(null)
    const [isLoading, setIsLoading] = useState(true)

    useEffect(() => {
        const fetchData = async () => {
            try {
                const [statusData, tradingResp] = await Promise.all([
                    getStatus(),
                    getTradingStatus()
                ])
                setBankerStatus((statusData.banker?.status || 'unknown') as 'online' | 'offline' | 'unknown')
                setTradingData(tradingResp)
                setIsLoading(false)
            } catch (e) {
                console.error("Error fetching trading data:", e)
                setBankerStatus('offline')
                setIsLoading(false)
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

    return (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-full animate-fade-in">
            {/* Portfolio Overview */}
            <div className="lg:col-span-2 flex flex-col gap-6">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <StatCard
                        title="Equity Total"
                        value={`${parseFloat(account.equity || 0).toLocaleString()} ${account.currency || '$'}`}
                        icon={<Wallet className="text-sky-400" />}
                        glow="sky"
                    />
                    <StatCard
                        title="Profit & Pertes"
                        value={`${totalProfit >= 0 ? '+' : ''}${totalProfit.toFixed(2)} $`}
                        icon={<ArrowUpCircle className={totalProfit >= 0 ? "text-emerald-400" : "text-red-400"} />}
                        trend={account.balance ? `${((totalProfit / parseFloat(account.balance)) * 100).toFixed(2)}%` : "0%"}
                        glow={totalProfit >= 0 ? "emerald" : "amber"}
                    />
                    <StatCard
                        title="VaR / Drawdown"
                        value={`${parseFloat(risk.daily_drawdown_percent || 0).toFixed(2)}% | VaR: ${parseFloat(risk.details?.var_check?.details || 0).toFixed(4)}`}
                        icon={<ShieldAlert className="text-amber-400" />}
                        trend={risk.trading_allowed ? "SECURE" : "LOCKED"}
                        glow="amber"
                    />
                </div>

                <div className="flex-grow glass rounded-[2.5rem] p-8 overflow-hidden flex flex-col shadow-2xl relative">
                    <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-sky-500/30 to-transparent" />
                    <div className="flex items-center justify-between mb-8">
                        <div>
                            <h4 className="font-black text-xl tracking-tight text-white/90">Positions de la Ruche</h4>
                            <p className="text-[10px] text-slate-500 uppercase font-bold tracking-[0.2em] mt-1">Surveillance Temps Réel</p>
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
                                        <p className="text-[10px] text-slate-500 font-bold uppercase tracking-widest mt-1">Volume: {pos.volume} • Ticket #{pos.ticket}</p>
                                    </div>
                                </div>
                                <div className="text-right">
                                    <p className={`font-black text-lg tracking-tight ${parseFloat(pos.profit) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                        {parseFloat(pos.profit) >= 0 ? '+' : ''}{pos.profit} $
                                    </p>
                                    <button className="text-[9px] text-slate-600 hover:text-red-400 mt-2 uppercase font-black tracking-[0.2em] transition-colors">Terminer Mission</button>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* Account Info & Hydra */}
            <div className="flex flex-col gap-6">
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
                    <h4 className="font-black text-sm text-white/80 mb-6 uppercase tracking-widest">Score de Sincérité Cognitive</h4>
                    <div className="mt-4">
                        <SincerityGauge score={tradingData?.last_sincerity_score || 0.85} />
                    </div>
                    <div className="mt-8 space-y-6">
                        <SentimentItem symbol="OR / XAUUSD" sentiment="Bullish" score={85} />
                        <SentimentItem symbol="NASDAQ 100" sentiment="Neutral" score={52} />
                    </div>
                </div>
            </div>
        </div>
    )
}

function StatCard({ title, value, icon, trend, glow }: { title: string, value: string, icon: any, trend?: string, glow: string }) {
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

function SentimentItem({ symbol, sentiment, score }: { symbol: string, sentiment: string, score: number }) {
    const barColor = sentiment === 'Bullish' ? 'bg-gradient-to-r from-emerald-600 to-emerald-400' : sentiment === 'Bearish' ? 'bg-gradient-to-r from-red-600 to-red-400' : 'bg-gradient-to-r from-slate-600 to-slate-400'
    const textColor = sentiment === 'Bullish' ? 'text-emerald-400' : sentiment === 'Bearish' ? 'text-red-400' : 'text-slate-400'

    return (
        <div className="space-y-3 group">
            <div className="flex justify-between text-[10px] font-black uppercase tracking-widest">
                <span className="text-slate-300 group-hover:text-white transition-colors">{symbol}</span>
                <span className={`${textColor} bg-white/[0.03] px-2 py-0.5 rounded-md`}>
                    {sentiment}
                </span>
            </div>
            <div className="h-1.5 bg-white/[0.03] rounded-full overflow-hidden border border-white/5 shadow-inner">
                <div
                    className={`h-full rounded-full transition-all duration-1500 ease-out shadow-lg ${barColor}`}
                    style={{ width: `${score}%` }}
                />
            </div>
        </div>
    )
}
