import { Wallet, ArrowUpCircle, ArrowDownCircle, ShieldAlert } from 'lucide-react'

export default function TradingPanel() {
    // Statuts simulés pour la démo UI
    const positions = [
        { id: 101, symbol: 'XAUUSD', type: 'BUY', volume: 0.50, profit: '+450.20', pnl: 'positive' },
        { id: 102, symbol: 'EURUSD', type: 'SELL', volume: 0.10, profit: '-12.40', pnl: 'negative' },
    ]

    return (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-full animate-fade-in">
            {/* Portfolio Overview */}
            <div className="lg:col-span-2 flex flex-col gap-6">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <StatCard
                        title="Equity"
                        value="102,450.20 $"
                        icon={<Wallet className="text-sky-400" />}
                    />
                    <StatCard
                        title="P&L Jour"
                        value="+437.80 $"
                        icon={<ArrowUpCircle className="text-emerald-400" />}
                        trend="+1.2%"
                    />
                    <StatCard
                        title="Drawdown"
                        value="0.45 %"
                        icon={<ShieldAlert className="text-amber-400" />}
                        trend="OK"
                    />
                </div>

                <div className="flex-grow glass rounded-3xl p-6 overflow-hidden flex flex-col">
                    <div className="flex items-center justify-between mb-6">
                        <h4 className="font-bold text-lg">Positions Actives</h4>
                        <span className="text-xs px-2 py-1 bg-sky-500/10 text-sky-400 rounded-md border border-sky-500/20 font-bold uppercase">
                            The Banker
                        </span>
                    </div>

                    <div className="space-y-3 overflow-y-auto">
                        {positions.map(pos => (
                            <div key={pos.id} className="p-4 rounded-2xl bg-slate-900/50 border border-slate-800 flex items-center justify-between hover:border-slate-700 transition-colors">
                                <div className="flex items-center gap-4">
                                    <div className={`w-10 h-10 rounded-xl flex items-center justify-center font-bold text-xs ${pos.type === 'BUY' ? 'bg-emerald-500/10 text-emerald-500' : 'bg-red-500/10 text-red-500'}`}>
                                        {pos.type}
                                    </div>
                                    <div>
                                        <p className="font-bold text-sm tracking-tight">{pos.symbol}</p>
                                        <p className="text-[10px] text-slate-500 font-medium">Vol: {pos.volume} • ID #{pos.id}</p>
                                    </div>
                                </div>
                                <div className="text-right">
                                    <p className={`font-bold text-sm ${pos.pnl === 'positive' ? 'text-emerald-400' : 'text-red-400'}`}>{pos.profit} $</p>
                                    <button className="text-[10px] text-slate-500 hover:text-red-400 mt-1 uppercase font-bold tracking-widest">Fermer</button>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* Account Info & Hydra */}
            <div className="flex flex-col gap-6">
                <div className="glass rounded-3xl p-6 border-l-4 border-l-indigo-500">
                    <h4 className="text-xs font-bold text-indigo-400 uppercase mb-4 tracking-tighter">Prop-Firm Hydra</h4>
                    <div className="space-y-4">
                        <AccountRow label="FTMO Demo" status="Active" color="emerald" />
                        <AccountRow label="FundedNext" status="Active" color="emerald" />
                        <AccountRow label="Apex Lead" status="Monitoring" color="sky" />
                    </div>
                </div>

                <div className="glass rounded-3xl p-6 flex-grow">
                    <h4 className="font-bold text-sm mb-4">Sentiment Marché (OSINT)</h4>
                    <div className="space-y-4">
                        <SentimentItem symbol="XAUUSD" sentiment="Bullish" score={85} />
                        <SentimentItem symbol="NAS100" sentiment="Neutral" score={52} />
                        <SentimentItem symbol="EURUSD" sentiment="Bearish" score={24} />
                    </div>
                </div>
            </div>
        </div>
    )
}

function StatCard({ title, value, icon, trend }: { title: string, value: string, icon: any, trend?: string }) {
    return (
        <div className="glass p-5 rounded-3xl border border-slate-800/50">
            <div className="flex items-center justify-between mb-2">
                <p className="text-xs font-bold text-slate-500 uppercase tracking-widest">{title}</p>
                <div className="p-1.5 bg-slate-800/50 rounded-lg">{icon}</div>
            </div>
            <div className="flex items-baseline gap-2">
                <p className="text-xl font-bold text-white">{value}</p>
                {trend && <span className={`text-[10px] font-bold ${trend.includes('+') ? 'text-emerald-400' : trend === 'OK' ? 'text-sky-400' : 'text-red-400'}`}>{trend}</span>}
            </div>
        </div>
    )
}

function AccountRow({ label, status, color }: { label: string, status: string, color: string }) {
    const colorClass = color === 'emerald' ? 'bg-emerald-500' : 'bg-sky-500'
    return (
        <div className="flex items-center justify-between p-3 rounded-xl bg-slate-900/30 border border-slate-800/50">
            <span className="text-xs font-medium text-slate-300">{label}</span>
            <div className="flex items-center gap-2">
                <div className={`w-1.5 h-1.5 rounded-full ${colorClass}`}></div>
                <span className="text-[10px] font-bold uppercase text-slate-500">{status}</span>
            </div>
        </div>
    )
}

function SentimentItem({ symbol, sentiment, score }: { symbol: string, sentiment: string, score: number }) {
    return (
        <div className="space-y-1.5">
            <div className="flex justify-between text-[11px] font-bold">
                <span>{symbol}</span>
                <span className={sentiment === 'Bullish' ? 'text-emerald-400' : sentiment === 'Bearish' ? 'text-red-400' : 'text-slate-400'}>
                    {sentiment}
                </span>
            </div>
            <div className="h-1 bg-slate-800 rounded-full overflow-hidden">
                <div
                    className={`h-full rounded-full transition-all duration-1000 ${sentiment === 'Bullish' ? 'bg-emerald-500' : sentiment === 'Bearish' ? 'bg-red-500' : 'bg-slate-500'}`}
                    style={{ width: `${score}%` }}
                />
            </div>
        </div>
    )
}
