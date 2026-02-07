import { useState, useEffect } from 'react'
import {
    MessageSquare,
    TrendingUp,
    Shield,
    Settings,
    User,
    ChevronRight,
    Activity,
    Cpu
} from 'lucide-react'
import Chat from './components/Chat'
import TradingPanel from './components/TradingPanel'
import { getStatus } from './services/api'

function App() {
    const [activeTab, setActiveTab] = useState<'chat' | 'trading' | 'security' | 'settings'>('chat')
    const [systemStatus, setSystemStatus] = useState({ core: 'online', banker: 'online', sentinel: 'online' })
    const [securityData, setSecurityData] = useState<any>(null)

    // Simulation monitoring - à remplacer par WebSocket
    useEffect(() => {
        const fetchStatus = async () => {
            try {
                const status = await getStatus()
                setSystemStatus({
                    core: status.core?.status || 'online',
                    banker: status.banker?.status || 'online',
                    sentinel: status.sentinel?.status || 'online'
                })
            } catch (e) {
                console.error("Erreur status", e)
            }
        }
        fetchStatus()
        const interval = setInterval(fetchStatus, 30000)
        return () => clearInterval(interval)
    }, [])

    // Real-time security metrics polling
    useEffect(() => {
        if (activeTab === 'security') {
            const fetchSecurity = async () => {
                try {
                    const { getSystemStatus } = await import('./services/api')
                    const data = await getSystemStatus()
                    setSecurityData(data)
                } catch (e) {
                    console.error("Sentinel sync error", e)
                }
            }
            fetchSecurity()
            const interval = setInterval(fetchSecurity, 5000)
            return () => clearInterval(interval)
        }
    }, [activeTab])

    const metrics = securityData?.metrics || {}

    return (
        <div className="flex h-full w-full bg-slate-950 text-white overflow-hidden neural-bg" style={{ height: '100vh' }}>
            {/* Sidebar Navigation */}
            <aside className="w-20 lg:w-72 flex flex-col border-r border-white/5 glass-heavy relative z-20">
                <div className="p-8 flex items-center gap-4">
                    <div className="w-12 h-12 bg-gradient-to-br from-sky-400 to-indigo-600 rounded-2xl flex items-center justify-center shadow-2xl shadow-sky-500/30 animate-pulse-glow">
                        <Cpu className="w-7 h-7 text-white" />
                    </div>
                    <div className="hidden lg:block">
                        <h1 className="font-bold text-xl tracking-wider bg-clip-text text-transparent bg-gradient-to-r from-white to-slate-400">THE HIVE</h1>
                        <p className="text-[10px] uppercase font-bold text-sky-500 tracking-[0.2em]">Neural Nexus</p>
                    </div>
                </div>

                <nav className="flex-grow flex flex-col gap-2 p-3">
                    <NavItem
                        icon={<MessageSquare size={20} />}
                        label="E.V.A. Chat"
                        active={activeTab === 'chat'}
                        onClick={() => setActiveTab('chat')}
                    />
                    <NavItem
                        icon={<TrendingUp size={20} />}
                        label="Trading"
                        active={activeTab === 'trading'}
                        onClick={() => setActiveTab('trading')}
                    />
                    <NavItem
                        icon={<Shield size={20} />}
                        label="Sécurité"
                        active={activeTab === 'security'}
                        onClick={() => setActiveTab('security')}
                    />
                    <div className="mt-auto">
                        <NavItem
                            icon={<Settings size={20} />}
                            label="Réglages"
                            active={activeTab === 'settings'}
                            onClick={() => setActiveTab('settings')}
                        />
                    </div>
                </nav>

                <div className="p-6 mt-auto border-t border-white/5">
                    <div className="flex items-center gap-4 p-3 rounded-2xl hover:bg-white/5 transition-all duration-300 cursor-pointer group border border-transparent hover:border-white/10">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-500 to-purple-600 flex items-center justify-center text-xs font-bold shadow-lg group-hover:scale-110 transition-transform">JM</div>
                        <div className="hidden lg:block">
                            <p className="text-sm font-bold text-slate-200">John Moncel</p>
                            <p className="text-[11px] text-sky-500 font-bold uppercase tracking-widest opacity-80">Maître</p>
                        </div>
                        <ChevronRight size={14} className="hidden lg:block ml-auto text-slate-600 group-hover:translate-x-1 transition-transform" />
                    </div>
                </div>
            </aside>

            {/* Main Content Area */}
            <main className="flex-grow flex flex-col relative overflow-hidden h-full z-10">
                {/* Header / Status Bar */}
                <header className="h-[var(--header-height)] border-b border-white/5 flex items-center justify-between px-8 bg-white/[0.02] backdrop-blur-md">
                    <div className="flex items-center gap-6">
                        <h2 className="text-base font-bold tracking-tight text-white/90">
                            {activeTab === 'chat' ? 'Orchestrateur Core' : activeTab === 'trading' ? 'The Banker' : 'The Sentinel'}
                        </h2>
                        <div className="flex items-center gap-3">
                            <StatusBadge label="Core" status={systemStatus.core} />
                            <StatusBadge label="Banker" status={systemStatus.banker} />
                            <StatusBadge label="Sentinel" status={systemStatus.sentinel} />
                        </div>
                    </div>

                    <div className="flex items-center gap-6 text-[11px] font-bold text-slate-400 uppercase tracking-widest">
                        <div className="flex items-center gap-2 px-4 py-2 bg-emerald-500/10 rounded-xl border border-emerald-500/20 text-emerald-400">
                            <Activity size={14} className="animate-pulse" />
                            <span>34ms latency</span>
                        </div>
                        <div className="px-4 py-2 bg-sky-500/10 rounded-xl border border-sky-500/20 text-sky-400">
                            Genesis Phase
                        </div>
                    </div>
                </header>

                {/* Content Switcher */}
                <div className="flex-grow overflow-hidden relative p-4 lg:p-6 bg-slate-950/20">
                    {activeTab === 'chat' && <Chat />}
                    {activeTab === 'trading' && <TradingPanel />}
                    {activeTab === 'security' && (
                        <div className="h-full flex flex-col gap-6 animate-fade-in">
                            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                                <MetricBox label="CPU Load" value={`${metrics.cpu_percent || 0}%`} color="sky" />
                                <MetricBox label="RAM Usage" value={`${metrics.ram_used_gb || 0} / ${metrics.ram_total_gb || 0} GB`} color="indigo" />
                                <MetricBox label="GPU Temp" value={`${metrics.gpu?.temperature_celsius || 0}°C`} color={metrics.gpu?.temperature_celsius > 75 ? 'red' : 'emerald'} />
                                <MetricBox label="Disk Usage" value={`${metrics.disk_used_percent || 0}%`} color="slate" />
                            </div>

                            <div className="flex-grow glass rounded-[2.5rem] p-12 flex flex-col items-center justify-center text-center relative overflow-hidden">
                                <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-sky-500/30 to-transparent" />
                                <Shield size={120} className={`mb-8 drop-shadow-[0_0_30px_rgba(56,189,248,0.3)] ${systemStatus.sentinel === 'online' ? 'text-sky-400' : 'text-slate-600'}`} />
                                <h3 className="text-3xl font-black mb-4 tracking-tighter">SURVEILLANCE SENTINEL</h3>
                                <p className="text-slate-400 max-w-lg text-sm leading-relaxed">
                                    {systemStatus.sentinel === 'online'
                                        ? "La Ruche est sous protection active. Le noyau Sentinel surveille les processus système et l'intégrité du réseau en temps réel."
                                        : "Le module de sécurité est hors ligne ou en cours de déploiement. Activez les protocoles de défense manuellement si nécessaire."}
                                </p>

                                <div className="mt-12 flex gap-8">
                                    <div className="text-left">
                                        <p className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-2">Statut IDS</p>
                                        <div className="flex items-center gap-2">
                                            <div className={`w-2 h-2 rounded-full ${systemStatus.sentinel === 'online' ? 'bg-emerald-500' : 'bg-red-500'} animate-pulse`} />
                                            <span className="text-xs font-bold text-slate-200">{systemStatus.sentinel === 'online' ? "ACTIF & SÉCURISÉ" : "HORS LIGNE"}</span>
                                        </div>
                                    </div>
                                    <div className="w-[1px] h-10 bg-white/5" />
                                    <div className="text-left">
                                        <p className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-2">Localisation</p>
                                        <span className="text-xs font-bold text-slate-200 tracking-tight">VIRTUAL_ENV_CLUSTER_01</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}
                    {activeTab === 'settings' && (
                        <div className="flex flex-col items-center justify-center h-full text-center animate-fade-in">
                            <Settings size={64} className="text-slate-600 mb-4 opacity-30" />
                            <p className="text-slate-400">Configuration en lecture seule via The Tablet.</p>
                        </div>
                    )}
                </div>
            </main>
        </div>
    )
}

function NavItem({ icon, label, active, onClick }: { icon: any, label: string, active?: boolean, onClick: () => void }) {
    return (
        <button
            onClick={onClick}
            className={`
        flex items-center gap-4 p-4 rounded-2xl transition-all duration-300 group relative
        ${active
                    ? 'bg-sky-500/5 text-sky-400 border border-sky-500/20 shadow-lg shadow-sky-500/10'
                    : 'text-slate-500 hover:bg-white/5 hover:text-slate-200 border border-transparent'}
      `}
        >
            {active && (
                <div className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-8 bg-sky-500 rounded-r-full shadow-[0_0_15px_var(--primary-glow)]" />
            )}
            <span className={`${active ? 'text-sky-400' : 'text-slate-600 group-hover:text-slate-300'}`}>{icon}</span>
            <span className="hidden lg:block text-xs font-bold uppercase tracking-[0.15em]">{label}</span>
        </button>
    )
}

function MetricBox({ label, value, color }: { label: string, value: string, color: string }) {
    const colorClass = color === 'sky' ? 'text-sky-400 border-sky-500/20' : color === 'emerald' ? 'text-emerald-400 border-emerald-500/20' : color === 'red' ? 'text-red-400 border-red-500/20' : color === 'indigo' ? 'text-indigo-400 border-indigo-500/20' : 'text-slate-400 border-white/10'

    return (
        <div className={`p-6 rounded-[2rem] glass-heavy border ${colorClass.split(' ')[1]} shadow-2xl transition-transform hover:scale-105`}>
            <p className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-2">{label}</p>
            <p className={`text-xl font-black ${colorClass.split(' ')[0]} tracking-tighter`}>{value}</p>
        </div>
    )
}

function StatusBadge({ label, status }: { label: string, status: string }) {
    const isOnline = status === 'online'
    return (
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-white/[0.03] border border-white/5 shadow-inner">
            <div className={`w-2 h-2 rounded-full ${isOnline ? 'bg-emerald-500 animate-pulse' : 'bg-red-500'} shadow-[0_0_10px_rgba(16,185,129,0.4)]`}></div>
            <span className="text-[9px] font-black text-slate-500 uppercase tracking-[0.2em]">{label}</span>
        </div>
    )
}

export default App
