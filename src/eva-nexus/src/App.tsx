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

    // Simulation monitoring - à remplacer par WebSocket
    useEffect(() => {
        const fetchStatus = async () => {
            try {
                const status = await getStatus()
                // Formatage simplifié pour le dev
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
    }, [])

    return (
        <div className="flex h-full w-full bg-slate-950 text-white overflow-hidden" style={{ height: '100vh' }}>
            {/* Sidebar Navigation */}
            <aside className="w-20 lg:w-64 flex flex-col border-r border-slate-800 bg-slate-900/50 backdrop-blur-xl">
                <div className="p-6 flex items-center gap-3">
                    <div className="w-10 h-10 bg-sky-500 rounded-xl flex items-center justify-center shadow-lg shadow-sky-500/20">
                        <Cpu className="w-6 h-6 text-white" />
                    </div>
                    <h1 className="hidden lg:block font-bold text-xl tracking-tight">THE HIVE</h1>
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

                <div className="p-4 border-t border-slate-800">
                    <div className="flex items-center gap-3 p-2 rounded-lg hover:bg-slate-800 transition-colors cursor-pointer">
                        <div className="w-8 h-8 rounded-full bg-indigo-500 flex items-center justify-center text-xs font-bold">JM</div>
                        <div className="hidden lg:block">
                            <p className="text-xs font-medium">John Moncel</p>
                            <p className="text-[10px] text-slate-400">Maître</p>
                        </div>
                        <ChevronRight size={14} className="hidden lg:block ml-auto text-slate-500" />
                    </div>
                </div>
            </aside>

            {/* Main Content Area */}
            <main className="flex-grow flex flex-col relative overflow-hidden h-full">
                {/* Header / Status Bar */}
                <header className="h-16 border-b border-slate-800 flex items-center justify-between px-6 bg-slate-900/30">
                    <div className="flex items-center gap-4">
                        <h2 className="text-sm font-semibold capitalize text-slate-200">
                            {activeTab === 'chat' ? 'Orchestrateur Core' : activeTab === 'trading' ? 'The Banker' : 'The Sentinel'}
                        </h2>
                        <div className="flex items-center gap-3 ml-4">
                            <StatusBadge label="Core" status={systemStatus.core} />
                            <StatusBadge label="Banker" status={systemStatus.banker} />
                            <StatusBadge label="Sentinel" status={systemStatus.sentinel} />
                        </div>
                    </div>

                    <div className="flex items-center gap-4 text-xs font-medium text-slate-400">
                        <div className="flex items-center gap-1.5">
                            <Activity size={14} className="text-emerald-500" />
                            <span>34ms latency</span>
                        </div>
                        <div className="px-3 py-1 bg-slate-800 rounded-full border border-slate-700">
                            Genesis Phase
                        </div>
                    </div>
                </header>

                {/* Content Switcher */}
                <div className="flex-grow overflow-hidden relative p-4 lg:p-6 bg-slate-950/20">
                    {activeTab === 'chat' && <Chat />}
                    {activeTab === 'trading' && <TradingPanel />}
                    {activeTab === 'security' && (
                        <div className="flex flex-col items-center justify-center h-full text-center p-12 glass rounded-3xl animate-fade-in">
                            <Shield size={64} className="text-sky-500 mb-6 opacity-20" />
                            <h3 className="text-2xl font-bold mb-2">Module Sentinel</h3>
                            <p className="text-slate-400 max-w-md">L'agent Sentinel surveille activement les menaces sur les VMs Proxmox et le réseau interne.</p>
                            <div className="mt-8 grid grid-cols-2 gap-4 w-full max-w-lg">
                                <div className="p-4 rounded-2xl bg-slate-900 border border-slate-800 text-left">
                                    <p className="text-xs font-bold text-slate-500 uppercase mb-1">Dernière alerte</p>
                                    <p className="text-sm">Aucune menace détectée</p>
                                </div>
                                <div className="p-4 rounded-2xl bg-slate-900 border border-slate-800 text-left">
                                    <p className="text-xs font-bold text-slate-500 uppercase mb-1">Status IDS</p>
                                    <p className="text-sm text-emerald-500 font-medium">Actif (OK)</p>
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
        flex items-center gap-3 p-3 rounded-xl transition-all duration-200 group
        ${active
                    ? 'bg-sky-500/10 text-sky-400 border border-sky-500/20'
                    : 'text-slate-400 hover:bg-slate-800/50 hover:text-slate-200 border border-transparent'}
      `}
        >
            <span className={`${active ? 'text-sky-400' : 'text-slate-500 group-hover:text-slate-300'}`}>{icon}</span>
            <span className="hidden lg:block text-sm font-medium">{label}</span>
        </button>
    )
}

function StatusBadge({ label, status }: { label: string, status: string }) {
    const isOnline = status === 'online'
    return (
        <div className="flex items-center gap-2 px-2 py-0.5 rounded-md bg-slate-800/50 border border-slate-800">
            <div className={`w-1.5 h-1.5 rounded-full ${isOnline ? 'bg-emerald-500' : 'bg-red-500'} shadow-[0_0_8px_rgba(16,185,129,0.5)]`}></div>
            <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">{label}</span>
        </div>
    )
}

export default App
