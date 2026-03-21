import { lazy, Suspense, useEffect, useState } from 'react'
import {
    LayoutDashboard,
    MessageSquare,
    TrendingUp,
    Settings,
    Activity,
    Cpu,
    Briefcase,
    Zap,
    Lock,
    Globe,
    Share2,
    Database,
    Camera,
    BookOpen,
    Radio
} from 'lucide-react'

import MatrixRain from './components/MatrixRain'
const Chat = lazy(() => import('./components/Chat'))
const TradingPanel = lazy(() => import('./components/TradingPanel'))
const Dashboard = lazy(() => import('./components/Dashboard'))
const MonitoringView = lazy(() => import('./components/MonitoringView'))
const OSINTView = lazy(() => import('./components/OSINTView'))
const FactoriesView = lazy(() => import('./components/FactoriesView'))
const AdminPanel = lazy(() => import('./components/AdminPanel'))
const GraphView = lazy(() => import('./components/GraphView'))
const MemoryExplorer = lazy(() => import('./components/MemoryExplorer'))
const MuseFactory = lazy(() => import('./components/MuseFactory'))
const KnowledgeVault = lazy(() => import('./components/KnowledgeVault'))
const AgentFeed = lazy(() => import('./components/AgentFeed'))

// Services
import { getAllNodesHealth } from './services/api'
import { onHiveNavigate, type HiveTabId } from './navigation'

function App() {
    const [activeTab, setActiveTab] = useState<HiveTabId>('chat')
    const [systemStatus, setSystemStatus] = useState({ core: 'online', banker: 'online', sentinel: 'online' })

    const tabTitles: Record<HiveTabId, string> = {
        dashboard: 'Dashboard',
        chat: 'E.V.A. Core',
        trading: 'Trading Floor',
        graph: 'Nexus Graph',
        memory: 'Memory Store',
        monitoring: 'Hardware Stats',
        osint: 'Intelligence',
        factories: 'Usines',
        admin: 'Admin Panel',
        settings: 'Settings',
        muse: 'Muse Factory',
        knowledge: 'Knowledge Vault',
        agentfeed: 'Agent Feed',
    }

    // Synchronise periodiquement l'etat global des services.
    useEffect(() => {
        const fetchStatus = async () => {
            try {
                const status = await getAllNodesHealth()
                const core = status.find((node) => node.name === 'EVA Core')?.status || 'offline'
                const banker = status.find((node) => node.name === 'Banker')?.status || 'offline'
                const sentinel = status.find((node) => node.name === 'Sentinel')?.status || 'offline'
                setSystemStatus({
                    core,
                    banker,
                    sentinel,
                })
            } catch (e) {
                console.error("Erreur de synchronisation du statut", e)
            }
        }
        fetchStatus()
        const interval = setInterval(fetchStatus, 30000)
        return () => clearInterval(interval)
    }, [])

    useEffect(() => onHiveNavigate(({ tab }) => setActiveTab(tab)), [])

    return (
        <div className="flex h-screen w-full bg-black text-white overflow-hidden neural-bg scanline relative">
            {/* Fond visuel matriciel */}
            <MatrixRain />

            {/* Navigation laterale */}
            <aside className="w-20 lg:w-64 flex flex-col border-r border-matrix/10 glass-heavy relative z-20">
                <div className="p-6 flex items-center gap-4">
                    <div className="w-10 h-10 border border-matrix/30 bg-matrix/5 rounded-lg flex items-center justify-center shadow-[0_0_15px_rgba(0,255,65,0.2)] animate-pulse-glow">
                        <Cpu className="w-6 h-6 text-matrix" />
                    </div>
                    <div className="hidden lg:block overflow-hidden">
                        <h1 className="font-display font-black text-lg tracking-tighter neon-text">THE HIVE</h1>
                        <p className="text-[8px] uppercase font-bold text-matrix/40 tracking-[0.3em] truncate">Neural Protocol v1.4</p>
                    </div>
                </div>

                <nav className="flex-grow flex flex-col gap-1 p-3 overflow-y-auto custom-scrollbar">
                    <NavItem
                        id="dashboard"
                        icon={<LayoutDashboard size={18} />}
                        label="Dashboard"
                        active={activeTab === 'dashboard'}
                        onClick={() => setActiveTab('dashboard')}
                    />
                    <NavItem
                        id="chat"
                        icon={<MessageSquare size={18} />}
                        label="E.V.A. Core"
                        active={activeTab === 'chat'}
                        onClick={() => setActiveTab('chat')}
                    />
                    <NavItem
                        id="trading"
                        icon={<TrendingUp size={18} />}
                        label="Trading Floor"
                        active={activeTab === 'trading'}
                        onClick={() => setActiveTab('trading')}
                    />
                    <NavItem
                        id="graph"
                        icon={<Share2 size={18} />}
                        label="Nexus Graph"
                        active={activeTab === 'graph'}
                        onClick={() => setActiveTab('graph')}
                    />
                    <NavItem
                        id="memory"
                        icon={<Database size={18} />}
                        label="Memory Store"
                        active={activeTab === 'memory'}
                        onClick={() => setActiveTab('memory')}
                    />
                    <div className="my-2 border-t border-matrix/5 mx-2"></div>
                    <NavItem
                        id="monitoring"
                        icon={<Activity size={18} />}
                        label="Hardware Stats"
                        active={activeTab === 'monitoring'}
                        onClick={() => setActiveTab('monitoring')}
                    />
                    <NavItem
                        id="osint"
                        icon={<Globe size={18} />}
                        label="Intelligence"
                        active={activeTab === 'osint'}
                        onClick={() => setActiveTab('osint')}
                    />
                    <NavItem
                        id="factories"
                        icon={<Briefcase size={18} />}
                        label="Usines"
                        active={activeTab === 'factories'}
                        onClick={() => setActiveTab('factories')}
                    />
                    <NavItem
                        id="muse"
                        icon={<Camera size={18} />}
                        label="Muse Factory"
                        active={activeTab === 'muse'}
                        onClick={() => setActiveTab('muse')}
                    />
                    <NavItem
                        id="knowledge"
                        icon={<BookOpen size={18} />}
                        label="Knowledge Vault"
                        active={activeTab === 'knowledge'}
                        onClick={() => setActiveTab('knowledge')}
                    />
                    <NavItem
                        id="agentfeed"
                        icon={<Radio size={18} />}
                        label="Agent Feed"
                        active={activeTab === 'agentfeed'}
                        onClick={() => setActiveTab('agentfeed')}
                    />

                    <div className="mt-auto space-y-1">
                        <NavItem
                            id="admin"
                            icon={<Lock size={18} />}
                            label="Admin Panel"
                            active={activeTab === 'admin'}
                            onClick={() => setActiveTab('admin')}
                        />
                        <NavItem
                            id="settings"
                            icon={<Settings size={18} />}
                            label="Settings"
                            active={activeTab === 'settings'}
                            onClick={() => setActiveTab('settings')}
                        />
                    </div>
                </nav>

                    {/* Bloc profil utilisateur */}
                <div className="p-4 mt-auto border-t border-matrix/10">
                    <div className="flex items-center gap-3 p-2 rounded-lg hover:bg-white/5 transition-all cursor-pointer group">
                        <div className="w-8 h-8 border border-matrix/20 bg-matrix/5 flex items-center justify-center text-[10px] font-bold text-matrix">JM</div>
                        <div className="hidden lg:block">
                            <p className="text-[10px] font-bold text-white/80">John Moncel</p>
                            <p className="text-[8px] text-matrix/50 uppercase tracking-widest font-black">Master</p>
                        </div>
                    </div>
                </div>
            </aside>

            {/* Zone de contenu principale */}
            <main className="flex-grow flex flex-col relative overflow-hidden h-full z-10 bg-black/40">
                {/* Barre de statut globale */}
                <header className="h-[var(--header-height)] border-b border-matrix/10 flex items-center justify-between px-6 glass backdrop-blur-md">
                    <div className="flex items-center gap-6">
                        <h2 className="text-[11px] font-black uppercase tracking-[0.3em] neon-text">
                            // {tabTitles[activeTab].toUpperCase()}
                        </h2>
                        <div className="flex items-center gap-2">
                            <StatusBadge label="CORE" status={systemStatus.core} />
                            <StatusBadge label="BANKER" status={systemStatus.banker} />
                            <StatusBadge label="SENTINEL" status={systemStatus.sentinel} />
                        </div>
                    </div>

                    <div className="flex items-center gap-4 text-[9px] font-mono font-bold text-matrix/40 uppercase tracking-widest">
                        <div className="flex items-center gap-2 px-3 py-1 bg-matrix/5 border border-matrix/10 rounded">
                            <Zap size={10} className="text-matrix" />
                            <span>Charge systeme: optimale</span>
                        </div>
                        <div className="hidden md:block px-3 py-1 border border-matrix/5 rounded">
                            Phase Genesis | {new Date().toLocaleTimeString()}
                        </div>
                    </div>
                </header>

                {/* Conteneur de vue partage */}
                <div className="flex-grow overflow-hidden relative">
                    <div className="absolute inset-0 p-4 lg:p-6 overflow-hidden">
                        <Suspense fallback={<ViewLoadingState label={tabTitles[activeTab]} />}>
                            <ViewSwitcher activeTab={activeTab} />
                        </Suspense>
                    </div>
                </div>
            </main>
        </div>
    )
}

function ViewSwitcher({ activeTab }: { activeTab: HiveTabId }) {
    switch (activeTab) {
        case 'dashboard': return <Dashboard />
        case 'chat': return <Chat />
        case 'trading': return <TradingPanel />
        case 'graph': return <GraphView />
        case 'memory': return <MemoryExplorer />
        case 'monitoring': return <MonitoringView />
        case 'osint': return <OSINTView />
        case 'factories': return <FactoriesView />
        case 'muse': return <MuseFactory />
        case 'knowledge': return <KnowledgeVault />
        case 'agentfeed': return <AgentFeed />
        case 'admin': return <AdminPanel />
        case 'settings': return (
            <div className="h-full flex flex-col items-center justify-center text-center space-y-4 opacity-50">
                <Settings size={64} className="text-matrix" />
                <div className="font-mono">
                    <p className="text-matrix text-sm uppercase tracking-[0.2em] font-bold">Terminal Configuration</p>
                    <p className="text-[10px] text-white/40 mt-2">ACCESS_LEVEL: MASTER_ONLY<br />STATUS: READ_ONLY_REMOTE</p>
                </div>
            </div>
        )
        default: return <Chat />
    }
}

function NavItem({ icon, label, active, onClick }: { id: string, icon: any, label: string, active?: boolean, onClick: () => void }) {
    return (
        <button
            onClick={onClick}
            className={`
                flex items-center gap-3 p-3 rounded-lg transition-all duration-200 group relative
                ${active
                    ? 'bg-matrix/10 text-matrix border border-matrix/20 shadow-[0_0_15px_rgba(0,255,65,0.1)]'
                    : 'text-white/40 hover:bg-white/5 hover:text-white/80 border border-transparent'}
            `}
        >
            {active && (
                <div className="absolute left-0 top-1/2 -translate-y-1/2 w-0.5 h-6 bg-matrix shadow-[0_0_10px_var(--matrix)]" />
            )}
            <span className={`${active ? 'text-matrix' : 'text-white/20 group-hover:text-white/60'}`}>{icon}</span>
            <span className="hidden lg:block text-[10px] font-bold uppercase tracking-[0.15em] font-display">{label}</span>
        </button>
    )
}

function StatusBadge({ label, status }: { label: string, status: string }) {
    const isOnline = status === 'online'
    return (
        <div className="flex items-center gap-2 px-2 py-1 bg-black/40 border border-matrix/10 rounded">
            <div className={`w-1.5 h-1.5 rounded-full ${isOnline ? 'bg-matrix animate-pulse shadow-[0_0_8px_var(--matrix)]' : 'bg-red-500'}`} />
            <span className="text-[8px] font-bold text-matrix/60 font-mono tracking-tighter">{label}</span>
        </div>
    )
}

function ViewLoadingState({ label }: { label: string }) {
    return (
        <div className="h-full flex flex-col items-center justify-center text-center space-y-4 opacity-70">
            <div className="w-12 h-12 rounded-full border-2 border-matrix/20 border-t-matrix animate-spin" />
            <div className="font-mono">
                <p className="text-matrix text-sm uppercase tracking-[0.2em] font-bold">{label}</p>
                <p className="text-[10px] text-white/35 mt-2">Chargement de la vue...</p>
            </div>
        </div>
    )
}

export default App
