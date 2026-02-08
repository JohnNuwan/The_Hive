import { useState, useEffect } from 'react'
import {
    LayoutDashboard,
    MessageSquare,
    TrendingUp,
    Shield,
    Settings,
    Activity,
    Cpu,
    Briefcase,
    Zap,
    Lock,
    Globe
} from 'lucide-react'

// Core Components
import Chat from './components/Chat'
import TradingPanel from './components/TradingPanel'
import Dashboard from './components/Dashboard'
import MonitoringView from './components/MonitoringView'
import OSINTView from './components/OSINTView'
import FactoriesView from './components/FactoriesView'
import AdminPanel from './components/AdminPanel'
import MatrixRain from './components/MatrixRain'

// Services
import { getStatus } from './services/api'

type TabId = 'dashboard' | 'chat' | 'trading' | 'monitoring' | 'osint' | 'factories' | 'admin' | 'settings'

function App() {
    const [activeTab, setActiveTab] = useState<TabId>('chat')
    const [systemStatus, setSystemStatus] = useState({ core: 'online', banker: 'online', sentinel: 'online' })
    const [securityData, setSecurityData] = useState<any>(null)

    // WebSocket / Status Poll
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
                console.error("Status Sync Error", e)
            }
        }
        fetchStatus()
        const interval = setInterval(fetchStatus, 30000)
        return () => clearInterval(interval)
    }, [])

    return (
        <div className="flex h-screen w-full bg-black text-white overflow-hidden neural-bg scanline relative">
            {/* Matrix Rain Background */}
            <MatrixRain />

            {/* Sidebar Navigation */}
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
                        label="Enterprise"
                        active={activeTab === 'factories'}
                        onClick={() => setActiveTab('factories')}
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

                {/* User Profile Hook */}
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

            {/* Main Content Area */}
            <main className="flex-grow flex flex-col relative overflow-hidden h-full z-10 bg-black/40">
                {/* Global Status Bar */}
                <header className="h-[var(--header-height)] border-b border-matrix/10 flex items-center justify-between px-6 glass backdrop-blur-md">
                    <div className="flex items-center gap-6">
                        <h2 className="text-[11px] font-black uppercase tracking-[0.3em] neon-text">
                            // {activeTab.replace(/([A-Z])/g, ' $1').toUpperCase()}
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
                            <span>System Load: Optimal</span>
                        </div>
                        <div className="hidden md:block px-3 py-1 border border-matrix/5 rounded">
                            Genesis Phase • {new Date().toLocaleTimeString()}
                        </div>
                    </div>
                </header>

                {/* Shared View Container */}
                <div className="flex-grow overflow-hidden relative">
                    <div className="absolute inset-0 p-4 lg:p-6 overflow-hidden">
                        <ViewSwitcher activeTab={activeTab} />
                    </div>
                </div>
            </main>
        </div>
    )
}

function ViewSwitcher({ activeTab }: { activeTab: TabId }) {
    switch (activeTab) {
        case 'dashboard': return <Dashboard />
        case 'chat': return <Chat />
        case 'trading': return <TradingPanel />
        case 'monitoring': return <MonitoringView />
        case 'osint': return <OSINTView />
        case 'factories': return <FactoriesView />
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

export default App
