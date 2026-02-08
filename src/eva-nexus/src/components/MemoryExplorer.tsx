import { useEffect, useState } from 'react'
import { getMemoryGraph, safeFetch } from '../services/api'
import { Database, Search, Filter, Clock, User, Bot, ChevronRight, Binary } from 'lucide-react'

interface MemoryFragment {
    id: string
    content: string
    role: string
    timestamp: string
}

export default function MemoryExplorer() {
    const [fragments, setFragments] = useState<MemoryFragment[]>([])
    const [isLoading, setIsLoading] = useState(true)
    const [searchTerm, setSearchTerm] = useState('')

    useEffect(() => {
        const loadMemory = async () => {
            try {
                // Pour l'instant on utilise safeFetch car getMemoryFragments n'est pas encore exporté proprement
                const data = await safeFetch<MemoryFragment[]>('/api/core/memory/fragments?limit=50', [])
                setFragments(data)
            } catch (error) {
                console.error("Failed to load memory fragments", error)
            } finally {
                setIsLoading(false)
            }
        }
        loadMemory()
    }, [])

    const filtered = fragments.filter(f =>
        f.content.toLowerCase().includes(searchTerm.toLowerCase()) ||
        f.role.toLowerCase().includes(searchTerm.toLowerCase())
    )

    return (
        <div className="flex flex-col h-full gap-6 animate-fade-in">
            {/* Header & Search */}
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 glass p-6 rounded-[2rem] border border-white/5">
                <div className="flex items-center gap-4">
                    <div className="w-12 h-12 bg-sky-500/10 border border-sky-500/20 rounded-2xl flex items-center justify-center">
                        <Database className="text-sky-400" size={24} />
                    </div>
                    <div>
                        <h2 className="text-lg font-black text-white uppercase tracking-tighter">Memory Explorer</h2>
                        <p className="text-[10px] text-slate-500 font-bold uppercase tracking-[0.2em]">Qdrant Neural Storage • {fragments.length} Fragments</p>
                    </div>
                </div>

                <div className="relative group flex-grow max-w-md">
                    <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-500 group-focus-within:text-sky-400 transition-colors" size={18} />
                    <input
                        type="text"
                        placeholder="Rechercher dans la ruche..."
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        className="w-full bg-white/[0.03] border border-white/10 rounded-2xl py-3 pl-12 pr-4 text-sm outline-none focus:border-sky-500/50 transition-all font-medium"
                    />
                </div>
            </div>

            {/* Fragments List */}
            <div className="flex-grow glass rounded-[2.5rem] border border-white/5 overflow-hidden flex flex-col">
                <div className="p-4 border-b border-white/5 bg-white/[0.02] flex items-center justify-between text-[10px] font-black text-slate-500 uppercase tracking-widest px-8">
                    <div className="flex items-center gap-8">
                        <span className="w-8">Type</span>
                        <span>Contenu du Fragment</span>
                    </div>
                    <span>Timestamp</span>
                </div>

                <div className="flex-grow overflow-y-auto p-4 space-y-2 custom-scrollbar">
                    {isLoading ? (
                        <div className="flex items-center justify-center h-full opacity-30 animate-pulse">
                            <Binary size={48} className="text-sky-500" />
                        </div>
                    ) : filtered.length === 0 ? (
                        <div className="flex flex-col items-center justify-center h-full opacity-20 py-20">
                            <Search size={64} className="mb-4" />
                            <p className="font-black uppercase tracking-widest text-xs">Aucun fragment trouvé</p>
                        </div>
                    ) : (
                        filtered.map((fragment) => (
                            <div
                                key={fragment.id}
                                className="flex items-center justify-between p-4 rounded-2xl hover:bg-white/[0.03] border border-transparent hover:border-white/5 transition-all group cursor-pointer"
                            >
                                <div className="flex items-center gap-8 overflow-hidden">
                                    <div className={`w-10 h-10 rounded-xl flex items-center justify-center shrink-0 ${fragment.role === 'user' ? 'bg-indigo-500/10 text-indigo-400' : 'bg-sky-500/10 text-sky-400'}`}>
                                        {fragment.role === 'user' ? <User size={18} /> : <Bot size={18} />}
                                    </div>
                                    <div className="overflow-hidden">
                                        <p className="text-sm text-slate-300 truncate group-hover:text-white transition-colors">{fragment.content}</p>
                                        <div className="flex items-center gap-3 mt-1">
                                            <span className="text-[9px] font-black text-slate-600 uppercase tracking-widest">ID: {fragment.id.slice(0, 8)}</span>
                                            <div className="w-1 h-1 rounded-full bg-slate-800" />
                                            <span className="text-[9px] font-black text-slate-600 uppercase tracking-widest">{fragment.role}</span>
                                        </div>
                                    </div>
                                </div>
                                <div className="flex items-center gap-6 shrink-0">
                                    <div className="text-right">
                                        <p className="text-[10px] font-bold text-slate-400">{new Date(fragment.timestamp).toLocaleDateString()}</p>
                                        <p className="text-[9px] text-slate-600">{new Date(fragment.timestamp).toLocaleTimeString()}</p>
                                    </div>
                                    <ChevronRight size={16} className="text-slate-700 group-hover:text-sky-500 transition-colors" />
                                </div>
                            </div>
                        ))
                    )}
                </div>
            </div>
        </div>
    )
}
