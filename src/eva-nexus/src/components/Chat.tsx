import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
    Activity,
    ArrowRight,
    Bot,
    Database,
    Loader2,
    MessageSquare,
    Network,
    RefreshCw,
    Send,
    ShieldCheck,
    Sparkles,
    User,
} from 'lucide-react'

import {
    createSession,
    getCoreAgentsStatus,
    getCoreAutonomyContext,
    getCoreIntelligenceStatus,
    sendChatMessage,
    type ChatMessage as CoreChatResponse,
    type CoreAgentStatus,
} from '../services/api'
import { navigateToHiveTab } from '../navigation'

type Snapshot = Record<string, unknown>

interface ConversationMessage {
    id: string
    role: 'user' | 'assistant'
    content: string
    timestamp: string
    thoughts?: string | null
    metadata?: Record<string, unknown>
}

function getObject(value: unknown): Record<string, unknown> {
    return value && typeof value === 'object' ? value as Record<string, unknown> : {}
}

function getString(value: unknown, fallback = 'indisponible'): string {
    return typeof value === 'string' && value.trim() ? value : fallback
}

function getBoolean(value: unknown, fallback = false): boolean {
    return typeof value === 'boolean' ? value : fallback
}

function getStringList(value: unknown): string[] {
    return Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string' && item.trim().length > 0) : []
}

function formatDate(value: unknown): string {
    if (typeof value !== 'string' || !value.trim()) {
        return 'indisponible'
    }
    const date = new Date(value)
    if (Number.isNaN(date.getTime())) {
        return 'indisponible'
    }
    return date.toLocaleString('fr-FR', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' })
}

function formatTime(value: string): string {
    const date = new Date(value)
    if (Number.isNaN(date.getTime())) {
        return '--:--'
    }
    return date.toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' })
}

function buildGreeting(): ConversationMessage {
    return {
        id: 'assistant-greeting',
        role: 'assistant',
        content: 'Bonjour. Le canal EVA est de nouveau disponible dans cette vue.',
        timestamp: new Date().toISOString(),
        metadata: { expert: 'core' },
    }
}

function buildReply(payload: CoreChatResponse): ConversationMessage {
    return {
        id: `assistant-${Date.now()}`,
        role: 'assistant',
        content: getString(payload.message, 'Reponse indisponible.'),
        timestamp: new Date().toISOString(),
        thoughts: payload.thoughts ?? null,
        metadata: payload.metadata,
    }
}

function Metric({ label, value, accent = 'text-matrix' }: { label: string; value: string; accent?: string }) {
    return (
        <div className="cyber-panel hud-corners p-4">
            <div className="text-[8px] uppercase tracking-[0.18em] text-white/20">{label}</div>
            <div className={`mt-2 text-lg font-bold ${accent}`}>{value}</div>
        </div>
    )
}

function ActionCard({ title, description, targetLabel, onClick }: { title: string; description: string; targetLabel: string; onClick: () => void }) {
    return (
        <button onClick={onClick} className="cyber-panel hud-corners p-4 text-left transition-all hover:border-cyber-cyan/20 hover:bg-cyber-cyan/[0.03]">
            <div className="flex items-center justify-between gap-3">
                <div>
                    <div className="text-[10px] font-bold uppercase tracking-[0.14em] text-white/75">{title}</div>
                    <div className="mt-2 text-[10px] leading-relaxed text-white/30">{description}</div>
                </div>
                <ArrowRight size={16} className="shrink-0 text-cyber-cyan/60" />
            </div>
            <div className="mt-3 text-[9px] uppercase tracking-[0.16em] text-cyber-cyan/60">{targetLabel}</div>
        </button>
    )
}

export default function Chat() {
    const [agents, setAgents] = useState<Record<string, CoreAgentStatus>>({})
    const [autonomy, setAutonomy] = useState<Snapshot | null>(null)
    const [intelligence, setIntelligence] = useState<Snapshot | null>(null)
    const [messages, setMessages] = useState<ConversationMessage[]>([buildGreeting()])
    const [sessionId, setSessionId] = useState<string | null>(null)
    const [input, setInput] = useState('')
    const [isLoading, setIsLoading] = useState(true)
    const [isSending, setIsSending] = useState(false)
    const [chatError, setChatError] = useState<string | null>(null)
    const scrollRef = useRef<HTMLDivElement | null>(null)

    const loadCoreState = useCallback(async () => {
        const [agentsPayload, autonomyPayload, intelligencePayload] = await Promise.all([
            getCoreAgentsStatus(),
            getCoreAutonomyContext(),
            getCoreIntelligenceStatus(),
        ])
        setAgents(agentsPayload)
        setAutonomy(autonomyPayload)
        setIntelligence(intelligencePayload)
        setIsLoading(false)
    }, [])

    const startSession = useCallback(async (resetConversation: boolean) => {
        setChatError(null)
        const data = await createSession()
        setSessionId(data.session_id)
        if (resetConversation) {
            setMessages([buildGreeting()])
        }
    }, [])

    useEffect(() => {
        void loadCoreState()
        void startSession(false)
        const interval = setInterval(() => void loadCoreState(), 12000)
        return () => clearInterval(interval)
    }, [loadCoreState, startSession])

    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight
        }
    }, [messages, isSending])

    const handleSend = useCallback(async () => {
        const trimmed = input.trim()
        if (!trimmed || isSending) {
            return
        }
        const activeSessionId = sessionId || (await createSession()).session_id
        if (!sessionId) {
            setSessionId(activeSessionId)
        }
        setMessages((current) => [...current, { id: `user-${Date.now()}`, role: 'user', content: trimmed, timestamp: new Date().toISOString() }])
        setInput('')
        setIsSending(true)
        setChatError(null)
        try {
            const payload = await sendChatMessage(trimmed, activeSessionId)
            setMessages((current) => [...current, buildReply(payload)])
        } catch {
            setChatError('Le core ne repond pas pour le moment.')
            setMessages((current) => [...current, {
                id: `assistant-error-${Date.now()}`,
                role: 'assistant',
                content: 'Le canal EVA est indisponible. Le cockpit reste lisible.',
                timestamp: new Date().toISOString(),
            }])
        } finally {
            setIsSending(false)
        }
    }, [input, isSending, sessionId])

    const posture = getObject(autonomy?.posture)
    const dependencies = Object.entries(getObject(autonomy?.dependencies))
    const blockers = getStringList(posture.blockers)
    const autonomyStatus = getString(posture.status)
    const recommendedMode = getString(posture.recommended_mode)
    const agentsList = useMemo(() => Object.entries(agents), [agents])
    const onlineAgents = agentsList.filter(([, payload]) => payload.status === 'online').length
    const readyDependencies = dependencies.filter(([, value]) => {
        const state = getObject(value)
        return ['online', 'ready', 'ok'].includes(getString(state.status, getBoolean(state.ok) ? 'online' : 'offline'))
    }).length

    return (
        <div className="h-full overflow-y-auto p-4 space-y-4 animate-fade-in">
            <div className="cyber-panel hud-corners p-5 lg:p-6">
                <div className="flex flex-col gap-4 xl:flex-row xl:items-start xl:justify-between">
                    <div>
                        <div className="flex items-center gap-3 text-cyber-cyan/70">
                            <Bot size={18} />
                            <span className="text-[10px] uppercase tracking-[0.2em] font-bold">E.V.A. Core</span>
                        </div>
                        <h2 className="mt-3 font-display text-2xl font-black tracking-[0.08em] text-white/85">Dialogue EVA et cockpit core</h2>
                        <p className="mt-3 max-w-3xl text-[11px] leading-relaxed text-white/30">Le chat est revenu dans cette vue. Le cockpit core reste disponible a droite pour garder le contexte.</p>
                    </div>
                    <div className="flex flex-wrap gap-2">
                        <span className="border border-cyber-cyan/20 bg-cyber-cyan/10 px-3 py-1.5 text-[9px] font-black uppercase tracking-[0.18em] text-cyber-cyan">chat {sessionId ? 'actif' : 'degrade'}</span>
                        <span className="border border-matrix/20 bg-matrix/10 px-3 py-1.5 text-[9px] font-black uppercase tracking-[0.18em] text-matrix">autonomie {autonomyStatus}</span>
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-1 2xl:grid-cols-[1.1fr_0.9fr] gap-4">
                <section className="cyber-panel hud-corners min-h-[620px] overflow-hidden flex flex-col">
                    <div className="border-b border-white/5 px-5 py-4 flex items-center justify-between gap-3">
                        <div>
                            <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.2em] text-matrix/50">
                                <MessageSquare size={14} />
                                <span>Canal EVA</span>
                            </div>
                            <div className="mt-2 text-[11px] text-white/30">Session {sessionId ? sessionId.slice(0, 8) : 'indisponible'}</div>
                        </div>
                        <button type="button" onClick={() => void startSession(true)} disabled={isSending} className="inline-flex items-center gap-2 border border-cyber-cyan/20 bg-cyber-cyan/10 px-3 py-2 text-[9px] font-black uppercase tracking-[0.16em] text-cyber-cyan transition-all hover:bg-cyber-cyan/15 disabled:opacity-50">
                            <RefreshCw size={12} />
                            Nouvelle session
                        </button>
                    </div>

                    <div ref={scrollRef} className="flex-1 overflow-y-auto space-y-4 p-5 bg-black/20">
                        {messages.map((message) => (
                            <div key={message.id} className={`flex items-start gap-3 ${message.role === 'user' ? 'flex-row-reverse' : ''}`}>
                                <div className={`mt-1 flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border ${message.role === 'user' ? 'border-cyber-cyan/20 bg-cyber-cyan/10 text-cyber-cyan' : 'border-matrix/20 bg-matrix/10 text-matrix'}`}>
                                    {message.role === 'user' ? <User size={18} /> : <Bot size={18} />}
                                </div>
                                <div className={`max-w-[85%] border p-4 ${message.role === 'user' ? 'border-cyber-cyan/15 bg-cyber-cyan/[0.05] text-white/80' : 'border-white/[0.05] bg-white/[0.02] text-white/75'}`}>
                                    <div className="whitespace-pre-wrap text-[12px] leading-relaxed">{message.content}</div>
                                    {message.thoughts ? <div className="mt-3 border-t border-white/[0.05] pt-3 text-[10px] leading-relaxed text-white/35 whitespace-pre-wrap">{message.thoughts}</div> : null}
                                    <div className="mt-3 text-[9px] uppercase tracking-[0.14em] text-white/25">{formatTime(message.timestamp)}</div>
                                </div>
                            </div>
                        ))}
                        {isSending ? (
                            <div className="flex items-start gap-3">
                                <div className="mt-1 flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-matrix/20 bg-matrix/10 text-matrix">
                                    <Loader2 size={18} className="animate-spin" />
                                </div>
                                <div className="border border-white/[0.05] bg-white/[0.02] p-4 text-[11px] text-white/35">EVA prepare une reponse...</div>
                            </div>
                        ) : null}
                    </div>

                    <div className="border-t border-white/5 bg-black/30 p-4">
                        {chatError ? <div className="mb-3 border border-cyber-amber/20 bg-cyber-amber/10 px-3 py-2 text-[10px] text-cyber-amber">{chatError}</div> : null}
                        <div className="relative">
                            <Sparkles size={16} className="absolute left-4 top-4 text-cyber-cyan/60" />
                            <textarea
                                value={input}
                                onChange={(event) => setInput(event.target.value)}
                                onKeyDown={(event) => {
                                    if (event.key === 'Enter' && !event.shiftKey) {
                                        event.preventDefault()
                                        void handleSend()
                                    }
                                }}
                                rows={3}
                                placeholder="Parlez a EVA. Exemple: resume le run scalp actuel."
                                className="w-full resize-none border border-white/[0.06] bg-white/[0.02] py-3 pl-11 pr-24 text-[12px] text-white/80 outline-none transition-all placeholder:text-white/20 focus:border-cyber-cyan/25 focus:bg-cyber-cyan/[0.03]"
                            />
                            <button type="button" onClick={() => void handleSend()} disabled={!input.trim() || isSending} className="absolute bottom-3 right-3 inline-flex items-center gap-2 border border-cyber-cyan/20 bg-cyber-cyan/10 px-3 py-2 text-[10px] font-black uppercase tracking-[0.16em] text-cyber-cyan transition-all hover:bg-cyber-cyan/15 disabled:cursor-not-allowed disabled:opacity-50">
                                <Send size={14} />
                                Envoyer
                            </button>
                        </div>
                    </div>
                </section>

                <section className="space-y-4">
                    <div className="grid grid-cols-2 gap-3">
                        <Metric label="Agents en ligne" value={String(onlineAgents)} />
                        <Metric label="Dependances pretes" value={String(readyDependencies)} />
                        <Metric label="Mode recommande" value={recommendedMode.replace(/_/g, ' ')} accent="text-cyber-cyan" />
                        <Metric label="Dernier snapshot" value={formatDate(autonomy?.generated_at)} accent="text-cyber-pink" />
                    </div>
                    <div className="cyber-panel hud-corners p-4 space-y-3">
                        <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.2em] text-cyber-amber/60">
                            <ShieldCheck size={14} />
                            <span>Posture</span>
                        </div>
                        <div className="text-[11px] leading-relaxed text-white/35">
                            Trading autorise: {getBoolean(posture.can_trade) ? 'oui' : 'non'}
                            <br />
                            Memoire prete: {getBoolean(posture.memory_ready) ? 'oui' : 'non'}
                            <br />
                            Blocages: {blockers.length > 0 ? blockers.join(', ') : 'aucun'}
                            <br />
                            RLM: {getString(getObject(intelligence?.rlm).status)}
                        </div>
                    </div>
                    <div className="cyber-panel hud-corners p-4 space-y-3">
                        <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.2em] text-matrix/50">
                            <Activity size={14} />
                            <span>Agents</span>
                        </div>
                        {isLoading ? <div className="text-[10px] text-white/25">Chargement...</div> : (
                            <div className="space-y-2">
                                {agentsList.slice(0, 6).map(([name, payload]) => (
                                    <div key={name} className="flex items-center justify-between gap-3 border border-white/[0.05] bg-white/[0.02] px-3 py-2 text-[10px]">
                                        <span className="font-bold uppercase tracking-[0.12em] text-white/70">{name}</span>
                                        <span className="text-white/35">{getString(payload.status)}</span>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                </section>
            </div>

            <div className="grid grid-cols-1 xl:grid-cols-[1fr_1fr] gap-4">
                <section className="cyber-panel hud-corners p-4 space-y-3">
                    <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.2em] text-cyber-cyan/60">
                        <Database size={14} />
                        <span>Dependances memoire</span>
                    </div>
                    {dependencies.length === 0 ? <div className="text-[10px] text-white/25">Dependances indisponibles.</div> : (
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                            {dependencies.slice(0, 6).map(([name, value]) => {
                                const state = getObject(value)
                                return (
                                    <div key={name} className="border border-white/[0.05] bg-white/[0.02] p-3">
                                        <div className="text-[10px] font-bold uppercase tracking-[0.12em] text-white/70">{name}</div>
                                        <div className="mt-2 text-[10px] text-white/30">{getString(state.status, getBoolean(state.ok) ? 'online' : 'offline')}</div>
                                    </div>
                                )
                            })}
                        </div>
                    )}
                </section>

                <section className="cyber-panel hud-corners p-4 space-y-3">
                    <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.2em] text-matrix/50">
                        <Network size={14} />
                        <span>Navigation croisee</span>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                        <ActionCard title="Voir en memoire" description="Basculer vers Memory Store." targetLabel="Memory Store" onClick={() => navigateToHiveTab('memory')} />
                        <ActionCard title="Voir en revue" description="Ouvrir Knowledge Vault." targetLabel="Knowledge Vault" onClick={() => navigateToHiveTab('knowledge')} />
                        <ActionCard title="Voir le graphe" description="Ouvrir Nexus Graph." targetLabel="Nexus Graph" onClick={() => navigateToHiveTab('graph')} />
                    </div>
                </section>
            </div>
        </div>
    )
}
