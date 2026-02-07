import { useState, useRef, useEffect } from 'react'
import { Send, Bot, User, Loader2, Sparkles } from 'lucide-react'
import { sendChatMessage, createSession } from '../services/api'

interface Message {
    id: string
    role: 'user' | 'assistant'
    content: string
    timestamp: Date
    metadata?: {
        expert?: string
        confidence?: number
    }
}

export default function Chat() {
    const [messages, setMessages] = useState<Message[]>([
        {
            id: '1',
            role: 'assistant',
            content: "Bonjour Maître. Je suis E.V.A. L'orchestrateur est prêt. Comment puis-je vous assister aujourd'hui ?",
            timestamp: new Date()
        }
    ])
    const [input, setInput] = useState('')
    const [isLoading, setIsLoading] = useState(false)
    const [sessionId, setSessionId] = useState<string | null>(null)
    const scrollRef = useRef<HTMLDivElement>(null)

    useEffect(() => {
        const init = async () => {
            try {
                const data = await createSession()
                setSessionId(data.session_id)
            } catch (e) {
                console.error("Session init error", e)
            }
        }
        init()
    }, [])

    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight
        }
    }, [messages])

    const handleSend = async () => {
        if (!input.trim() || isLoading) return

        const userMsg: Message = {
            id: Date.now().toString(),
            role: 'user',
            content: input,
            timestamp: new Date()
        }

        setMessages(prev => [...prev, userMsg])
        setInput('')
        setIsLoading(true)

        try {
            const data = await sendChatMessage(input, sessionId || '00000000-0000-0000-0000-000000000000')

            const botMsg: Message = {
                id: (Date.now() + 1).toString(),
                role: 'assistant',
                content: data.message,
                timestamp: new Date(),
                metadata: data.metadata
            }
            setMessages(prev => [...prev, botMsg])
        } catch (e) {
            const errorMsg: Message = {
                id: (Date.now() + 1).toString(),
                role: 'assistant',
                content: "Désolé Maître, une erreur est survenue lors de la communication avec le Core. Veuillez vérifier les logs.",
                timestamp: new Date()
            }
            setMessages(prev => [...prev, errorMsg])
        } finally {
            setIsLoading(false)
        }
    }

    return (
        <div className="flex flex-col h-full glass rounded-[2rem] overflow-hidden animate-fade-in relative shadow-2xl">
            {/* Ambient Background Glow */}
            <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-sky-500/50 to-transparent opacity-50" />

            {/* Chat Messages */}
            <div
                ref={scrollRef}
                className="flex-grow overflow-y-auto p-8 space-y-8 scroll-smooth"
            >
                {messages.map((msg) => (
                    <div
                        key={msg.id}
                        className={`flex items-start gap-5 ${msg.role === 'user' ? 'flex-row-reverse' : ''} animate-fade-in`}
                    >
                        <div className={`
              w-11 h-11 rounded-xl flex items-center justify-center shrink-0 shadow-lg transition-transform hover:scale-105
              ${msg.role === 'user'
                                ? 'bg-gradient-to-br from-indigo-500 to-indigo-700 shadow-indigo-500/20'
                                : 'bg-slate-900 border border-white/10 shadow-black/40'}
            `}>
                            {msg.role === 'user' ? <User size={22} className="text-white" /> : <Bot size={22} className="text-sky-400" />}
                        </div>

                        <div className={`
              max-w-[75%] p-5 rounded-2xl text-[13px] leading-relaxed relative group shadow-xl transition-all
              ${msg.role === 'user'
                                ? 'bg-indigo-500/5 border border-indigo-500/20 text-slate-100'
                                : 'bg-white/[0.03] border border-white/10 text-slate-200'}
            `}>
                            {msg.metadata?.expert && msg.role === 'assistant' && (
                                <div className="absolute -top-3 left-4 px-3 py-1 bg-slate-900 border border-sky-500/30 rounded-lg text-[9px] font-black text-sky-400 uppercase tracking-[0.2em] shadow-2xl">
                                    Agent {msg.metadata.expert}
                                </div>
                            )}
                            <div className="whitespace-pre-wrap">{msg.content}</div>
                            <div className="mt-3 flex items-center justify-between text-[10px] font-bold text-slate-500 uppercase tracking-widest">
                                <span>{msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                                {msg.metadata?.confidence && (
                                    <div className="flex items-center gap-2 px-2 py-0.5 bg-white/5 rounded-md opacity-0 group-hover:opacity-100 transition-opacity">
                                        <div className="w-1 h-1 rounded-full bg-sky-500" />
                                        <span>Confiance: {Math.round(msg.metadata.confidence * 100)}%</span>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                ))}
                {isLoading && (
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-2xl bg-slate-800 border border-slate-700 flex items-center justify-center">
                            <Loader2 size={20} className="text-sky-400 animate-spin" />
                        </div>
                        <div className="p-4 rounded-2xl bg-slate-800/40 border border-slate-800 text-slate-400 text-xs italic">
                            EVA est en train de réfléchir...
                        </div>
                    </div>
                )}
            </div>

            {/* Input Area */}
            <div className="p-6 border-t border-white/5 bg-black/20 backdrop-blur-xl">
                <div className="relative max-w-5xl mx-auto flex items-center gap-4">
                    <div className="absolute left-5 top-1/2 -translate-y-1/2 text-sky-400 group-focus-within:animate-pulse">
                        <Sparkles size={20} />
                    </div>
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                        placeholder="Envoyez une commande à la Ruche..."
                        className="w-full bg-white/[0.03] border border-white/10 focus:border-sky-500/50 focus:ring-4 focus:ring-sky-500/5 rounded-2xl py-5 pl-14 pr-16 text-sm transition-all outline-none placeholder:text-slate-600 font-medium"
                    />
                    <button
                        onClick={handleSend}
                        disabled={!input.trim() || isLoading}
                        className={`
              absolute right-3.5 top-1/2 -translate-y-1/2 p-3 rounded-xl transition-all duration-300
              ${input.trim() && !isLoading
                                ? 'bg-gradient-to-br from-sky-400 to-indigo-600 text-white shadow-xl shadow-sky-500/20 hover:scale-105 active:scale-95'
                                : 'bg-white/5 text-slate-600 cursor-not-allowed'}
            `}
                    >
                        <Send size={20} />
                    </button>
                </div>
                <div className="flex items-center justify-center gap-4 mt-3 opacity-40">
                    <div className="h-[1px] w-12 bg-gradient-to-r from-transparent to-slate-500" />
                    <p className="text-[9px] font-black uppercase tracking-[0.3em] text-slate-400">
                        Constitution Genesis 1.0 • Sécurité Active
                    </p>
                    <div className="h-[1px] w-12 bg-gradient-to-l from-transparent to-slate-500" />
                </div>
            </div>
        </div>
    )
}
