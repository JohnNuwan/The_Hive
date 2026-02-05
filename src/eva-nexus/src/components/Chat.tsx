import { useState, useRef, useEffect } from 'react'
import { Send, Bot, User, Loader2, Sparkles } from 'lucide-react'
import { sendChatMessage, createSession } from '../services/api'

interface Message {
    id: string
    role: 'user' | 'assistant'
    content: string
    timestamp: Date
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
            const data = await sendChatMessage(input, sessionId || '')

            const botMsg: Message = {
                id: (Date.now() + 1).toString(),
                role: 'assistant',
                content: data.message,
                timestamp: new Date()
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
        <div className="flex flex-col h-full glass rounded-3xl overflow-hidden animate-fade-in">
            {/* Chat Messages */}
            <div
                ref={scrollRef}
                className="flex-grow overflow-y-auto p-6 space-y-6"
            >
                {messages.map((msg) => (
                    <div
                        key={msg.id}
                        className={`flex items-start gap-4 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}
                    >
                        <div className={`
              w-10 h-10 rounded-2xl flex items-center justify-center shrink-0
              ${msg.role === 'user' ? 'bg-indigo-600 shadow-indigo-500/20' : 'bg-slate-800 border border-slate-700'}
            `}>
                            {msg.role === 'user' ? <User size={20} /> : <Bot size={20} className="text-sky-400" />}
                        </div>

                        <div className={`
              max-w-[80%] p-4 rounded-2xl text-sm leading-relaxed
              ${msg.role === 'user'
                                ? 'bg-indigo-600/10 border border-indigo-500/20 text-slate-100'
                                : 'bg-slate-800/40 border border-slate-800 text-slate-200'}
            `}>
                            {msg.content}
                            <div className="mt-2 text-[10px] text-slate-500">
                                {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
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
            <div className="p-5 border-t border-slate-800 bg-slate-900/40 backdrop-blur-md">
                <div className="relative max-w-4xl mx-auto flex items-center gap-3">
                    <div className="absolute left-4 top-1/2 -translate-y-1/2 text-sky-500">
                        <Sparkles size={18} />
                    </div>
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                        placeholder="Commandez votre entité... (ex: 'Quel est l'état du DAX ?')"
                        className="w-full bg-slate-950/50 border border-slate-700/50 focus:border-sky-500/50 focus:ring-1 focus:ring-sky-500/30 rounded-2xl py-4 pl-12 pr-14 text-sm transition-all outline-none placeholder:text-slate-600"
                    />
                    <button
                        onClick={handleSend}
                        disabled={!input.trim() || isLoading}
                        className={`
              absolute right-3 top-1/2 -translate-y-1/2 p-2.5 rounded-xl transition-all
              ${input.trim() && !isLoading
                                ? 'bg-sky-500 text-white shadow-lg shadow-sky-500/20 hover:bg-sky-400'
                                : 'bg-slate-800 text-slate-500 cursor-not-allowed'}
            `}
                    >
                        <Send size={18} />
                    </button>
                </div>
                <p className="text-center text-[10px] text-slate-600 mt-2">
                    Protocoles de sécurité actifs • Constitution Genesis 1.0
                </p>
            </div>
        </div>
    )
}
