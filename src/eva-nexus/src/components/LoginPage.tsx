import { useState, useEffect, useRef } from 'react'
import MatrixRain from './MatrixRain'
import { useAuthStore } from '../stores/authStore'

export default function LoginPage() {
    const [username, setUsername] = useState('')
    const [password, setPassword] = useState('')
    const [isAnimating, setIsAnimating] = useState(true)
    const [bootLines, setBootLines] = useState<string[]>([])
    const { login, isLoading, error, clearError } = useAuthStore()
    const inputRef = useRef<HTMLInputElement>(null)

    // Boot animation
    useEffect(() => {
        const lines = [
            'INITIALIZING SECURE LINK...',
            'LOADING ENCRYPTION KEYS... AES-256',
            'VERIFYING CERTIFICATE CHAIN... OK',
            'CONNECTING TO NEURAL CORE... OK',
            'AUTHENTICATION MODULE... READY',
            '',
            'AWAITING CREDENTIALS.',
        ]

        let i = 0
        const timer = setInterval(() => {
            if (i < lines.length) {
                setBootLines(prev => [...prev, lines[i]])
                i++
            } else {
                clearInterval(timer)
                setTimeout(() => {
                    setIsAnimating(false)
                    inputRef.current?.focus()
                }, 400)
            }
        }, 100)

        return () => clearInterval(timer)
    }, [])

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault()
        if (!username.trim() || !password.trim()) return
        clearError()
        await login(username, password)
    }

    return (
        <div className="w-screen h-screen bg-black flex items-center justify-center relative overflow-hidden">
            <MatrixRain />

            {/* Grid background */}
            <div className="absolute inset-0 z-[1] grid-bg pointer-events-none" />

            {/* Scanlines */}
            <div className="absolute inset-0 z-[2] pointer-events-none scanlines" />

            {/* Main card */}
            <div className="relative z-10 w-full max-w-md p-6">
                {/* Logo */}
                <div className="text-center mb-8">
                    <div className="text-5xl mb-3 animate-glow-pulse inline-block">🐝</div>
                    <h1 className="font-display text-3xl font-black tracking-[0.3em] neon-text">THE HIVE</h1>
                    <div className="text-[9px] text-matrix/40 tracking-[0.5em] mt-1">NEURAL COMMAND INTERFACE</div>
                </div>

                {/* Terminal boot */}
                {isAnimating ? (
                    <div className="cyber-panel hud-corners p-5 font-mono">
                        <div className="text-[10px] text-matrix/30 mb-2 tracking-wider">{'>'} SECURITY HANDSHAKE</div>
                        <div className="space-y-0.5">
                            {bootLines.map((line, i) => {
                                const l = line || ''
                                const isSuccess = l.includes('OK') || l.includes('READY') || l.includes('AES')
                                const isAwait = l.includes('AWAITING')
                                return (
                                    <div
                                        key={i}
                                        className={`text-[11px] animate-fade-in ${isSuccess ? 'text-matrix' : isAwait ? 'text-cyber-cyan font-bold' : 'text-white/40'}`}
                                    >
                                        {l && <span className="text-matrix/30 mr-2">{'>'}</span>}
                                        {l}
                                    </div>
                                )
                            })}
                        </div>
                        <div className="mt-1 terminal-cursor text-[11px]">&nbsp;</div>
                    </div>
                ) : (
                    <form onSubmit={handleSubmit} className="space-y-4 animate-fade-in">
                        {/* Login form */}
                        <div className="cyber-panel hud-corners p-6 space-y-5">
                            <div className="text-center mb-2">
                                <div className="text-[9px] text-matrix/40 tracking-[0.3em] uppercase">Identification Required</div>
                            </div>

                            {/* Error message */}
                            {error && (
                                <div className="p-3 border border-cyber-pink/30 bg-cyber-pink/5 text-cyber-pink text-[11px] text-center animate-fade-in">
                                    ⚠ {error}
                                </div>
                            )}

                            {/* Username */}
                            <div>
                                <label className="text-[8px] text-white/20 tracking-[0.2em] uppercase mb-1 block">
                                    Identifiant
                                </label>
                                <div className="flex items-center border border-matrix/15 focus-within:border-matrix focus-within:shadow-[0_0_10px_rgba(0,255,65,0.1)] transition-all bg-black/40">
                                    <span className="text-matrix/40 text-[11px] pl-3 shrink-0">{'>'}</span>
                                    <input
                                        ref={inputRef}
                                        type="text"
                                        value={username}
                                        onChange={e => setUsername(e.target.value)}
                                        placeholder="username"
                                        autoComplete="username"
                                        className="flex-1 bg-transparent py-3 px-2 text-sm text-white/80 outline-none font-mono placeholder:text-matrix/20"
                                    />
                                </div>
                            </div>

                            {/* Password */}
                            <div>
                                <label className="text-[8px] text-white/20 tracking-[0.2em] uppercase mb-1 block">
                                    Mot de passe
                                </label>
                                <div className="flex items-center border border-matrix/15 focus-within:border-matrix focus-within:shadow-[0_0_10px_rgba(0,255,65,0.1)] transition-all bg-black/40">
                                    <span className="text-matrix/40 text-[11px] pl-3 shrink-0">🔒</span>
                                    <input
                                        type="password"
                                        value={password}
                                        onChange={e => setPassword(e.target.value)}
                                        placeholder="••••••••"
                                        autoComplete="current-password"
                                        onKeyDown={e => e.key === 'Enter' && handleSubmit(e)}
                                        className="flex-1 bg-transparent py-3 px-2 text-sm text-white/80 outline-none font-mono placeholder:text-white/10"
                                    />
                                </div>
                            </div>

                            {/* Submit button */}
                            <button
                                type="submit"
                                disabled={isLoading || !username.trim() || !password.trim()}
                                className={`w-full cyber-btn py-3 text-center tracking-[0.2em] ${isLoading ? 'animate-pulse opacity-60' : ''}`}
                            >
                                {isLoading ? (
                                    <span className="flex items-center justify-center gap-2">
                                        <span className="w-2 h-2 bg-matrix/60 rounded-full animate-pulse" />
                                        AUTHENTICATION...
                                    </span>
                                ) : (
                                    'ACCESS SYSTEM'
                                )}
                            </button>
                        </div>

                        {/* Footer info */}
                        <div className="text-center text-[8px] text-white/10 tracking-[0.2em] space-y-1">
                            <div>PROTOCOL: JWT HS256 • ENCRYPTION: AES-256</div>
                            <div>THE HIVE © 2026 — ALL RIGHTS RESERVED</div>
                        </div>
                    </form>
                )}
            </div>
        </div>
    )
}
