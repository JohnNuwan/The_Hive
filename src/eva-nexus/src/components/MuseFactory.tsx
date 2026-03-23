import { useCallback, useEffect, useMemo, useState } from 'react'
import { Activity, Camera, Film, Gauge, Image, Lock, RefreshCw, Sparkles } from 'lucide-react'

import {
    checkNodeHealth,
    getMuseNiches,
    getMuseNicheScores,
    getMuseStats,
    type MuseNiche,
    type MuseStats,
    type NodeHealth,
} from '../services/api'

function formatPercent(score?: number): string {
    if (typeof score !== 'number' || Number.isNaN(score)) {
        return 'indisponible'
    }
    return `${Math.round(score * 100)}%`
}

function HealthBadge({ status }: { status: string }) {
    const tone = status === 'online'
        ? 'text-matrix border-matrix/20 bg-matrix/10'
        : status === 'degraded'
            ? 'text-cyber-amber border-cyber-amber/20 bg-cyber-amber/10'
            : 'text-cyber-pink border-cyber-pink/20 bg-cyber-pink/10'

    return <span className={`px-3 py-1 text-[9px] font-black uppercase tracking-[0.18em] border ${tone}`}>{status}</span>
}

function MetricCard({ label, value, meta, tone = 'matrix' }: { label: string; value: string; meta?: string; tone?: 'matrix' | 'cyan' | 'amber' }) {
    const toneClass = {
        matrix: 'text-matrix',
        cyan: 'text-cyber-cyan',
        amber: 'text-cyber-amber',
    }[tone]

    return (
        <div className="cyber-panel hud-corners p-4">
            <div className="text-[8px] uppercase tracking-[0.2em] text-white/20 mb-1">{label}</div>
            <div className={`text-xl font-bold ${toneClass}`}>{value}</div>
            {meta ? <div className="text-[9px] text-white/25 mt-2">{meta}</div> : null}
        </div>
    )
}

export default function MuseFactory() {
    const [health, setHealth] = useState<NodeHealth | null>(null)
    const [stats, setStats] = useState<MuseStats | null>(null)
    const [niches, setNiches] = useState<MuseNiche[]>([])
    const [scores, setScores] = useState<Record<string, number>>({})
    const [isLoading, setIsLoading] = useState(true)
    const [isRefreshing, setIsRefreshing] = useState(false)

    const loadMuse = useCallback(async (refreshScores = false) => {
        if (refreshScores) {
            setIsRefreshing(true)
        } else {
            setIsLoading(true)
        }

        const [healthPayload, statsPayload, nichesPayload, scoresPayload] = await Promise.all([
            checkNodeHealth('Muse', '/api/muse/health'),
            getMuseStats(),
            getMuseNiches(),
            getMuseNicheScores(),
        ])

        setHealth(healthPayload)
        setStats(statsPayload)
        setNiches(nichesPayload)
        setScores(scoresPayload)
        setIsLoading(false)
        setIsRefreshing(false)
    }, [])

    useEffect(() => {
        void loadMuse()
        const interval = setInterval(() => {
            void loadMuse(true)
        }, 20000)
        return () => clearInterval(interval)
    }, [loadMuse])

    const rankedNiches = useMemo(
        () => [...niches].sort((left, right) => (scores[right.id] || 0) - (scores[left.id] || 0)),
        [niches, scores],
    )

    return (
        <div className="h-full overflow-y-auto p-4 space-y-4 animate-fade-in">
            <div className="cyber-panel hud-corners p-5 lg:p-6">
                <div className="flex flex-col gap-4 xl:flex-row xl:items-start xl:justify-between">
                    <div>
                        <div className="flex items-center gap-3 text-cyber-cyan/70">
                            <Camera size={18} />
                            <span className="text-[10px] uppercase tracking-[0.2em] font-bold">Muse Factory</span>
                        </div>
                        <h2 className="mt-3 font-display text-2xl font-black tracking-[0.08em] text-white/85">
                            Cockpit media en lecture seule
                        </h2>
                        <p className="mt-3 max-w-3xl text-[11px] leading-relaxed text-white/30">
                            Cette vue expose l etat du service Muse, les niches actives, les scores de marche et le modele courant. Les actions de generation restent volontairement desactivees pendant le run trading.
                        </p>
                    </div>
                    <div className="flex items-center gap-3">
                        <HealthBadge status={health?.status || 'offline'} />
                        <button
                            onClick={() => void loadMuse(true)}
                            disabled={isRefreshing}
                            className="cyber-btn text-[9px] px-3 py-2 disabled:opacity-40"
                        >
                            <RefreshCw size={12} className={isRefreshing ? 'animate-spin' : ''} />
                            <span>Rafraichir</span>
                        </button>
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-2 xl:grid-cols-5 gap-3">
                <MetricCard label="Service" value={health?.status || 'offline'} tone={health?.status === 'online' ? 'matrix' : 'amber'} meta={`latence ${health?.latency ?? -1} ms`} />
                <MetricCard label="Generations" value={String(stats?.total_generations || 0)} tone="matrix" />
                <MetricCard label="Templates" value={String(stats?.available_templates || 0)} tone="cyan" />
                <MetricCard label="Mode" value={stats?.mode || 'indisponible'} tone="amber" />
                <MetricCard label="Modele" value={stats?.model || 'indisponible'} tone="matrix" />
            </div>

            <div className="grid grid-cols-1 xl:grid-cols-[0.95fr_1.05fr] gap-4">
                <section className="cyber-panel hud-corners p-4 lg:p-5 space-y-4">
                    <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.2em] text-matrix/50">
                        <Gauge size={14} />
                        <span>Statut de fonctionnement</span>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                        <div className="border border-white/[0.05] bg-white/[0.02] p-4">
                            <div className="text-[8px] uppercase tracking-[0.18em] text-white/20">Mode d execution</div>
                            <div className="mt-2 text-[12px] font-bold text-cyber-cyan">Observation / dry-run</div>
                            <div className="mt-2 text-[9px] text-white/25">Les commandes de generation sont masquees pendant le run.</div>
                        </div>
                        <div className="border border-white/[0.05] bg-white/[0.02] p-4">
                            <div className="text-[8px] uppercase tracking-[0.18em] text-white/20">Etat du modele</div>
                            <div className="mt-2 text-[12px] font-bold text-matrix">{stats?.model || 'indisponible'}</div>
                            <div className="mt-2 text-[9px] text-white/25">Mode {stats?.mode || 'indisponible'}</div>
                        </div>
                    </div>
                    <div className="border border-white/[0.05] bg-black/30 p-4">
                        <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.2em] text-cyber-amber/60 mb-2">
                            <Lock size={14} />
                            <span>Garde-fou run actif</span>
                        </div>
                        <div className="text-[10px] text-white/35 leading-relaxed">
                            Les ecrans media restent consultatifs tant que le trainer occupe les ressources critiques. Les generations image, video et viralisation ne sont pas exposees ici pendant cette fenetre.
                        </div>
                    </div>
                </section>

                <section className="cyber-panel hud-corners p-4 lg:p-5 space-y-4">
                    <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.2em] text-cyber-cyan/60">
                        <Sparkles size={14} />
                        <span>Niches et scores de marche</span>
                    </div>
                    {isLoading ? (
                        <div className="border border-dashed border-white/10 p-4 text-[10px] text-white/25">Chargement de l inventaire Muse...</div>
                    ) : rankedNiches.length === 0 ? (
                        <div className="border border-dashed border-white/10 p-4 text-[10px] text-white/25">Inventaire des niches indisponible.</div>
                    ) : (
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                            {rankedNiches.map((niche) => (
                                <div key={niche.id} className="border border-white/[0.05] bg-white/[0.02] p-4 space-y-3">
                                    <div className="flex items-start justify-between gap-3">
                                        <div>
                                            <div className="text-[11px] font-bold text-white/75">{niche.label}</div>
                                            <div className="text-[8px] text-white/20 uppercase tracking-[0.15em]">{niche.id}</div>
                                        </div>
                                        <span className={`px-2 py-0.5 text-[8px] uppercase tracking-[0.18em] border ${niche.is_nsfw ? 'border-cyber-pink/20 bg-cyber-pink/10 text-cyber-pink/70' : 'border-matrix/20 bg-matrix/10 text-matrix/70'}`}>
                                            {niche.is_nsfw ? 'nsfw' : 'safe'}
                                        </span>
                                    </div>
                                    <p className="text-[10px] text-white/35 leading-relaxed">{niche.description}</p>
                                    <div className="grid grid-cols-2 gap-2 text-[9px] text-white/30">
                                        <div className="border border-white/[0.05] bg-black/30 p-2">
                                            <div className="uppercase tracking-[0.15em] text-white/20">Score</div>
                                            <div className="mt-1 text-cyber-cyan font-bold">{formatPercent(scores[niche.id])}</div>
                                        </div>
                                        <div className="border border-white/[0.05] bg-black/30 p-2">
                                            <div className="uppercase tracking-[0.15em] text-white/20">Cadence</div>
                                            <div className="mt-1 text-matrix font-bold">{niche.post_interval_hours}h</div>
                                        </div>
                                    </div>
                                    <div className="flex items-center gap-2 flex-wrap text-[8px] uppercase tracking-[0.15em]">
                                        <span className={`px-2 py-0.5 border ${niche.enabled ? 'border-matrix/20 bg-matrix/10 text-matrix/70' : 'border-white/10 bg-white/[0.03] text-white/30'}`}>
                                            {niche.enabled ? 'active' : 'inactive'}
                                        </span>
                                        <span className="px-2 py-0.5 border border-white/10 bg-white/[0.03] text-white/30">
                                            loras {niche.recommended_loras.length}
                                        </span>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </section>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="cyber-panel hud-corners p-4">
                    <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.2em] text-white/40 mb-3">
                        <Image size={14} />
                        <span>Atelier image</span>
                    </div>
                    <div className="text-[10px] text-white/35 leading-relaxed">
                        Mode consultation. Les prompts et generations image restent hors execution pendant le run trading.
                    </div>
                </div>
                <div className="cyber-panel hud-corners p-4">
                    <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.2em] text-white/40 mb-3">
                        <Film size={14} />
                        <span>Atelier video</span>
                    </div>
                    <div className="text-[10px] text-white/35 leading-relaxed">
                        Les pipelines AnimateDiff / media sont visibles via le service, mais aucune generation n est declenchee depuis cet ecran.
                    </div>
                </div>
                <div className="cyber-panel hud-corners p-4">
                    <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.2em] text-white/40 mb-3">
                        <Activity size={14} />
                        <span>Publication</span>
                    </div>
                    <div className="text-[10px] text-white/35 leading-relaxed">
                        Historique de publication indisponible sur cette API. L ecran reste donc strictement observatoire pour eviter toute fausse metrique.
                    </div>
                </div>
            </div>
        </div>
    )
}
