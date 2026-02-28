import { useState, useEffect } from 'react';
import {
    Zap, Image, Video, Film, TrendingUp,
    Power, RefreshCw, Lock, Eye, ChevronDown, ChevronUp,
    Clock, Target, Download, PlayCircle
} from 'lucide-react';

// ─── Types ──────────────────────────────────────────────────────────────────

interface Niche {
    id: string;
    label: string;
    description: string;
    base_prompt?: string;
    enabled: boolean;
    is_nsfw: boolean;
    post_interval_hours: number;
    recommended_loras: { filename: string; strength: number }[];
    trend_score?: number;
}

type ContentType = 'image' | 'video';

// ─── Main Component ───────────────────────────────────────────────────────────

export default function MuseFactory() {
    const MUSE_API = 'http://192.168.1.5:9100';

    const [niches, setNiches] = useState<Niche[]>([]);
    const [trendScores, setTrendScores] = useState<Record<string, number>>({});
    const [selectedNiche, setSelectedNiche] = useState<Niche | null>(null);
    const [contentType, setContentType] = useState<ContentType>('image');
    const [prompt, setPrompt] = useState('');
    const [privateMode, setPrivateMode] = useState(false);
    const [faceSwap, setFaceSwap] = useState(false);
    const [loading, setLoading] = useState(false);
    const [loadingScores, setLoadingScores] = useState(false);
    const [outputUrl, setOutputUrl] = useState<string | null>(null);
    const [outputType, setOutputType] = useState<'image' | 'video'>('image');
    const [expanded, setExpanded] = useState<string | null>(null);
    const [autoMode, setAutoMode] = useState(false);

    // Load niches from API
    useEffect(() => {
        fetch(`${MUSE_API}/niches`)
            .then(r => r.json())
            .then(data => {
                const loaded: Niche[] = data.niches || [];
                setNiches(loaded);
                if (loaded.length > 0) setSelectedNiche(loaded[0]);
            })
            .catch(() => {
                // Fallback hardcoded niches while API loads
                const fallback: Niche[] = [
                    { id: 'girlfriend', label: '💕 Girlfriend Experience', description: 'Sweet, intimate, candid', enabled: true, is_nsfw: false, post_interval_hours: 6, recommended_loras: [] },
                    { id: 'fitness', label: '🏋️ Fitness & Athletic', description: 'Sport, athletic, energetic', enabled: true, is_nsfw: false, post_interval_hours: 8, recommended_loras: [] },
                    { id: 'dominatrice', label: '⛓️ Dominatrice', description: 'BDSM dominant, latex, leather', enabled: true, is_nsfw: true, post_interval_hours: 12, recommended_loras: [] },
                    { id: 'soumise', label: '🎀 Douce & Soumise', description: 'Shy, submissive, pastel', enabled: true, is_nsfw: true, post_interval_hours: 10, recommended_loras: [] },
                    { id: 'pied', label: '🦶 Foot Fetish', description: 'Elegant feet, pedicure, close-up', enabled: true, is_nsfw: false, post_interval_hours: 12, recommended_loras: [] },
                    { id: 'rousse', label: '🦊 Rousse', description: 'Red hair, freckles, natural', enabled: true, is_nsfw: false, post_interval_hours: 8, recommended_loras: [] },
                    { id: 'petite', label: '🌸 Petite & Cute', description: 'Small frame, playful, kawaii', enabled: true, is_nsfw: false, post_interval_hours: 8, recommended_loras: [] },
                    { id: 'milf', label: '👑 MILF & Mature', description: 'Mature, confident, elegant', enabled: true, is_nsfw: false, post_interval_hours: 10, recommended_loras: [] },
                    { id: 'cosplay', label: '🎮 Cosplay & Anime', description: 'Gaming, anime, costume', enabled: true, is_nsfw: false, post_interval_hours: 12, recommended_loras: [] },
                    { id: 'furry', label: '🦊 Furry Anthro', description: 'Anthropomorphic art', enabled: true, is_nsfw: false, post_interval_hours: 12, recommended_loras: [] },
                ];
                setNiches(fallback);
                setSelectedNiche(fallback[0]);
            });
    }, []);

    const loadTrendScores = async () => {
        setLoadingScores(true);
        try {
            const res = await fetch(`${MUSE_API}/niches/scores`);
            const data = await res.json();
            setTrendScores(data.scores || {});
        } catch (e) {
            console.error('Could not load trend scores', e);
        } finally {
            setLoadingScores(false);
        }
    };

    const nichesWithScores = niches
        .map(n => ({ ...n, trend_score: trendScores[n.id] }))
        .sort((a, b) => (b.trend_score ?? 0) - (a.trend_score ?? 0));

    const handleGenerate = async () => {
        if (!selectedNiche || !prompt.trim()) return;
        setLoading(true);
        setOutputUrl(null);

        const finalPrompt = privateMode
            ? `${selectedNiche.base_prompt ?? prompt}, nsfw, boudoir, seductive, extremely intimate, private selfie`
            : prompt || selectedNiche.description;

        try {
            if (contentType === 'image') {
                const res = await fetch(`${MUSE_API}/generate/influencer`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        influencer_id: 'inf-001',
                        prompt: finalPrompt,
                        onlyfans_mode: privateMode,
                        use_faceswap: faceSwap,
                        niche_id: selectedNiche.id,
                    }),
                });
                if (!res.ok) throw new Error(await res.text());
                const blob = await res.blob();
                setOutputUrl(URL.createObjectURL(blob));
                setOutputType('image');
            } else {
                const res = await fetch(`${MUSE_API}/generate/video`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        prompt: finalPrompt,
                        niche_id: selectedNiche.id,
                        use_faceswap: faceSwap,
                        influencer_id: 'inf-001',
                    }),
                });
                if (!res.ok) throw new Error(await res.text());
                const blob = await res.blob();
                setOutputUrl(URL.createObjectURL(blob));
                setOutputType('video');
            }
        } catch (e: any) {
            console.error(e);
            alert(`Error: ${e.message}`);
        } finally {
            setLoading(false);
        }
    };

    const TrendBar = ({ score }: { score?: number }) => {
        const pct = score !== undefined ? Math.round(score * 100) : 0;
        const color = pct > 80 ? '#00ff41' : pct > 60 ? '#ffd700' : '#ff6b35';
        return (
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginTop: '4px' }}>
                <div style={{ flex: 1, height: '3px', background: '#1a1a2e', borderRadius: '2px' }}>
                    <div style={{ width: `${pct}%`, height: '100%', background: color, borderRadius: '2px', transition: 'width 0.6s ease' }} />
                </div>
                {score !== undefined && <span style={{ fontSize: '10px', color, fontFamily: 'monospace', minWidth: '28px' }}>{pct}%</span>}
            </div>
        );
    };

    return (
        <div style={{ display: 'flex', height: '100%', gap: '16px', padding: '16px', background: '#060611', color: '#e0e0e0', fontFamily: "'Inter', sans-serif" }}>

            {/* LEFT — Niche Roster */}
            <div style={{ width: '340px', display: 'flex', flexDirection: 'column', gap: '8px', overflowY: 'auto' }}>

                {/* Header */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <Target size={14} color="#00ff41" />
                        <span style={{ fontSize: '12px', letterSpacing: '2px', color: '#00ff41', textTransform: 'uppercase' }}>Niche Roster</span>
                    </div>
                    <div style={{ display: 'flex', gap: '6px' }}>
                        <button
                            onClick={loadTrendScores}
                            disabled={loadingScores}
                            style={{ background: 'none', border: '1px solid #1a2a3a', borderRadius: '4px', padding: '4px 8px', color: '#888', cursor: 'pointer', fontSize: '11px', display: 'flex', alignItems: 'center', gap: '4px' }}
                        >
                            <RefreshCw size={10} style={{ animation: loadingScores ? 'spin 1s linear infinite' : 'none' }} />
                            {loadingScores ? 'Analyse...' : 'Score marché'}
                        </button>
                    </div>
                </div>

                {/* Niche Cards */}
                {nichesWithScores.map(niche => (
                    <div
                        key={niche.id}
                        onClick={() => { setSelectedNiche(niche); setPrompt(niche.description); }}
                        style={{
                            background: selectedNiche?.id === niche.id ? '#0d1f35' : '#0a0a1a',
                            border: `1px solid ${selectedNiche?.id === niche.id ? '#00ff41' : '#1a2a3a'}`,
                            borderRadius: '6px',
                            padding: '10px 12px',
                            cursor: 'pointer',
                            transition: 'all 0.15s ease',
                        }}
                    >
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <span style={{ fontSize: '13px', fontWeight: 600 }}>{niche.label}</span>
                            <div style={{ display: 'flex', gap: '4px', alignItems: 'center' }}>
                                {niche.is_nsfw && (
                                    <span style={{ fontSize: '9px', background: '#2a0a2a', color: '#ff69b4', border: '1px solid #4a0a4a', borderRadius: '3px', padding: '1px 4px' }}>NSFW</span>
                                )}
                                <Clock size={10} color="#555" />
                                <span style={{ fontSize: '10px', color: '#555' }}>{niche.post_interval_hours}h</span>
                            </div>
                        </div>
                        <p style={{ fontSize: '11px', color: '#666', margin: '3px 0 0', lineHeight: 1.4 }}>{niche.description}</p>
                        <TrendBar score={niche.trend_score} />
                    </div>
                ))}
            </div>

            {/* CENTER — Generator */}
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '12px', overflowY: 'auto' }}>

                <div style={{ fontSize: '12px', letterSpacing: '2px', color: '#00ff41', textTransform: 'uppercase', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Zap size={14} />
                    Studio — {selectedNiche?.label || 'Sélectionne une niche'}
                </div>

                {/* Content Type Toggle */}
                <div style={{ display: 'flex', background: '#0a0a1a', border: '1px solid #1a2a3a', borderRadius: '6px', padding: '3px' }}>
                    {(['image', 'video'] as ContentType[]).map(type => (
                        <button
                            key={type}
                            onClick={() => setContentType(type)}
                            style={{
                                flex: 1, border: 'none', borderRadius: '4px',
                                background: contentType === type ? '#00ff41' : 'none',
                                color: contentType === type ? '#000' : '#555',
                                padding: '7px', cursor: 'pointer', fontSize: '12px',
                                display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px',
                                fontWeight: contentType === type ? 700 : 400
                            }}
                        >
                            {type === 'image' ? <Image size={13} /> : <Film size={13} />}
                            {type === 'image' ? 'Image' : 'Vidéo (AnimateDiff)'}
                        </button>
                    ))}
                </div>

                {/* Prompt */}
                <div>
                    <label style={{ fontSize: '11px', color: '#444', letterSpacing: '1px', display: 'block', marginBottom: '6px' }}>PROMPT</label>
                    <textarea
                        value={prompt}
                        onChange={e => setPrompt(e.target.value)}
                        rows={3}
                        placeholder={selectedNiche?.description || 'Décris le contenu à générer...'}
                        style={{
                            width: '100%', background: '#0a0a1a', border: '1px solid #1a2a3a', borderRadius: '6px',
                            color: '#c0c0c0', padding: '10px 12px', fontSize: '13px', resize: 'vertical',
                            outline: 'none', boxSizing: 'border-box', lineHeight: 1.5
                        }}
                    />
                </div>

                {/* Toggles */}
                <div style={{ display: 'flex', gap: '10px' }}>

                    {/* Private Mode */}
                    <div
                        onClick={() => setPrivateMode(v => !v)}
                        style={{
                            flex: 1, padding: '10px 14px', borderRadius: '6px', cursor: 'pointer',
                            border: `1px solid ${privateMode ? '#ff1493' : '#1a2a3a'}`,
                            background: privateMode ? '#1a0015' : '#0a0a1a',
                            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                        }}
                    >
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <Lock size={13} color={privateMode ? '#ff1493' : '#444'} />
                            <div>
                                <div style={{ fontSize: '12px', color: privateMode ? '#ff1493' : '#888' }}>Contenu Privé</div>
                                <div style={{ fontSize: '10px', color: '#444' }}>Mode OnlyFans / NSFW</div>
                            </div>
                        </div>
                        <div style={{ width: '34px', height: '18px', background: privateMode ? '#ff1493' : '#1a2a3a', borderRadius: '9px', position: 'relative', transition: 'all 0.2s' }}>
                            <div style={{ position: 'absolute', top: '2px', left: privateMode ? '16px' : '2px', width: '14px', height: '14px', background: '#fff', borderRadius: '50%', transition: 'all 0.2s' }} />
                        </div>
                    </div>

                    {/* Face Swap */}
                    <div
                        onClick={() => setFaceSwap(v => !v)}
                        style={{
                            flex: 1, padding: '10px 14px', borderRadius: '6px', cursor: 'pointer',
                            border: `1px solid ${faceSwap ? '#00bfff' : '#1a2a3a'}`,
                            background: faceSwap ? '#001a2a' : '#0a0a1a',
                            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                        }}
                    >
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <Eye size={13} color={faceSwap ? '#00bfff' : '#444'} />
                            <div>
                                <div style={{ fontSize: '12px', color: faceSwap ? '#00bfff' : '#888' }}>ReActor FaceSwap</div>
                                <div style={{ fontSize: '10px', color: '#444' }}>Injecte l'identité</div>
                            </div>
                        </div>
                        <div style={{ width: '34px', height: '18px', background: faceSwap ? '#00bfff' : '#1a2a3a', borderRadius: '9px', position: 'relative', transition: 'all 0.2s' }}>
                            <div style={{ position: 'absolute', top: '2px', left: faceSwap ? '16px' : '2px', width: '14px', height: '14px', background: '#fff', borderRadius: '50%', transition: 'all 0.2s' }} />
                        </div>
                    </div>
                </div>

                {/* LoRAs Info */}
                {selectedNiche && selectedNiche.recommended_loras.length > 0 && (
                    <div style={{ background: '#0a0a1a', border: '1px solid #1a2a3a', borderRadius: '6px', padding: '10px 12px' }}>
                        <div style={{ fontSize: '11px', color: '#444', letterSpacing: '1px', marginBottom: '6px' }}>LORAS ACTIFS</div>
                        {selectedNiche.recommended_loras.map((l, i) => (
                            <div key={i} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: '#666', padding: '2px 0' }}>
                                <span>{l.filename}</span>
                                <span style={{ color: '#00ff41' }}>×{l.strength}</span>
                            </div>
                        ))}
                    </div>
                )}

                {/* Generate Button */}
                <button
                    onClick={handleGenerate}
                    disabled={loading || !selectedNiche}
                    style={{
                        padding: '14px', border: 'none', borderRadius: '6px',
                        background: loading ? '#1a2a3a' : '#00ff41',
                        color: loading ? '#444' : '#000', fontWeight: 700,
                        fontSize: '13px', cursor: loading ? 'wait' : 'pointer',
                        display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px',
                        letterSpacing: '1px', textTransform: 'uppercase',
                        transition: 'all 0.2s'
                    }}
                >
                    {loading ? (
                        <><RefreshCw size={14} style={{ animation: 'spin 1s linear infinite' }} /> Génération en cours...</>
                    ) : contentType === 'video' ? (
                        <><PlayCircle size={14} /> Générer Clip Vidéo</>
                    ) : (
                        <><Zap size={14} /> Générer Image</>
                    )}
                </button>
            </div>

            {/* RIGHT — Output */}
            <div style={{ width: '380px', display: 'flex', flexDirection: 'column', gap: '12px' }}>

                <div style={{ fontSize: '12px', letterSpacing: '2px', color: '#00ff41', textTransform: 'uppercase', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <TrendingUp size={14} />
                    Studio Output
                </div>

                <div style={{
                    flex: 1, minHeight: '380px', background: '#0a0a1a', border: '1px solid #1a2a3a',
                    borderRadius: '6px', display: 'flex', alignItems: 'center', justifyContent: 'center',
                    position: 'relative', overflow: 'hidden'
                }}>
                    {outputUrl ? (
                        outputType === 'video' ? (
                            <video controls src={outputUrl} style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }} />
                        ) : (
                            <img src={outputUrl} alt="Output" style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }} />
                        )
                    ) : (
                        <div style={{ textAlign: 'center', color: '#2a2a3a' }}>
                            {loading ? (
                                <div>
                                    <div style={{ width: '40px', height: '40px', border: '2px solid #1a2a3a', borderTop: '2px solid #00ff41', borderRadius: '50%', animation: 'spin 1s linear infinite', margin: '0 auto 12px' }} />
                                    <div style={{ fontSize: '12px', color: '#444' }}>{contentType === 'video' ? 'AnimateDiff generating...' : 'ComfyUI generating...'}</div>
                                </div>
                            ) : (
                                <>
                                    {contentType === 'video' ? <Video size={40} style={{ marginBottom: '12px' }} /> : <Image size={40} style={{ marginBottom: '12px' }} />}
                                    <div style={{ fontSize: '12px' }}>En attente de génération</div>
                                </>
                            )}
                        </div>
                    )}
                </div>

                {outputUrl && (
                    <a
                        href={outputUrl}
                        download={outputType === 'video' ? 'hive_clip.mp4' : 'hive_img.png'}
                        style={{
                            display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px',
                            padding: '10px', border: '1px solid #1a2a3a', borderRadius: '6px',
                            color: '#00ff41', textDecoration: 'none', fontSize: '12px'
                        }}
                    >
                        <Download size={13} /> Télécharger
                    </a>
                )}

                {/* Niche Stats */}
                {selectedNiche && (
                    <div style={{ background: '#0a0a1a', border: '1px solid #1a2a3a', borderRadius: '6px', padding: '12px' }}>
                        <div style={{ fontSize: '11px', color: '#444', letterSpacing: '1px', marginBottom: '8px' }}>NICHE STATS</div>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '5px' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px' }}>
                                <span style={{ color: '#666' }}>Posting Schedule</span>
                                <span style={{ color: '#aaa' }}>Every {selectedNiche.post_interval_hours}h</span>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px' }}>
                                <span style={{ color: '#666' }}>Content Rating</span>
                                <span style={{ color: selectedNiche.is_nsfw ? '#ff69b4' : '#00ff41' }}>{selectedNiche.is_nsfw ? 'NSFW' : 'SFW'}</span>
                            </div>
                            {trendScores[selectedNiche.id] !== undefined && (
                                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px' }}>
                                    <span style={{ color: '#666' }}>Trend Score</span>
                                    <span style={{ color: '#ffd700' }}>{Math.round((trendScores[selectedNiche.id] || 0) * 100)}%</span>
                                </div>
                            )}
                            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px' }}>
                                <span style={{ color: '#666' }}>LoRAs chargés</span>
                                <span style={{ color: '#aaa' }}>{selectedNiche.recommended_loras.length}</span>
                            </div>
                        </div>
                    </div>
                )}
            </div>

            <style>{`
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
      `}</style>
        </div>
    );
}
