import { useState } from 'react'
import { Camera, Wand2, User, ImageIcon, Download, Sparkles, SlidersHorizontal, Settings, Music } from 'lucide-react'

// Dummy data for influencers
const INFLUENCERS = [
    {
        id: 'inf-001',
        name: 'Neo Spectra',
        style: 'Cyberpunk Fashion Model',
        bio: 'Neon aesthetic, tactical street wear, high-tech props.',
        avatar: 'https://images.unsplash.com/photo-1605806616949-1e87b487cb2a?w=150&h=150&fit=crop&q=80',
    },
    {
        id: 'inf-002',
        name: 'Lois',
        style: 'Finance & Crypto Analyst',
        bio: 'Professional corporate wear, trading screens background, sharp look.',
        avatar: 'https://images.unsplash.com/photo-1573496359142-b8d87734a5a2?w=150&h=150&fit=crop&q=80',
    },
    {
        id: 'inf-003',
        name: 'Athena',
        style: 'Gamer & Tech Streamer',
        bio: 'RGB lighting, gaming headsets, casual tech wear, energetic.',
        avatar: 'https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=150&h=150&fit=crop&q=80',
    }
]

export default function MuseFactory() {
    const [selectedInfluencer, setSelectedInfluencer] = useState(INFLUENCERS[0])
    const [prompt, setPrompt] = useState('Neon genesis evangelion style, highly detailed portrait, 8k resolution, cinematic lighting')
    const [faceSwapEnabled, setFaceSwapEnabled] = useState(true)
    const [mediaType, setMediaType] = useState<'image' | 'audio'>('image')
    const [duration, setDuration] = useState(15)

    const [isGenerating, setIsGenerating] = useState(false)
    const [generatedImage, setGeneratedImage] = useState<string | null>(null)
    const [generatedAudio, setGeneratedAudio] = useState<string | null>(null)

    const handleGenerate = async () => {
        setIsGenerating(true)
        if (mediaType === 'image') {
            setTimeout(() => {
                setGeneratedImage(`https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?w=800&h=800&fit=crop&q=80&rand=${Math.random()}`)
                setIsGenerating(false)
            }, 3000)
        } else {
            try {
                // Call our explicit Audio API
                const res = await fetch('/api/muse/generate/audio', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt, duration })
                })

                if (res.ok) {
                    const blob = await res.blob()
                    const url = URL.createObjectURL(blob)
                    setGeneratedAudio(url)
                } else {
                    console.error("Audio generation failed")
                }
            } catch (e) {
                console.error("Audio generation error", e)
            } finally {
                setIsGenerating(false)
            }
        }
    }

    return (
        <div className="h-full flex flex-col lg:flex-row gap-6">

            {/* Left Column: Config Panel */}
            <div className="w-full lg:w-1/3 flex flex-col gap-6">

                {/* Roster Selection */}
                <div className="glass p-4 rounded-xl border border-matrix/20 flex flex-col gap-4">
                    <div className="flex items-center gap-2 text-matrix font-display font-bold uppercase tracking-widest border-b border-matrix/10 pb-2">
                        <User size={16} />
                        <h2>Hive Virtual Roster</h2>
                    </div>

                    <div className="flex flex-col gap-3 max-h-[250px] overflow-y-auto custom-scrollbar pr-2">
                        {INFLUENCERS.map((inf) => (
                            <button
                                key={inf.id}
                                onClick={() => setSelectedInfluencer(inf)}
                                className={`flex items-start gap-4 p-3 rounded-lg border transition-all text-left
                                    ${selectedInfluencer.id === inf.id
                                        ? 'bg-matrix/10 border-matrix/40 shadow-[0_0_10px_rgba(0,255,65,0.1)]'
                                        : 'bg-black/40 border-matrix/10 hover:border-matrix/30 hover:bg-matrix/5'}
                                `}
                            >
                                <img src={inf.avatar} alt={inf.name} className="w-12 h-12 rounded-lg object-cover border border-matrix/30" />
                                <div>
                                    <h3 className="text-sm font-bold text-white neon-text-subtle">{inf.name}</h3>
                                    <p className="text-[10px] text-matrix font-mono">{inf.style}</p>
                                    <p className="text-[10px] text-white/50 mt-1 line-clamp-2">{inf.bio}</p>
                                </div>
                            </button>
                        ))}
                    </div>
                </div>

                {/* Generator Settings */}
                <div className="glass p-4 rounded-xl border border-matrix/20 flex flex-col gap-4 flex-grow">
                    <div className="flex items-center justify-between border-b border-matrix/10 pb-2">
                        <div className="flex items-center gap-2 text-matrix font-display font-bold uppercase tracking-widest">
                            <SlidersHorizontal size={16} />
                            <h2>Generator Settings</h2>
                        </div>
                    </div>

                    <div className="flex gap-2 mb-2">
                        <button
                            onClick={() => setMediaType('image')}
                            className={`flex-1 py-2 text-xs font-bold uppercase tracking-widest flex items-center justify-center gap-2 border rounded-lg transition-all ${mediaType === 'image' ? 'bg-matrix/20 border-matrix text-matrix' : 'bg-black/30 border-matrix/20 text-matrix/40 hover:border-matrix/40'}`}
                        >
                            <ImageIcon size={14} /> Image
                        </button>
                        <button
                            onClick={() => setMediaType('audio')}
                            className={`flex-1 py-2 text-xs font-bold uppercase tracking-widest flex items-center justify-center gap-2 border rounded-lg transition-all ${mediaType === 'audio' ? 'bg-matrix/20 border-matrix text-matrix' : 'bg-black/30 border-matrix/20 text-matrix/40 hover:border-matrix/40'}`}
                        >
                            <Music size={14} /> Audio
                        </button>
                    </div>

                    <div className="flex flex-col gap-2">
                        <label className="text-[10px] text-matrix font-mono uppercase tracking-widest">
                            {mediaType === 'image' ? 'Creative Prompt (Scene)' : 'Audio Prompt (Genre, Instruments, Mood)'}
                        </label>
                        <textarea
                            value={prompt}
                            onChange={(e) => setPrompt(e.target.value)}
                            className="bg-black/50 border border-matrix/20 rounded-lg p-3 text-sm text-white focus:outline-none focus:border-matrix/50 h-28 custom-scrollbar resize-none"
                            placeholder={mediaType === 'image' ? "Describe the scene..." : "e.g. 80s synthwave fast beat with heavy bass"}
                        />
                    </div>

                    {mediaType === 'audio' && (
                        <div className="flex flex-col gap-2">
                            <label className="text-[10px] text-matrix font-mono uppercase tracking-widest flex justify-between">
                                <span>Duration</span>
                                <span>{duration}s</span>
                            </label>
                            <input
                                type="range"
                                min="5"
                                max="30"
                                value={duration}
                                onChange={(e) => setDuration(parseInt(e.target.value))}
                                className="w-full accent-matrix"
                            />
                        </div>
                    )}

                    {mediaType === 'image' && (
                        <div className="flex items-center justify-between bg-black/40 border border-matrix/20 p-3 rounded-lg mt-auto">
                            <div className="flex items-center gap-2">
                                <User size={16} className={faceSwapEnabled ? "text-matrix" : "text-white/40"} />
                                <div>
                                    <p className="text-xs font-bold text-white">ReActor Face Swap</p>
                                    <p className="text-[9px] text-white/50">Inject {selectedInfluencer.name}'s identity</p>
                                </div>
                            </div>
                            <button
                                onClick={() => setFaceSwapEnabled(!faceSwapEnabled)}
                                className={`w-10 h-5 rounded-full relative transition-colors ${faceSwapEnabled ? 'bg-matrix' : 'bg-white/20'}`}
                            >
                                <div className={`w-3 h-3 bg-black rounded-full absolute top-1 transition-all ${faceSwapEnabled ? 'right-1' : 'left-1'}`} />
                            </button>
                        </div>
                    )}

                    <button
                        onClick={handleGenerate}
                        disabled={isGenerating}
                        className={`mt-2 flex items-center justify-center gap-2 w-full py-3 rounded-lg font-bold uppercase tracking-widest transition-all
                            ${isGenerating
                                ? 'bg-matrix/20 text-matrix/50 cursor-not-allowed'
                                : 'bg-matrix/10 text-matrix border border-matrix/30 hover:bg-matrix hover:text-black hover:shadow-[0_0_20px_rgba(0,255,65,0.4)]'}
                        `}
                    >
                        {isGenerating ? (
                            <>
                                <Settings size={18} className="animate-spin" />
                                Generating...
                            </>
                        ) : (
                            <>
                                <Wand2 size={18} />
                                Generate Media
                            </>
                        )}
                    </button>
                </div>
            </div>

            {/* Right Column: Preview Area */}
            <div className="w-full lg:w-2/3 glass rounded-xl border border-matrix/20 relative overflow-hidden flex flex-col">
                <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-matrix to-transparent opacity-50"></div>

                <div className="p-4 border-b border-matrix/10 flex items-center justify-between bg-black/20 z-10">
                    <div className="flex items-center gap-2 text-matrix font-display font-bold uppercase tracking-widest">
                        {mediaType === 'image' ? <Camera size={16} /> : <Music size={16} />}
                        <h2>{mediaType === 'image' ? "Studio Output" : "Audio Track Output"}</h2>
                    </div>
                    {(generatedImage || generatedAudio) && (
                        <button className="flex items-center gap-2 text-[10px] bg-matrix/10 text-matrix px-3 py-1.5 rounded border border-matrix/20 hover:bg-matrix/20 transition-all">
                            <Download size={12} />
                            Save
                        </button>
                    )}
                </div>

                <div className="flex-grow flex items-center justify-center p-6 bg-black/40 relative">
                    {/* Background grid */}
                    <div className="absolute inset-0 grid-bg opacity-10"></div>

                    {isGenerating ? (
                        <div className="flex flex-col items-center gap-6 animate-pulse z-10">
                            <div className="w-24 h-24 border-2 border-dashed border-matrix rounded-full flex items-center justify-center relative">
                                <Sparkles size={32} className="text-matrix animate-ping absolute" />
                                <Wand2 size={32} className="text-matrix" />
                            </div>
                            <div className="text-center">
                                <p className="text-matrix font-bold tracking-widest uppercase">Rendering Request</p>
                                <p className="text-xs text-matrix/50 font-mono mt-2">Connecting to Proxmox GPU Engine...</p>
                            </div>
                        </div>
                    ) : mediaType === 'image' && generatedImage ? (
                        <img
                            src={generatedImage}
                            alt="Generated Output"
                            className="max-h-full max-w-full rounded-lg border border-matrix/30 shadow-[0_0_30px_rgba(0,255,65,0.1)] z-10 object-contain"
                        />
                    ) : mediaType === 'audio' && generatedAudio ? (
                        <div className="flex flex-col items-center gap-8 w-full max-w-md z-10 p-8 glass rounded-xl border border-matrix/30">
                            <div className="w-16 h-16 rounded-full bg-matrix/10 border border-matrix/40 flex items-center justify-center shadow-[0_0_20px_rgba(0,255,65,0.2)]">
                                <Music size={32} className="text-matrix" />
                            </div>
                            <div className="text-center w-full">
                                <h3 className="text-white font-bold tracking-widest uppercase mb-1">Generated Track</h3>
                                <p className="text-xs text-white/40 mb-6 truncate px-4">{prompt}</p>
                                <audio controls src={generatedAudio} className="w-full custom-audio-player" autoPlay />
                            </div>
                        </div>
                    ) : (
                        <div className="flex flex-col items-center gap-4 text-white/20 z-10">
                            {mediaType === 'image' ? <ImageIcon size={64} /> : <Music size={64} />}
                            <p className="text-sm font-mono uppercase tracking-[0.2em]">Awaiting Input Sequence</p>
                        </div>
                    )}
                </div>
            </div>

        </div>
    )
}
