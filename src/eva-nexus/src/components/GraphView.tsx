import { useEffect, useState, useRef } from 'react'
import ForceGraph2D from 'react-force-graph-2d'
import { getMemoryGraph } from '../services/api'
import { Loader2, Zap, Share2, Target } from 'lucide-react'

export default function GraphView() {
    const [data, setData] = useState({ nodes: [], links: [] })
    const [isLoading, setIsLoading] = useState(true)
    const [selectedNode, setSelectedNode] = useState<any>(null)
    const fgRef = useRef<any>()

    useEffect(() => {
        const loadData = async () => {
            try {
                const graphData = await getMemoryGraph(60, 0.75)
                setData(graphData)
            } catch (error) {
                console.error("Failed to load graph data", error)
            } finally {
                setIsLoading(false)
            }
        }
        loadData()
        const interval = setInterval(loadData, 30000)
        return () => clearInterval(interval)
    }, [])

    if (isLoading) {
        return (
            <div className="flex flex-col items-center justify-center h-full space-y-4 animate-pulse">
                <div className="relative">
                    <Loader2 size={48} className="text-sky-400 animate-spin" />
                    <Zap size={24} className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-sky-500 animate-pulse" />
                </div>
                <p className="text-sky-400/60 font-black uppercase tracking-[0.3em] text-xs">Extraction du Graphe Neural...</p>
            </div>
        )
    }

    return (
        <div className="relative h-full w-full glass rounded-[2.5rem] overflow-hidden border border-white/5 shadow-2xl">
            {/* Header Overlay */}
            <div className="absolute top-8 left-8 z-10 flex flex-col gap-2">
                <div className="flex items-center gap-3 px-4 py-2 bg-slate-950/80 border border-sky-500/30 rounded-xl backdrop-blur-xl shadow-2xl">
                    <Share2 size={18} className="text-sky-400 animate-pulse" />
                    <span className="text-sm font-black text-white tracking-widest uppercase">GNN Knowledge Graph</span>
                    <div className="flex items-center gap-2 px-2 py-0.5 bg-sky-500/10 rounded-md">
                        <div className="w-1.5 h-1.5 rounded-full bg-sky-500 animate-ping" />
                        <span className="text-[10px] text-sky-400 font-bold">{data.nodes.length} Nodes actif</span>
                    </div>
                </div>
            </div>

            {/* Selection Info Overlay */}
            {selectedNode && (
                <div className="absolute bottom-8 left-8 z-10 w-80 bg-slate-950/90 border border-white/10 p-6 rounded-3xl backdrop-blur-2xl shadow-3xl animate-slide-up">
                    <div className="flex items-center gap-3 mb-4">
                        <div className={`p-2 rounded-lg ${selectedNode.role === 'user' ? 'bg-indigo-500/20 text-indigo-400' : 'bg-sky-500/20 text-sky-400'}`}>
                            <Target size={20} />
                        </div>
                        <div>
                            <h3 className="text-xs font-black text-white uppercase tracking-wider">Node Details</h3>
                            <p className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">Type: {selectedNode.role}</p>
                        </div>
                    </div>
                    <div className="p-4 bg-white/5 rounded-2xl border border-white/5 mb-4 group hover:border-sky-500/30 transition-colors">
                        <p className="text-[11px] text-slate-300 leading-relaxed italic">"{selectedNode.label}"</p>
                    </div>
                    <div className="flex items-center justify-between text-[9px] font-black text-slate-500 uppercase tracking-[0.2em]">
                        <span>ID: {selectedNode.id.slice(0, 8)}</span>
                        <span>{selectedNode.timestamp && new Date(selectedNode.timestamp).toLocaleTimeString()}</span>
                    </div>
                </div>
            )}

            {/* Controls Overlay */}
            <div className="absolute top-8 right-8 z-10 flex flex-col gap-3">
                <button
                    onClick={() => fgRef.current?.zoomToFit(400)}
                    className="p-3 bg-slate-900/80 border border-white/10 rounded-xl text-slate-400 hover:text-sky-400 hover:border-sky-500/30 transition-all hover:scale-105 backdrop-blur-xl shadow-xl hover:shadow-sky-500/10"
                >
                    <Target size={20} />
                </button>
            </div>

            <ForceGraph2D
                ref={fgRef}
                graphData={data}
                nodeLabel="label"
                nodeColor={(node: any) => node.role === 'user' ? '#6366f1' : '#0ea5e9'}
                nodeRelSize={6}
                nodeCanvasObject={(node: any, ctx, globalScale) => {
                    const label = node.label;
                    const fontSize = 12 / globalScale;
                    ctx.font = `${fontSize}px Inter`;
                    const textWidth = ctx.measureText(label).width;
                    const bckgDimensions: [number, number] = [textWidth, fontSize].map(n => n + fontSize * 0.2) as [number, number];

                    // Draw Node Circle
                    ctx.beginPath();
                    ctx.arc(node.x, node.y, 4, 0, 2 * Math.PI, false);
                    ctx.fillStyle = node.role === 'user' ? '#6366f1' : '#0ea5e9';
                    ctx.fill();

                    if (globalScale > 3) {
                        ctx.fillStyle = 'rgba(15, 23, 42, 0.8)';
                        ctx.fillRect(node.x - bckgDimensions[0] / 2, node.y - bckgDimensions[1] / 2 - 8, bckgDimensions[0], bckgDimensions[1]);
                        ctx.textAlign = 'center';
                        ctx.textBaseline = 'middle';
                        ctx.fillStyle = node.role === 'user' ? '#a5b4fc' : '#bae6fd';
                        ctx.fillText(label, node.x, node.y - 8);
                    }
                }}
                linkColor={() => 'rgba(255, 255, 255, 0.08)'}
                linkWidth={(link: any) => (link.value || 1) * 2}
                backgroundColor="transparent"
                onNodeClick={(node) => setSelectedNode(node)}
                cooldownTicks={100}
                d3VelocityDecay={0.3}
            />
        </div>
    )
}
