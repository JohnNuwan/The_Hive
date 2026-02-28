import { useState, useEffect, useRef, useCallback } from 'react';
import { MessageSquare, Zap, Circle, Filter, Download, ChevronRight } from 'lucide-react';

interface AgentMessage {
    id: string;
    agent: string;
    company: string;
    type: 'thought' | 'action' | 'message' | 'result' | 'error';
    content: string;
    timestamp: string;
    target?: string; // for agent-to-agent messages
}

const AGENT_COLORS: Record<string, string> = {
    'EVA Core': '#00ff41',
    'Banker': '#ffd700',
    'Sentinel': '#00bfff',
    'Compliance': '#ff8c00',
    'Accountant': '#da70d6',
    'Lab': '#7fff00',
    'Sage': '#ff69b4',
    'Researcher': '#87ceeb',
    'Wraith': '#dc143c',
    'Muse': '#ff1dce',
    'Shadow': '#9370db',
    'RWA': '#20b2aa',
};

const TYPE_ICONS: Record<string, string> = {
    thought: '💭',
    action: '⚡',
    message: '📨',
    result: '✅',
    error: '❌',
};

export default function AgentFeed() {
    const KERNEL_WS = `ws://${window.location.hostname}:8800/ws/feed`;
    const KERNEL_API = '/api/kernel';

    const [messages, setMessages] = useState<AgentMessage[]>([]);
    const [connected, setConnected] = useState(false);
    const [filter, setFilter] = useState<string>('ALL');
    const [typeFilter, setTypeFilter] = useState<string>('ALL');
    const [autoScroll, setAutoScroll] = useState(true);
    const [msgCount, setMsgCount] = useState(0);
    const feedRef = useRef<HTMLDivElement>(null);
    const wsRef = useRef<WebSocket | null>(null);

    const agents = ['ALL', ...Object.keys(AGENT_COLORS)];
    const types = ['ALL', 'thought', 'action', 'message', 'result', 'error'];

    const connect = useCallback(() => {
        try {
            // Try WebSocket first, fall back to SSE polling
            const ws = new WebSocket(KERNEL_WS);
            wsRef.current = ws;

            ws.onopen = () => setConnected(true);
            ws.onclose = () => {
                setConnected(false);
                // Reconnect after 3s
                setTimeout(connect, 3000);
            };
            ws.onerror = () => {
                ws.close();
                // Fallback: poll via HTTP
                startPolling();
            };
            ws.onmessage = (event) => {
                try {
                    const msg: AgentMessage = JSON.parse(event.data);
                    setMessages(prev => [...prev.slice(-500), { ...msg, id: crypto.randomUUID() }]);
                    setMsgCount(c => c + 1);
                } catch { }
            };
        } catch {
            startPolling();
        }
    }, []);

    const startPolling = useCallback(() => {
        // Fallback: poll `/api/kernel/feed/recent` every 2s
        const interval = setInterval(async () => {
            try {
                const res = await fetch(`${KERNEL_API}/feed/recent?limit=20`);
                if (res.ok) {
                    setConnected(true);
                    const data = await res.json();
                    if (data.messages) {
                        setMessages(prev => {
                            const existingIds = new Set(prev.map(m => m.id));
                            const newMsgs = data.messages.filter((m: any) => !existingIds.has(m.id));
                            if (newMsgs.length === 0) return prev;
                            setMsgCount(c => c + newMsgs.length);
                            return [...prev.slice(-500), ...newMsgs];
                        });
                    }
                }
            } catch {
                setConnected(false);
            }
        }, 2000);
        return () => clearInterval(interval);
    }, []);

    useEffect(() => {
        connect();
        return () => wsRef.current?.close();
    }, []);

    useEffect(() => {
        if (autoScroll && feedRef.current) {
            feedRef.current.scrollTop = feedRef.current.scrollHeight;
        }
    }, [messages, autoScroll]);

    const filtered = messages.filter(m => {
        const agentOk = filter === 'ALL' || m.agent === filter || m.company === filter;
        const typeOk = typeFilter === 'ALL' || m.type === typeFilter;
        return agentOk && typeOk;
    });

    const clearFeed = () => {
        setMessages([]);
        setMsgCount(0);
    };

    const exportFeed = () => {
        const blob = new Blob([JSON.stringify(filtered, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `hive_feed_${Date.now()}.json`;
        a.click();
    };

    const formatTime = (ts: string) => {
        try { return new Date(ts).toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit', second: '2-digit' }); }
        catch { return ts; }
    };

    // Demo messages when disconnected
    const demoMessages: AgentMessage[] = connected ? [] : [
        { id: '1', agent: 'EVA Core', company: 'Genesis', type: 'thought', content: 'Analysing market conditions for EUR/USD. Checking GNN correlation matrix...', timestamp: new Date().toISOString() },
        { id: '2', agent: 'Banker', company: 'Trading', type: 'action', content: 'Placing long position EUR/USD 0.01 lot at 1.0842 — SL 1.0820 TP 1.0890', timestamp: new Date().toISOString() },
        { id: '3', agent: 'Sentinel', company: 'Risk', type: 'message', content: 'Risk threshold check passed. Exposure within limits.', timestamp: new Date().toISOString(), target: 'Banker' },
        { id: '4', agent: 'Muse', company: 'Media', type: 'action', content: 'Generating niche content for [girlfriend] — AnimateDiff 16 frames 512x768', timestamp: new Date().toISOString() },
        { id: '5', agent: 'Sage', company: 'Analysis', type: 'result', content: 'PEA Radar updated: LVMH score +0.82 | Hermès +0.79 | ASM International +0.91', timestamp: new Date().toISOString() },
    ];

    const displayMessages = connected ? filtered : demoMessages;

    return (
        <div style={{ display: 'flex', flexDirection: 'column', height: '100%', background: '#060611', color: '#e0e0e0', fontFamily: "'Inter', sans-serif", overflow: 'hidden' }}>

            {/* Header Bar */}
            <div style={{ padding: '12px 16px', borderBottom: '1px solid #1a2a3a', display: 'flex', alignItems: 'center', gap: '12px', flexShrink: 0 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <MessageSquare size={14} color="#00ff41" />
                    <span style={{ fontSize: '12px', letterSpacing: '2px', color: '#00ff41', textTransform: 'uppercase' }}>Live Agent Feed</span>
                </div>

                {/* Connection status */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                    <Circle size={7} color={connected ? '#00ff41' : '#ff4444'} fill={connected ? '#00ff41' : '#ff4444'} />
                    <span style={{ fontSize: '10px', color: connected ? '#00ff41' : '#ff4444' }}>
                        {connected ? 'LIVE' : 'DEMO — Kernel offline'}
                    </span>
                </div>

                <div style={{ marginLeft: 'auto', display: 'flex', gap: '6px', alignItems: 'center' }}>
                    {/* Message count */}
                    <span style={{ fontSize: '11px', color: '#444' }}>{msgCount} msgs total</span>

                    {/* Auto scroll toggle */}
                    <button
                        onClick={() => setAutoScroll(v => !v)}
                        style={{ padding: '4px 8px', background: autoScroll ? '#001500' : '#0a0a1a', border: `1px solid ${autoScroll ? '#00ff41' : '#1a2a3a'}`, borderRadius: '4px', color: autoScroll ? '#00ff41' : '#555', fontSize: '10px', cursor: 'pointer' }}
                    >
                        AUTO-SCROLL
                    </button>

                    <button onClick={exportFeed} style={{ background: 'none', border: '1px solid #1a2a3a', borderRadius: '4px', padding: '4px 8px', color: '#555', cursor: 'pointer', fontSize: '10px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                        <Download size={10} /> Export
                    </button>
                    <button onClick={clearFeed} style={{ background: 'none', border: '1px solid #1a2a3a', borderRadius: '4px', padding: '4px 8px', color: '#555', cursor: 'pointer', fontSize: '10px' }}>
                        Clear
                    </button>
                </div>
            </div>

            {/* Filters */}
            <div style={{ padding: '8px 16px', borderBottom: '1px solid #1a2a3a', display: 'flex', gap: '6px', overflowX: 'auto', flexShrink: 0 }}>
                <Filter size={11} color="#444" style={{ flexShrink: 0, marginTop: '3px' }} />
                {agents.map(a => (
                    <button key={a} onClick={() => setFilter(a)} style={{
                        padding: '3px 8px', borderRadius: '3px', border: 'none', cursor: 'pointer', fontSize: '10px', whiteSpace: 'nowrap',
                        background: filter === a ? (AGENT_COLORS[a] || '#00ff41') : '#0d0d1a',
                        color: filter === a ? '#000' : '#555', fontWeight: filter === a ? 700 : 400
                    }}>{a}</button>
                ))}
                <div style={{ width: '1px', background: '#1a2a3a', margin: '0 4px' }} />
                {types.map(t => (
                    <button key={t} onClick={() => setTypeFilter(t)} style={{
                        padding: '3px 8px', borderRadius: '3px', border: 'none', cursor: 'pointer', fontSize: '10px',
                        background: typeFilter === t ? '#1a2a3a' : 'none',
                        color: typeFilter === t ? '#aaa' : '#444'
                    }}>{TYPE_ICONS[t] || ''} {t}</button>
                ))}
            </div>

            {/* Feed */}
            <div ref={feedRef} style={{ flex: 1, overflowY: 'auto', padding: '8px 16px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                {displayMessages.map(msg => (
                    <div key={msg.id} style={{
                        display: 'flex', gap: '10px', padding: '7px 10px', borderRadius: '5px',
                        background: msg.type === 'error' ? '#1a0a0a' : msg.type === 'result' ? '#001500' : 'transparent',
                        border: `1px solid ${msg.type === 'error' ? '#330000' : msg.type === 'result' ? '#003300' : '#0d0d1a'}`,
                        alignItems: 'flex-start',
                    }}>
                        {/* Timestamp */}
                        <span style={{ fontSize: '10px', color: '#333', fontFamily: 'monospace', flexShrink: 0, marginTop: '1px' }}>
                            {formatTime(msg.timestamp)}
                        </span>

                        {/* Type icon */}
                        <span style={{ fontSize: '11px', flexShrink: 0 }}>{TYPE_ICONS[msg.type]}</span>

                        {/* Agent badge */}
                        <span style={{
                            fontSize: '10px', fontWeight: 700, flexShrink: 0,
                            background: `${AGENT_COLORS[msg.agent] || '#444'}22`,
                            color: AGENT_COLORS[msg.agent] || '#888',
                            border: `1px solid ${AGENT_COLORS[msg.agent] || '#444'}44`,
                            borderRadius: '3px', padding: '0px 5px'
                        }}>{msg.agent}</span>

                        {/* Arrow for messages */}
                        {msg.target && (
                            <>
                                <ChevronRight size={11} color="#333" style={{ flexShrink: 0, marginTop: '1px' }} />
                                <span style={{ fontSize: '10px', color: AGENT_COLORS[msg.target] || '#888', flexShrink: 0 }}>{msg.target}</span>
                            </>
                        )}

                        {/* Content */}
                        <span style={{ fontSize: '12px', color: msg.type === 'error' ? '#ff6666' : '#c0c0c0', lineHeight: 1.4, wordBreak: 'break-word' }}>
                            {msg.content}
                        </span>
                    </div>
                ))}

                {displayMessages.length === 0 && (
                    <div style={{ textAlign: 'center', color: '#2a2a3a', padding: '60px', fontSize: '13px' }}>
                        <Zap size={32} style={{ marginBottom: '12px', opacity: 0.3 }} />
                        <div>En attente des transmissions des agents...</div>
                    </div>
                )}
            </div>
        </div>
    );
}
