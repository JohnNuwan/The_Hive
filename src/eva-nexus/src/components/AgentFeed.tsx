import { useState, useEffect, useRef, useCallback } from 'react';
import { MessageSquare, Zap, Circle, Filter, Download, ChevronRight } from 'lucide-react';

interface AgentMessage {
    id: string;
    agent: string;
    company: string;
    type: 'thought' | 'action' | 'message' | 'result' | 'error';
    content: string;
    timestamp: string;
    target?: string;
}

const AGENT_COLORS: Record<string, string> = {
    'EVA Core': '#00ff41',
    Banker: '#ffd700',
    Sentinel: '#00bfff',
    Compliance: '#ff8c00',
    Accountant: '#da70d6',
    Lab: '#7fff00',
    Sage: '#ff69b4',
    Researcher: '#87ceeb',
    Wraith: '#dc143c',
    Muse: '#ff1dce',
    Shadow: '#9370db',
    RWA: '#20b2aa',
};

const TYPE_LABELS: Record<AgentMessage['type'], string> = {
    thought: 'TH',
    action: 'ACT',
    message: 'MSG',
    result: 'OK',
    error: 'ERR',
};

function normalizeAgentName(raw: unknown): string {
    const value = typeof raw === 'string' ? raw.trim() : '';
    if (!value) {
        return 'System';
    }

    const normalized = value
        .replace(/[._/-]+/g, ' ')
        .replace(/\s+/g, ' ')
        .trim()
        .toLowerCase();

    if (normalized.includes('core')) return 'EVA Core';
    if (normalized.includes('banker')) return 'Banker';
    if (normalized.includes('sentinel')) return 'Sentinel';
    if (normalized.includes('compliance')) return 'Compliance';
    if (normalized.includes('accountant')) return 'Accountant';
    if (normalized.includes('lab')) return 'Lab';
    if (normalized.includes('sage')) return 'Sage';
    if (normalized.includes('researcher')) return 'Researcher';
    if (normalized.includes('wraith')) return 'Wraith';
    if (normalized.includes('muse')) return 'Muse';
    if (normalized.includes('shadow')) return 'Shadow';
    if (normalized.includes('rwa')) return 'RWA';
    if (normalized.includes('kernel')) return 'Kernel';

    return normalized
        .split(' ')
        .filter(Boolean)
        .map((segment) => segment.charAt(0).toUpperCase() + segment.slice(1))
        .join(' ');
}

function normalizeMessageType(raw: unknown): AgentMessage['type'] {
    const value = typeof raw === 'string' ? raw.trim().toLowerCase() : '';
    if (value === 'thought' || value === 'action' || value === 'message' || value === 'result' || value === 'error') {
        return value;
    }
    if (['alert', 'critical', 'fatal'].includes(value)) return 'error';
    if (['event', 'trade', 'order'].includes(value)) return 'action';
    if (['request', 'analysis', 'reasoning'].includes(value)) return 'thought';
    if (['response', 'success', 'done'].includes(value)) return 'result';
    return 'message';
}

function normalizeTimestamp(raw: unknown): string {
    const value = typeof raw === 'string' ? raw.trim() : '';
    if (!value) {
        return new Date().toISOString();
    }

    const parsed = new Date(value);
    if (!Number.isNaN(parsed.getTime())) {
        return parsed.toISOString();
    }

    const epoch = Number(value);
    if (!Number.isNaN(epoch)) {
        const parsedEpoch = new Date(epoch > 1_000_000_000_000 ? epoch : epoch * 1000);
        if (!Number.isNaN(parsedEpoch.getTime())) {
            return parsedEpoch.toISOString();
        }
    }

    return new Date().toISOString();
}

function normalizeMessageContent(raw: unknown, fallbackSource: unknown): string {
    const direct = typeof raw === 'string' ? raw.trim() : '';
    if (direct) {
        return direct;
    }
    if (typeof fallbackSource === 'string' && fallbackSource.trim()) {
        return fallbackSource.trim();
    }
    return 'Message sans contenu';
}

function normalizeFeedMessage(input: unknown, createClientId: () => string): AgentMessage {
    const payload = input && typeof input === 'object' ? (input as Record<string, unknown>) : {};
    const agent = normalizeAgentName(payload.agent ?? payload.source_agent ?? payload.source ?? payload.service);
    const type = normalizeMessageType(payload.type ?? payload.message_type ?? payload.kind);
    const timestamp = normalizeTimestamp(payload.timestamp ?? payload.created_at ?? payload.time ?? payload.date);
    const content = normalizeMessageContent(
        payload.content ?? payload.message ?? payload.action ?? payload.text ?? payload.summary,
        typeof input === 'string' ? input : '',
    );
    const company = typeof payload.company === 'string' && payload.company.trim() ? payload.company.trim() : 'Hive Swarm';
    const target = typeof payload.target === 'string' && payload.target.trim()
        ? normalizeAgentName(payload.target)
        : typeof payload.target_agent === 'string' && payload.target_agent.trim()
            ? normalizeAgentName(payload.target_agent)
            : undefined;
    const rawId = typeof payload.id === 'string' ? payload.id.trim() : '';
    const fallbackId = [agent, String(payload.timestamp ?? ''), type, content].join('|');

    return {
        id: rawId || fallbackId || createClientId(),
        agent,
        company,
        type,
        content,
        timestamp,
        target,
    };
}

export default function AgentFeed() {
    const wsProtocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const kernelWs = `${wsProtocol}://${window.location.host}/api/kernel/ws/feed`;
    const kernelApi = '/api/kernel';

    const [messages, setMessages] = useState<AgentMessage[]>([]);
    const [connected, setConnected] = useState(false);
    const [feedAvailable, setFeedAvailable] = useState<boolean | null>(null);
    const [filter, setFilter] = useState<string>('ALL');
    const [typeFilter, setTypeFilter] = useState<string>('ALL');
    const [autoScroll, setAutoScroll] = useState(true);
    const [msgCount, setMsgCount] = useState(0);
    const feedRef = useRef<HTMLDivElement>(null);
    const wsRef = useRef<WebSocket | null>(null);
    const pollingRef = useRef<number | null>(null);
    const reconnectRef = useRef<number | null>(null);
    const connectRef = useRef<(() => void) | null>(null);
    const feedUnavailableRef = useRef(false);

    const agents = ['ALL', ...Object.keys(AGENT_COLORS)];
    const types: Array<'ALL' | AgentMessage['type']> = ['ALL', 'thought', 'action', 'message', 'result', 'error'];

    const createClientId = () => {
        const cryptoApi = globalThis.crypto;
        if (cryptoApi && typeof cryptoApi.randomUUID === 'function') {
            return cryptoApi.randomUUID();
        }
        return `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
    };

    const stopPolling = useCallback(() => {
        if (pollingRef.current !== null) {
            window.clearInterval(pollingRef.current);
            pollingRef.current = null;
        }
    }, []);

    const probeFeedAvailability = useCallback(async () => {
        try {
            const response = await fetch(`${kernelApi}/feed/recent?limit=1`);
            if (response.status === 404) {
                feedUnavailableRef.current = true;
                setFeedAvailable(false);
                setConnected(false);
                stopPolling();
                return false;
            }
        } catch {
            // Un echec reseau ne prouve pas l'absence de la route.
        }
        setFeedAvailable(true);
        return true;
    }, [kernelApi, stopPolling]);

    const startPolling = useCallback(() => {
        if (pollingRef.current !== null) {
            return;
        }

        pollingRef.current = window.setInterval(async () => {
            try {
                const response = await fetch(`${kernelApi}/feed/recent?limit=20`);
                if (response.status === 404) {
                    feedUnavailableRef.current = true;
                    setFeedAvailable(false);
                    setConnected(false);
                    stopPolling();
                    return;
                }
                if (!response.ok) {
                    setConnected(false);
                    return;
                }

                setConnected(true);
                setFeedAvailable(true);
                const data = await response.json();
                if (data.messages) {
                    setMessages((previous) => {
                        const existingIds = new Set(previous.map((message) => message.id));
                        const nextMessages = data.messages
                            .map((message: unknown) => normalizeFeedMessage(message, createClientId))
                            .filter((message: AgentMessage) => !existingIds.has(message.id));
                        if (nextMessages.length === 0) {
                            return previous;
                        }
                        setMsgCount((count) => count + nextMessages.length);
                        return [...previous.slice(-500), ...nextMessages];
                    });
                }
            } catch {
                setConnected(false);
            }
        }, 2000);
    }, [kernelApi, stopPolling]);

    const scheduleReconnect = useCallback(() => {
        if (reconnectRef.current !== null || feedUnavailableRef.current) {
            return;
        }
        reconnectRef.current = window.setTimeout(() => {
            reconnectRef.current = null;
            connectRef.current?.();
        }, 3000);
    }, []);

    const connect = useCallback(() => {
        if (feedUnavailableRef.current) {
            return;
        }

        try {
            stopPolling();
            const socket = new WebSocket(kernelWs);
            wsRef.current = socket;

            socket.onopen = () => {
                setConnected(true);
                setFeedAvailable(true);
                stopPolling();
            };
            socket.onclose = () => {
                setConnected(false);
                startPolling();
                scheduleReconnect();
            };
            socket.onerror = () => {
                socket.close();
                startPolling();
            };
            socket.onmessage = (event) => {
                try {
                    const message = normalizeFeedMessage(JSON.parse(event.data), createClientId);
                    setMessages((previous) => {
                        if (previous.some((existing) => existing.id === message.id)) {
                            return previous;
                        }
                        setMsgCount((count) => count + 1);
                        return [...previous.slice(-500), message];
                    });
                } catch {
                    // Certaines trames peuvent etre corrompues.
                }
            };
        } catch {
            startPolling();
            scheduleReconnect();
        }
    }, [kernelWs, scheduleReconnect, startPolling, stopPolling]);

    useEffect(() => {
        connectRef.current = connect;
    }, [connect]);

    useEffect(() => {
        let cancelled = false;
        void (async () => {
            const feedAvailable = await probeFeedAvailability();
            if (!cancelled && feedAvailable) {
                connect();
            }
        })();

        return () => {
            cancelled = true;
            stopPolling();
            wsRef.current?.close();
            if (reconnectRef.current !== null) {
                window.clearTimeout(reconnectRef.current);
            }
        };
    }, [connect, probeFeedAvailability, stopPolling]);

    useEffect(() => {
        if (autoScroll && feedRef.current) {
            feedRef.current.scrollTop = feedRef.current.scrollHeight;
        }
    }, [messages, autoScroll]);

    const filtered = messages.filter((message) => {
        const agentOk = filter === 'ALL' || message.agent === filter || message.company === filter;
        const typeOk = typeFilter === 'ALL' || message.type === typeFilter;
        return agentOk && typeOk;
    });

    const clearFeed = () => {
        setMessages([]);
        setMsgCount(0);
    };

    const exportFeed = () => {
        const blob = new Blob([JSON.stringify(filtered, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `hive_feed_${Date.now()}.json`;
        link.click();
    };

    const formatTime = (timestamp: string) => {
        const parsed = new Date(timestamp);
        if (Number.isNaN(parsed.getTime())) {
            return '--:--:--';
        }
        return parsed.toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    };

    const displayMessages = filtered;

    return (
        <div
            style={{
                display: 'flex',
                flexDirection: 'column',
                height: '100%',
                background: '#060611',
                color: '#e0e0e0',
                fontFamily: "'Inter', sans-serif",
                overflow: 'hidden',
            }}
        >
            <div
                style={{
                    padding: '12px 16px',
                    borderBottom: '1px solid #1a2a3a',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '12px',
                    flexShrink: 0,
                }}
            >
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <MessageSquare size={14} color="#00ff41" />
                    <span style={{ fontSize: '12px', letterSpacing: '2px', color: '#00ff41', textTransform: 'uppercase' }}>
                        Flux agents
                    </span>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                    <Circle size={7} color={connected ? '#00ff41' : '#ff4444'} fill={connected ? '#00ff41' : '#ff4444'} />
                    <span style={{ fontSize: '10px', color: connected ? '#00ff41' : '#ff4444' }}>
                        {connected ? 'LIVE' : feedAvailable === false ? 'INDISPONIBLE' : 'REPRISE / POLLING'}
                    </span>
                </div>

                <div style={{ marginLeft: 'auto', display: 'flex', gap: '6px', alignItems: 'center' }}>
                    <span style={{ fontSize: '11px', color: '#444' }}>{msgCount} messages</span>
                    <button
                        onClick={() => setAutoScroll((value) => !value)}
                        style={{
                            padding: '4px 8px',
                            background: autoScroll ? '#001500' : '#0a0a1a',
                            border: `1px solid ${autoScroll ? '#00ff41' : '#1a2a3a'}`,
                            borderRadius: '4px',
                            color: autoScroll ? '#00ff41' : '#555',
                            fontSize: '10px',
                            cursor: 'pointer',
                        }}
                    >
                        AUTO
                    </button>
                    <button
                        onClick={exportFeed}
                        style={{
                            background: 'none',
                            border: '1px solid #1a2a3a',
                            borderRadius: '4px',
                            padding: '4px 8px',
                            color: '#555',
                            cursor: 'pointer',
                            fontSize: '10px',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '4px',
                        }}
                    >
                        <Download size={10} /> Export
                    </button>
                    <button
                        onClick={clearFeed}
                        style={{
                            background: 'none',
                            border: '1px solid #1a2a3a',
                            borderRadius: '4px',
                            padding: '4px 8px',
                            color: '#555',
                            cursor: 'pointer',
                            fontSize: '10px',
                        }}
                    >
                        Vider
                    </button>
                </div>
            </div>

            <div
                style={{
                    padding: '8px 16px',
                    borderBottom: '1px solid #1a2a3a',
                    display: 'flex',
                    gap: '6px',
                    overflowX: 'auto',
                    flexShrink: 0,
                }}
            >
                <Filter size={11} color="#444" style={{ flexShrink: 0, marginTop: '3px' }} />
                {agents.map((agent) => (
                    <button
                        key={agent}
                        onClick={() => setFilter(agent)}
                        style={{
                            padding: '3px 8px',
                            borderRadius: '3px',
                            border: 'none',
                            cursor: 'pointer',
                            fontSize: '10px',
                            whiteSpace: 'nowrap',
                            background: filter === agent ? AGENT_COLORS[agent] || '#00ff41' : '#0d0d1a',
                            color: filter === agent ? '#000' : '#555',
                            fontWeight: filter === agent ? 700 : 400,
                        }}
                    >
                        {agent}
                    </button>
                ))}
                <div style={{ width: '1px', background: '#1a2a3a', margin: '0 4px' }} />
                {types.map((type) => (
                    <button
                        key={type}
                        onClick={() => setTypeFilter(type)}
                        style={{
                            padding: '3px 8px',
                            borderRadius: '3px',
                            border: 'none',
                            cursor: 'pointer',
                            fontSize: '10px',
                            background: typeFilter === type ? '#1a2a3a' : 'none',
                            color: typeFilter === type ? '#aaa' : '#444',
                        }}
                    >
                        {type === 'ALL' ? 'ALL' : TYPE_LABELS[type]} {type}
                    </button>
                ))}
            </div>

            <div
                ref={feedRef}
                style={{
                    flex: 1,
                    overflowY: 'auto',
                    padding: '8px 16px',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '4px',
                }}
            >
                {connected && displayMessages.length === 0 && (
                    <div
                        style={{
                            display: 'flex',
                            flexDirection: 'column',
                            alignItems: 'center',
                            justifyContent: 'center',
                            height: '100%',
                            opacity: 0.3,
                        }}
                    >
                        <div style={{ fontSize: '10px', letterSpacing: '2px', color: '#00ff41', textTransform: 'uppercase', marginBottom: '8px' }}>
                            Surveillance du reseau neural...
                        </div>
                        <div style={{ fontSize: '11px', color: '#555' }}>En attente des signaux Kernel</div>
                    </div>
                )}

                {feedAvailable === false && (
                    <div
                        style={{
                            display: 'flex',
                            flexDirection: 'column',
                            alignItems: 'center',
                            justifyContent: 'center',
                            height: '100%',
                            opacity: 0.5,
                        }}
                    >
                        <div style={{ fontSize: '10px', letterSpacing: '2px', color: '#f0a500', textTransform: 'uppercase', marginBottom: '8px' }}>
                            Feed kernel indisponible
                        </div>
                        <div style={{ fontSize: '11px', color: '#777' }}>
                            Aucun message simule n est affiche. La vue attend un flux backend reel.
                        </div>
                    </div>
                )}

                {feedAvailable !== false && displayMessages.map((message) => (
                    <div
                        key={message.id}
                        style={{
                            display: 'flex',
                            gap: '10px',
                            padding: '7px 10px',
                            borderRadius: '5px',
                            background: message.type === 'error' ? '#1a0a0a' : message.type === 'result' ? '#001500' : 'transparent',
                            border: `1px solid ${message.type === 'error' ? '#330000' : message.type === 'result' ? '#003300' : '#0d0d1a'}`,
                            alignItems: 'flex-start',
                        }}
                    >
                        <span style={{ fontSize: '10px', color: '#333', fontFamily: 'monospace', flexShrink: 0, marginTop: '1px' }}>
                            {formatTime(message.timestamp)}
                        </span>
                        <span style={{ fontSize: '11px', flexShrink: 0 }}>{TYPE_LABELS[message.type]}</span>
                        <span
                            style={{
                                fontSize: '10px',
                                fontWeight: 700,
                                flexShrink: 0,
                                background: `${AGENT_COLORS[message.agent] || '#444'}22`,
                                color: AGENT_COLORS[message.agent] || '#888',
                                border: `1px solid ${AGENT_COLORS[message.agent] || '#444'}44`,
                                borderRadius: '3px',
                                padding: '0 5px',
                            }}
                        >
                            {message.agent}
                        </span>
                        {message.target && (
                            <>
                                <ChevronRight size={11} color="#333" style={{ flexShrink: 0, marginTop: '1px' }} />
                                <span style={{ fontSize: '10px', color: AGENT_COLORS[message.target] || '#888', flexShrink: 0 }}>
                                    {message.target}
                                </span>
                            </>
                        )}
                        <span style={{ fontSize: '12px', color: message.type === 'error' ? '#ff6666' : '#c0c0c0', lineHeight: 1.4, wordBreak: 'break-word' }}>
                            {message.content}
                        </span>
                    </div>
                ))}

                {feedAvailable !== false && displayMessages.length === 0 && (
                    <div style={{ textAlign: 'center', color: '#2a2a3a', padding: '60px', fontSize: '13px' }}>
                        <Zap size={32} style={{ marginBottom: '12px', opacity: 0.3 }} />
                        <div>En attente des transmissions des agents...</div>
                    </div>
                )}
            </div>
        </div>
    );
}
