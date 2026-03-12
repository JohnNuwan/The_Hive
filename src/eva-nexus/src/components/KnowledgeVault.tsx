import { useState, useCallback, useRef } from 'react';
import {
    Upload, FileText, Database, Trash2, CheckCircle,
    AlertCircle, BookOpen, Search, Loader
} from 'lucide-react';
import { createClientUuid } from '../services/api';

interface UploadedDoc {
    id: string;
    name: string;
    type: string;
    size: number;
    status: 'uploading' | 'processing' | 'indexed' | 'error';
    chunks?: number;
    error?: string;
    uploadedAt: string;
}

export default function KnowledgeVault() {
    const CORE_API = '/api/core';
    const [docs, setDocs] = useState<UploadedDoc[]>([]);
    const [dragging, setDragging] = useState(false);
    const [searchQuery, setSearchQuery] = useState('');
    const [searchResults, setSearchResults] = useState<any[]>([]);
    const [searching, setSearching] = useState(false);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const uploadFile = async (file: File) => {
        const docId = createClientUuid();
        const newDoc: UploadedDoc = {
            id: docId,
            name: file.name,
            type: file.type || 'application/octet-stream',
            size: file.size,
            status: 'uploading',
            uploadedAt: new Date().toISOString(),
        };
        setDocs(prev => [newDoc, ...prev]);

        try {
            const formData = new FormData();
            formData.append('file', file);
            formData.append('collection', 'hive_knowledge');

            const res = await fetch(`${CORE_API}/knowledge/upload`, {
                method: 'POST',
                body: formData,
            });

            if (!res.ok) throw new Error(await res.text());
            const data = await res.json();

            setDocs(prev => prev.map(d => d.id === docId ? {
                ...d, status: 'indexed', chunks: data.chunks
            } : d));
        } catch (e: any) {
            setDocs(prev => prev.map(d => d.id === docId ? {
                ...d, status: 'error', error: e.message
            } : d));
        }
    };

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setDragging(false);
        Array.from(e.dataTransfer.files).forEach(uploadFile);
    }, []);

    const handleSearch = async () => {
        if (!searchQuery.trim()) return;
        setSearching(true);
        try {
            const res = await fetch(`${CORE_API}/knowledge/search?q=${encodeURIComponent(searchQuery)}&limit=5`);
            const data = await res.json();
            setSearchResults(data.results || []);
        } catch {
            setSearchResults([]);
        } finally {
            setSearching(false);
        }
    };

    const deleteDoc = (id: string) => setDocs(prev => prev.filter(d => d.id !== id));

    const formatSize = (bytes: number) => {
        if (bytes < 1024) return `${bytes} B`;
        if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
        return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
    };

    const statusIcon = (status: string) => {
        switch (status) {
            case 'uploading': return <Loader size={13} color="#ffd700" style={{ animation: 'spin 1s linear infinite' }} />;
            case 'processing': return <Loader size={13} color="#00bfff" style={{ animation: 'spin 1s linear infinite' }} />;
            case 'indexed': return <CheckCircle size={13} color="#00ff41" />;
            case 'error': return <AlertCircle size={13} color="#ff4444" />;
            default: return null;
        }
    };

    const statusColor = (status: string) => ({
        uploading: '#ffd700', processing: '#00bfff', indexed: '#00ff41', error: '#ff4444'
    })[status] || '#888';

    const acceptedTypes = '.pdf,.txt,.md,.docx,.csv,.json,.html,.py,.ts,.js';

    return (
        <div style={{ display: 'flex', height: '100%', gap: '16px', padding: '16px', background: '#060611', color: '#e0e0e0', fontFamily: "'Inter', sans-serif", overflow: 'hidden' }}>

            {/* LEFT â€” Upload + List */}
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '12px', overflow: 'hidden' }}>

                {/* Header */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Database size={14} color="#00ff41" />
                    <span style={{ fontSize: '12px', letterSpacing: '2px', color: '#00ff41', textTransform: 'uppercase' }}>Knowledge Vault â€” RAG Engine</span>
                </div>

                {/* Drop Zone */}
                <div
                    onDrop={handleDrop}
                    onDragOver={e => { e.preventDefault(); setDragging(true); }}
                    onDragLeave={() => setDragging(false)}
                    onClick={() => fileInputRef.current?.click()}
                    style={{
                        border: `2px dashed ${dragging ? '#00ff41' : '#1a2a3a'}`,
                        borderRadius: '8px',
                        padding: '32px 16px',
                        textAlign: 'center',
                        cursor: 'pointer',
                        background: dragging ? '#001a00' : '#0a0a1a',
                        transition: 'all 0.2s',
                    }}
                >
                    <input
                        ref={fileInputRef}
                        type="file"
                        multiple
                        accept={acceptedTypes}
                        style={{ display: 'none' }}
                        onChange={e => Array.from(e.target.files || []).forEach(uploadFile)}
                    />
                    <Upload size={28} color={dragging ? '#00ff41' : '#2a3a4a'} style={{ marginBottom: '10px' }} />
                    <div style={{ fontSize: '14px', fontWeight: 600, color: dragging ? '#00ff41' : '#888' }}>
                        {dragging ? 'DÃ©poser ici' : 'Glisser des fichiers ou cliquer pour parcourir'}
                    </div>
                    <div style={{ fontSize: '11px', color: '#444', marginTop: '6px' }}>
                        PDF, TXT, DOCX, MD, CSV, JSON, HTML, Python, TypeScript
                    </div>
                </div>

                {/* Document List */}
                <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '6px' }}>
                    {docs.length === 0 ? (
                        <div style={{ textAlign: 'center', color: '#2a2a3a', padding: '40px', fontSize: '13px' }}>
                            <BookOpen size={32} style={{ marginBottom: '10px', opacity: 0.3 }} />
                            <div>Aucun document indexÃ©<br />Glissez des fichiers pour alimenter le RAG</div>
                        </div>
                    ) : (
                        docs.map(doc => (
                            <div key={doc.id} style={{
                                background: '#0a0a1a', border: '1px solid #1a2a3a', borderRadius: '6px',
                                padding: '10px 12px', display: 'flex', alignItems: 'center', gap: '10px'
                            }}>
                                <FileText size={14} color="#555" style={{ flexShrink: 0 }} />
                                <div style={{ flex: 1, minWidth: 0 }}>
                                    <div style={{ fontSize: '12px', fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{doc.name}</div>
                                    <div style={{ fontSize: '10px', color: '#555', display: 'flex', gap: '8px', marginTop: '2px' }}>
                                        <span>{formatSize(doc.size)}</span>
                                        {doc.chunks && <span style={{ color: '#00ff41' }}>{doc.chunks} chunks RAG</span>}
                                        {doc.error && <span style={{ color: '#ff4444' }}>{doc.error}</span>}
                                    </div>
                                </div>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                                    {statusIcon(doc.status)}
                                    <span style={{ fontSize: '10px', color: statusColor(doc.status), textTransform: 'uppercase' }}>{doc.status}</span>
                                </div>
                                <button onClick={() => deleteDoc(doc.id)} style={{ background: 'none', border: 'none', cursor: 'pointer', padding: '2px', color: '#333' }}>
                                    <Trash2 size={12} />
                                </button>
                            </div>
                        ))
                    )}
                </div>
            </div>

            {/* RIGHT â€” RAG Search */}
            <div style={{ width: '380px', display: 'flex', flexDirection: 'column', gap: '12px' }}>

                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Search size={14} color="#00ff41" />
                    <span style={{ fontSize: '12px', letterSpacing: '2px', color: '#00ff41', textTransform: 'uppercase' }}>RequÃªte RAG</span>
                </div>

                {/* Search Box */}
                <div style={{ display: 'flex', gap: '8px' }}>
                    <input
                        value={searchQuery}
                        onChange={e => setSearchQuery(e.target.value)}
                        onKeyDown={e => e.key === 'Enter' && handleSearch()}
                        placeholder="Chercher dans la base de connaissance..."
                        style={{
                            flex: 1, background: '#0a0a1a', border: '1px solid #1a2a3a', borderRadius: '6px',
                            color: '#c0c0c0', padding: '9px 12px', fontSize: '12px', outline: 'none'
                        }}
                    />
                    <button
                        onClick={handleSearch}
                        disabled={searching}
                        style={{
                            padding: '9px 14px', background: '#00ff41', border: 'none', borderRadius: '6px',
                            color: '#000', cursor: 'pointer', fontWeight: 700, fontSize: '12px'
                        }}
                    >
                        {searching ? <Loader size={12} style={{ animation: 'spin 1s linear infinite' }} /> : 'KG'}
                    </button>
                </div>

                {/* Results */}
                <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    {searchResults.length === 0 && !searching && (
                        <div style={{ textAlign: 'center', color: '#2a2a3a', padding: '30px', fontSize: '12px' }}>
                            <Search size={24} style={{ marginBottom: '8px', opacity: 0.3 }} />
                            <div>Lancez une requÃªte pour interroger<br />le graphe de connaissance d'EVA</div>
                        </div>
                    )}
                    {searchResults.map((r: any, i: number) => (
                        <div key={i} style={{ background: '#0a0a1a', border: '1px solid #1a2a3a', borderRadius: '6px', padding: '10px 12px' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
                                <span style={{ fontSize: '10px', color: '#00ff41' }}>Score: {(r.score * 100).toFixed(0)}%</span>
                                {r.source && <span style={{ fontSize: '10px', color: '#555' }}>{r.source}</span>}
                            </div>
                            <p style={{ fontSize: '12px', color: '#c0c0c0', margin: 0, lineHeight: 1.5 }}>{r.text}</p>
                        </div>
                    ))}
                </div>

                {/* Stats */}
                <div style={{ background: '#0a0a1a', border: '1px solid #1a2a3a', borderRadius: '6px', padding: '12px' }}>
                    <div style={{ fontSize: '11px', color: '#444', letterSpacing: '1px', marginBottom: '8px' }}>VAULT STATS</div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '4px' }}>
                        <span style={{ color: '#666' }}>Documents</span>
                        <span style={{ color: '#aaa' }}>{docs.length}</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '4px' }}>
                        <span style={{ color: '#666' }}>IndexÃ©s</span>
                        <span style={{ color: '#00ff41' }}>{docs.filter(d => d.status === 'indexed').length}</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px' }}>
                        <span style={{ color: '#666' }}>Total Chunks</span>
                        <span style={{ color: '#aaa' }}>{docs.reduce((s, d) => s + (d.chunks || 0), 0)}</span>
                    </div>
                </div>
            </div>

            <style>{`@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }`}</style>
        </div>
    );
}

