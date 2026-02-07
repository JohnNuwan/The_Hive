import { useState, useEffect, useCallback } from 'react'
import { useAuthStore, getAuthHeaders, type User, type UserRole } from '../stores/authStore'

// ═══ TYPES ═══
interface UserCreateForm {
    username: string
    password: string
    display_name: string
    role: UserRole
}

// ═══ ROLE CONFIG ═══
const ROLE_CONFIG: Record<UserRole, { label: string; icon: string; color: string; description: string }> = {
    admin: { label: 'ADMIN', icon: '👑', color: 'text-cyber-amber', description: 'Full system access + user management' },
    operator: { label: 'OPERATOR', icon: '⚙', color: 'neon-text-cyan', description: 'Trading, monitoring, OSINT access' },
    viewer: { label: 'VIEWER', icon: '👁', color: 'text-white/50', description: 'Read-only dashboard access' },
}

export default function AdminPanel() {
    const { user: currentUser } = useAuthStore()
    const [users, setUsers] = useState<User[]>([])
    const [isLoading, setIsLoading] = useState(true)
    const [error, setError] = useState<string | null>(null)
    const [success, setSuccess] = useState<string | null>(null)
    const [showCreateForm, setShowCreateForm] = useState(false)
    const [editingUser, setEditingUser] = useState<string | null>(null)
    const [editRole, setEditRole] = useState<UserRole>('viewer')
    const [form, setForm] = useState<UserCreateForm>({
        username: '', password: '', display_name: '', role: 'viewer',
    })

    // ═══ FETCH USERS ═══
    const fetchUsers = useCallback(async () => {
        try {
            const res = await fetch('/api/core/auth/users', {
                headers: getAuthHeaders(),
            })
            if (res.ok) {
                const data = await res.json()
                setUsers(data)
            } else if (res.status === 403) {
                setError('Accès refusé — rôle Admin requis')
            } else {
                setError('Erreur lors du chargement des utilisateurs')
            }
        } catch {
            setError('Connexion au serveur impossible')
        } finally {
            setIsLoading(false)
        }
    }, [])

    useEffect(() => {
        fetchUsers()
    }, [fetchUsers])

    // Clear messages after delay
    useEffect(() => {
        if (success) {
            const t = setTimeout(() => setSuccess(null), 3000)
            return () => clearTimeout(t)
        }
    }, [success])

    // ═══ CREATE USER ═══
    const handleCreate = async (e: React.FormEvent) => {
        e.preventDefault()
        setError(null)

        if (!form.username.trim() || !form.password.trim()) {
            setError('Nom d\'utilisateur et mot de passe requis')
            return
        }

        try {
            const res = await fetch('/api/core/auth/users', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
                body: JSON.stringify(form),
            })

            if (res.ok) {
                setSuccess(`Utilisateur "${form.username}" créé avec succès`)
                setForm({ username: '', password: '', display_name: '', role: 'viewer' })
                setShowCreateForm(false)
                fetchUsers()
            } else if (res.status === 400) {
                setError('L\'utilisateur existe déjà ou rôle invalide')
            } else {
                setError('Erreur lors de la création')
            }
        } catch {
            setError('Connexion au serveur impossible')
        }
    }

    // ═══ UPDATE ROLE ═══
    const handleUpdateRole = async (username: string, role: UserRole) => {
        try {
            const res = await fetch(`/api/core/auth/users/${username}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
                body: JSON.stringify({ role }),
            })
            if (res.ok) {
                setSuccess(`Rôle de "${username}" mis à jour → ${role.toUpperCase()}`)
                setEditingUser(null)
                fetchUsers()
            } else {
                setError('Erreur lors de la mise à jour')
            }
        } catch {
            setError('Connexion au serveur impossible')
        }
    }

    // ═══ TOGGLE ACTIVE ═══
    const handleToggleActive = async (username: string, is_active: boolean) => {
        try {
            const res = await fetch(`/api/core/auth/users/${username}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
                body: JSON.stringify({ is_active: !is_active }),
            })
            if (res.ok) {
                setSuccess(`Utilisateur "${username}" ${!is_active ? 'activé' : 'désactivé'}`)
                fetchUsers()
            }
        } catch {
            setError('Connexion au serveur impossible')
        }
    }

    // ═══ DELETE USER ═══
    const handleDelete = async (username: string) => {
        if (username === 'admin') return
        if (!confirm(`Supprimer l'utilisateur "${username}" ?`)) return

        try {
            const res = await fetch(`/api/core/auth/users/${username}`, {
                method: 'DELETE',
                headers: getAuthHeaders(),
            })
            if (res.ok) {
                setSuccess(`Utilisateur "${username}" supprimé`)
                fetchUsers()
            } else {
                setError('Impossible de supprimer cet utilisateur')
            }
        } catch {
            setError('Connexion au serveur impossible')
        }
    }

    if (currentUser?.role !== 'admin') {
        return (
            <div className="h-full flex items-center justify-center animate-fade-in">
                <div className="cyber-panel hud-corners p-8 text-center">
                    <div className="text-4xl mb-4">🔒</div>
                    <div className="text-sm text-cyber-pink tracking-wider">ACCÈS REFUSÉ</div>
                    <div className="text-[10px] text-white/20 mt-2">Rôle ADMIN requis pour la gestion des utilisateurs</div>
                </div>
            </div>
        )
    }

    return (
        <div className="h-full overflow-y-auto p-4 space-y-4 animate-fade-in">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                    <span className="text-2xl">👥</span>
                    <div>
                        <h3 className="font-display text-sm font-bold tracking-[0.15em] text-white/80">
                            GESTION DES UTILISATEURS
                        </h3>
                        <div className="text-[8px] text-matrix/40 tracking-[0.3em]">
                            ADMIN PANEL — RBAC MANAGEMENT
                        </div>
                    </div>
                </div>
                <button
                    onClick={() => setShowCreateForm(!showCreateForm)}
                    className={`cyber-btn ${showCreateForm ? 'cyber-btn-danger' : 'cyber-btn-cyan'} py-2`}
                >
                    {showCreateForm ? '✕ ANNULER' : '+ NOUVEL UTILISATEUR'}
                </button>
            </div>

            {/* Messages */}
            {error && (
                <div className="p-3 border border-cyber-pink/30 bg-cyber-pink/5 text-cyber-pink text-[11px] animate-fade-in flex items-center gap-2">
                    <span>⚠</span> {error}
                    <button onClick={() => setError(null)} className="ml-auto text-white/20 hover:text-white/50">✕</button>
                </div>
            )}
            {success && (
                <div className="p-3 border border-matrix/30 bg-matrix/5 text-matrix text-[11px] animate-fade-in flex items-center gap-2">
                    <span>✓</span> {success}
                </div>
            )}

            {/* Create User Form */}
            {showCreateForm && (
                <div className="cyber-panel hud-corners p-5 animate-fade-in">
                    <div className="text-[9px] uppercase tracking-[0.2em] text-cyber-cyan/50 mb-4">
                        ➕ CRÉER UN UTILISATEUR
                    </div>
                    <form onSubmit={handleCreate} className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                            <label className="text-[8px] text-white/20 tracking-[0.2em] uppercase mb-1 block">
                                Identifiant *
                            </label>
                            <input
                                type="text"
                                value={form.username}
                                onChange={e => setForm({ ...form, username: e.target.value })}
                                className="cyber-input"
                                placeholder="john_doe"
                            />
                        </div>
                        <div>
                            <label className="text-[8px] text-white/20 tracking-[0.2em] uppercase mb-1 block">
                                Mot de passe *
                            </label>
                            <input
                                type="password"
                                value={form.password}
                                onChange={e => setForm({ ...form, password: e.target.value })}
                                className="cyber-input"
                                placeholder="••••••••"
                            />
                        </div>
                        <div>
                            <label className="text-[8px] text-white/20 tracking-[0.2em] uppercase mb-1 block">
                                Nom affiché
                            </label>
                            <input
                                type="text"
                                value={form.display_name}
                                onChange={e => setForm({ ...form, display_name: e.target.value })}
                                className="cyber-input"
                                placeholder="John Doe"
                            />
                        </div>
                        <div>
                            <label className="text-[8px] text-white/20 tracking-[0.2em] uppercase mb-1 block">
                                Rôle
                            </label>
                            <div className="flex gap-2">
                                {(Object.keys(ROLE_CONFIG) as UserRole[]).map(role => (
                                    <button
                                        key={role}
                                        type="button"
                                        onClick={() => setForm({ ...form, role })}
                                        className={`flex-1 p-2 text-[9px] border transition-all text-center ${form.role === role
                                            ? 'border-matrix/40 bg-matrix/10 text-matrix'
                                            : 'border-white/[0.06] text-white/25 hover:text-white/50'}`}
                                    >
                                        <div>{ROLE_CONFIG[role].icon}</div>
                                        <div className="mt-0.5">{ROLE_CONFIG[role].label}</div>
                                    </button>
                                ))}
                            </div>
                        </div>
                        <div className="md:col-span-2">
                            <button type="submit" className="cyber-btn cyber-btn-cyan w-full py-2.5">
                                CRÉER L'UTILISATEUR
                            </button>
                        </div>
                    </form>
                </div>
            )}

            {/* Roles Legend */}
            <div className="cyber-panel hud-corners p-4">
                <div className="text-[9px] uppercase tracking-[0.2em] text-matrix/40 mb-3">
                    🔐 RÔLES & PERMISSIONS
                </div>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                    {(Object.entries(ROLE_CONFIG) as [UserRole, typeof ROLE_CONFIG[UserRole]][]).map(([role, cfg]) => (
                        <div key={role} className="p-3 border border-white/[0.04] bg-white/[0.01]">
                            <div className="flex items-center gap-2 mb-1">
                                <span>{cfg.icon}</span>
                                <span className={`text-[10px] font-bold tracking-wider ${cfg.color}`}>{cfg.label}</span>
                            </div>
                            <div className="text-[9px] text-white/25">{cfg.description}</div>
                        </div>
                    ))}
                </div>
            </div>

            {/* Users Table */}
            <div className="cyber-panel hud-corners p-4">
                <div className="text-[9px] uppercase tracking-[0.2em] text-matrix/40 mb-3">
                    👥 UTILISATEURS ({users.length})
                </div>

                {isLoading ? (
                    <div className="py-8 text-center text-[10px] text-white/20 animate-pulse">
                        CHARGEMENT DES UTILISATEURS...
                    </div>
                ) : users.length === 0 ? (
                    <div className="py-8 text-center text-[10px] text-white/20">
                        AUCUN UTILISATEUR TROUVÉ — Redis indisponible ?
                    </div>
                ) : (
                    <>
                        {/* Table Header */}
                        <div className="flex items-center gap-3 py-2 px-3 text-[8px] text-white/15 tracking-wider border-b border-white/[0.04]">
                            <div className="w-8 shrink-0" />
                            <div className="w-32 shrink-0">IDENTIFIANT</div>
                            <div className="w-36 shrink-0">NOM AFFICHÉ</div>
                            <div className="w-24 shrink-0">RÔLE</div>
                            <div className="w-20 shrink-0">STATUT</div>
                            <div className="w-32 shrink-0">CRÉÉ LE</div>
                            <div className="flex-1 text-right">ACTIONS</div>
                        </div>

                        {/* User Rows */}
                        {users.map(u => (
                            <div
                                key={u.username}
                                className={`flex items-center gap-3 py-3 px-3 border-b border-white/[0.02] hover:bg-white/[0.01] transition-all ${!u.is_active ? 'opacity-40' : ''}`}
                            >
                                {/* Avatar */}
                                <div className={`w-8 h-8 flex items-center justify-center border text-[9px] font-bold shrink-0 ${u.role === 'admin' ? 'border-cyber-amber/30 text-cyber-amber bg-cyber-amber/5' : u.role === 'operator' ? 'border-cyber-cyan/30 text-cyber-cyan bg-cyber-cyan/5' : 'border-white/10 text-white/30 bg-white/[0.02]'}`}>
                                    {u.display_name?.slice(0, 2).toUpperCase() || u.username.slice(0, 2).toUpperCase()}
                                </div>

                                {/* Username */}
                                <div className="w-32 shrink-0">
                                    <div className="text-[10px] text-white/70 font-bold tracking-wider">
                                        {u.username}
                                        {u.username === 'admin' && <span className="ml-1 text-[7px] text-cyber-amber/50">🔒</span>}
                                    </div>
                                </div>

                                {/* Display Name */}
                                <div className="w-36 shrink-0 text-[10px] text-white/40 truncate">
                                    {u.display_name || '—'}
                                </div>

                                {/* Role */}
                                <div className="w-24 shrink-0">
                                    {editingUser === u.username ? (
                                        <div className="flex items-center gap-1">
                                            <select
                                                value={editRole}
                                                onChange={e => setEditRole(e.target.value as UserRole)}
                                                className="text-[9px] bg-black border border-matrix/30 text-matrix py-0.5 px-1 outline-none"
                                            >
                                                <option value="admin">ADMIN</option>
                                                <option value="operator">OPERATOR</option>
                                                <option value="viewer">VIEWER</option>
                                            </select>
                                            <button
                                                onClick={() => handleUpdateRole(u.username, editRole)}
                                                className="text-[9px] text-matrix hover:text-matrix/80"
                                            >
                                                ✓
                                            </button>
                                            <button
                                                onClick={() => setEditingUser(null)}
                                                className="text-[9px] text-white/20 hover:text-white/50"
                                            >
                                                ✕
                                            </button>
                                        </div>
                                    ) : (
                                        <span className={`text-[9px] tracking-wider ${ROLE_CONFIG[u.role as UserRole]?.color || 'text-white/30'}`}>
                                            {ROLE_CONFIG[u.role as UserRole]?.icon} {u.role.toUpperCase()}
                                        </span>
                                    )}
                                </div>

                                {/* Status */}
                                <div className="w-20 shrink-0 flex items-center gap-1.5">
                                    <div className={`w-2 h-2 rounded-full ${u.is_active ? 'bg-matrix' : 'bg-cyber-pink'}`}
                                        style={u.is_active ? { boxShadow: '0 0 6px rgba(0,255,65,0.4)' } : {}} />
                                    <span className={`text-[9px] ${u.is_active ? 'text-matrix/60' : 'text-cyber-pink/60'}`}>
                                        {u.is_active ? 'ACTIF' : 'INACTIF'}
                                    </span>
                                </div>

                                {/* Created */}
                                <div className="w-32 shrink-0 text-[8px] text-white/15">
                                    {u.created_at ? new Date(u.created_at).toLocaleDateString('fr-FR', {
                                        day: '2-digit', month: 'short', year: 'numeric',
                                    }) : '—'}
                                </div>

                                {/* Actions */}
                                <div className="flex-1 flex items-center justify-end gap-2">
                                    {u.username !== currentUser?.username && (
                                        <>
                                            <button
                                                onClick={() => { setEditingUser(u.username); setEditRole(u.role as UserRole) }}
                                                className="text-[9px] px-2 py-1 border border-white/[0.06] text-white/25 hover:text-cyber-cyan hover:border-cyber-cyan/30 transition-all"
                                                title="Modifier le rôle"
                                            >
                                                ✎ RÔLE
                                            </button>
                                            <button
                                                onClick={() => handleToggleActive(u.username, u.is_active)}
                                                className={`text-[9px] px-2 py-1 border transition-all ${u.is_active
                                                    ? 'border-cyber-amber/20 text-cyber-amber/40 hover:text-cyber-amber'
                                                    : 'border-matrix/20 text-matrix/40 hover:text-matrix'}`}
                                            >
                                                {u.is_active ? '⏸ DISABLE' : '▶ ENABLE'}
                                            </button>
                                            {u.username !== 'admin' && (
                                                <button
                                                    onClick={() => handleDelete(u.username)}
                                                    className="text-[9px] px-2 py-1 border border-cyber-pink/10 text-cyber-pink/30 hover:text-cyber-pink hover:border-cyber-pink/40 transition-all"
                                                    title="Supprimer"
                                                >
                                                    ✕
                                                </button>
                                            )}
                                        </>
                                    )}
                                    {u.username === currentUser?.username && (
                                        <span className="text-[8px] text-white/10 tracking-wider">VOUS</span>
                                    )}
                                </div>
                            </div>
                        ))}
                    </>
                )}
            </div>
        </div>
    )
}
