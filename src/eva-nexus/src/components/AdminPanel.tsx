import { useCallback, useEffect, useState } from 'react'
import { Lock, ShieldCheck, UserPlus, Users } from 'lucide-react'

import { getAuthHeaders, useAuthStore, type User, type UserRole } from '../stores/authStore'

interface UserCreateForm {
    username: string
    password: string
    display_name: string
    role: UserRole
}

const ROLE_CONFIG: Record<UserRole, { label: string; color: string; description: string }> = {
    admin: {
        label: 'ADMIN',
        color: 'text-cyber-amber',
        description: 'Acces complet, gestion des utilisateurs et des droits.',
    },
    operator: {
        label: 'OPERATEUR',
        color: 'text-cyber-cyan',
        description: 'Acces exploitation, trading et monitoring.',
    },
    viewer: {
        label: 'LECTURE',
        color: 'text-white/50',
        description: 'Consultation seule des vues Nexus.',
    },
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
        username: '',
        password: '',
        display_name: '',
        role: 'viewer',
    })

    const fetchUsers = useCallback(async () => {
        try {
            const res = await fetch('/api/core/auth/users', { headers: getAuthHeaders() })
            if (res.ok) {
                setUsers(await res.json())
                setError(null)
            } else if (res.status === 403) {
                setError('Acces refuse: role admin requis.')
            } else {
                setError('Erreur lors du chargement des utilisateurs.')
            }
        } catch {
            setError('Connexion au serveur impossible.')
        } finally {
            setIsLoading(false)
        }
    }, [])

    useEffect(() => {
        void fetchUsers()
    }, [fetchUsers])

    useEffect(() => {
        if (!success) {
            return
        }
        const timeout = setTimeout(() => setSuccess(null), 3000)
        return () => clearTimeout(timeout)
    }, [success])

    const handleCreate = async (event: React.FormEvent) => {
        event.preventDefault()
        setError(null)

        if (!form.username.trim() || !form.password.trim()) {
            setError('Nom d utilisateur et mot de passe requis.')
            return
        }

        try {
            const res = await fetch('/api/core/auth/users', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
                body: JSON.stringify(form),
            })

            if (res.ok) {
                setSuccess(`Utilisateur ${form.username} cree avec succes.`)
                setForm({ username: '', password: '', display_name: '', role: 'viewer' })
                setShowCreateForm(false)
                await fetchUsers()
            } else if (res.status === 400) {
                setError('Utilisateur deja existant ou role invalide.')
            } else {
                setError('Creation impossible pour le moment.')
            }
        } catch {
            setError('Connexion au serveur impossible.')
        }
    }

    const handleUpdateRole = async (username: string, role: UserRole) => {
        try {
            const res = await fetch(`/api/core/auth/users/${username}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
                body: JSON.stringify({ role }),
            })
            if (res.ok) {
                setSuccess(`Role de ${username} mis a jour vers ${role}.`)
                setEditingUser(null)
                await fetchUsers()
            } else {
                setError('Mise a jour du role impossible.')
            }
        } catch {
            setError('Connexion au serveur impossible.')
        }
    }

    const handleToggleActive = async (username: string, isActive: boolean) => {
        try {
            const res = await fetch(`/api/core/auth/users/${username}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
                body: JSON.stringify({ is_active: !isActive }),
            })
            if (res.ok) {
                setSuccess(`Utilisateur ${username} ${!isActive ? 'active' : 'desactive'}.`)
                await fetchUsers()
            } else {
                setError('Changement d etat impossible.')
            }
        } catch {
            setError('Connexion au serveur impossible.')
        }
    }

    const handleDelete = async (username: string) => {
        if (username === 'admin') {
            return
        }
        if (!confirm(`Supprimer l utilisateur ${username} ?`)) {
            return
        }

        try {
            const res = await fetch(`/api/core/auth/users/${username}`, {
                method: 'DELETE',
                headers: getAuthHeaders(),
            })
            if (res.ok) {
                setSuccess(`Utilisateur ${username} supprime.`)
                await fetchUsers()
            } else {
                setError('Suppression impossible pour cet utilisateur.')
            }
        } catch {
            setError('Connexion au serveur impossible.')
        }
    }

    if (currentUser?.role !== 'admin') {
        return (
            <div className="h-full flex items-center justify-center animate-fade-in">
                <div className="cyber-panel hud-corners p-8 text-center max-w-md">
                    <Lock size={42} className="mx-auto text-cyber-pink mb-4" />
                    <div className="text-sm text-cyber-pink tracking-wider uppercase font-bold">Acces refuse</div>
                    <div className="text-[10px] text-white/20 mt-2">Le role admin est requis pour gerer les utilisateurs.</div>
                </div>
            </div>
        )
    }

    return (
        <div className="h-full overflow-y-auto p-4 space-y-4 animate-fade-in">
            <div className="flex items-center justify-between gap-4">
                <div className="flex items-center gap-3">
                    <Users size={22} className="text-cyber-cyan" />
                    <div>
                        <h3 className="font-display text-sm font-bold tracking-[0.15em] text-white/80 uppercase">
                            Gestion des utilisateurs
                        </h3>
                        <div className="text-[8px] text-matrix/40 tracking-[0.3em] uppercase">
                            Admin panel | RBAC
                        </div>
                    </div>
                </div>
                <button
                    onClick={() => setShowCreateForm((value) => !value)}
                    className={`cyber-btn py-2 ${showCreateForm ? 'cyber-btn-danger' : 'cyber-btn-cyan'}`}
                >
                    {showCreateForm ? 'Annuler' : 'Nouvel utilisateur'}
                </button>
            </div>

            {error && (
                <div className="p-3 border border-cyber-pink/30 bg-cyber-pink/5 text-cyber-pink text-[11px] animate-fade-in flex items-center gap-2">
                    <ShieldCheck size={14} />
                    <span>{error}</span>
                    <button onClick={() => setError(null)} className="ml-auto text-white/20 hover:text-white/50">
                        Fermer
                    </button>
                </div>
            )}
            {success && (
                <div className="p-3 border border-matrix/30 bg-matrix/5 text-matrix text-[11px] animate-fade-in flex items-center gap-2">
                    <ShieldCheck size={14} />
                    <span>{success}</span>
                </div>
            )}

            {showCreateForm && (
                <div className="cyber-panel hud-corners p-5 animate-fade-in">
                    <div className="flex items-center gap-2 text-[9px] uppercase tracking-[0.2em] text-cyber-cyan/50 mb-4">
                        <UserPlus size={14} />
                        <span>Creer un utilisateur</span>
                    </div>
                    <form onSubmit={handleCreate} className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                            <label className="text-[8px] text-white/20 tracking-[0.2em] uppercase mb-1 block">Identifiant *</label>
                            <input type="text" value={form.username} onChange={(e) => setForm({ ...form, username: e.target.value })} className="cyber-input" placeholder="john_doe" />
                        </div>
                        <div>
                            <label className="text-[8px] text-white/20 tracking-[0.2em] uppercase mb-1 block">Nom affiche</label>
                            <input type="text" value={form.display_name} onChange={(e) => setForm({ ...form, display_name: e.target.value })} className="cyber-input" placeholder="John Doe" />
                        </div>
                        <div>
                            <label className="text-[8px] text-white/20 tracking-[0.2em] uppercase mb-1 block">Mot de passe *</label>
                            <input type="password" value={form.password} onChange={(e) => setForm({ ...form, password: e.target.value })} className="cyber-input" placeholder="********" />
                        </div>
                        <div>
                            <label className="text-[8px] text-white/20 tracking-[0.2em] uppercase mb-1 block">Role</label>
                            <select value={form.role} onChange={(e) => setForm({ ...form, role: e.target.value as UserRole })} className="cyber-input">
                                <option value="viewer">Lecture</option>
                                <option value="operator">Operateur</option>
                                <option value="admin">Admin</option>
                            </select>
                        </div>
                        <div className="md:col-span-2 flex justify-end">
                            <button type="submit" className="cyber-btn cyber-btn-cyan py-2">Creer</button>
                        </div>
                    </form>
                </div>
            )}

            <div className="grid grid-cols-1 xl:grid-cols-[0.8fr_1.2fr] gap-4">
                <div className="cyber-panel hud-corners p-5 space-y-3">
                    <div className="text-[9px] uppercase tracking-[0.2em] text-matrix/40">Roles</div>
                    {Object.entries(ROLE_CONFIG).map(([role, config]) => (
                        <div key={role} className="border border-white/[0.05] bg-white/[0.02] p-4">
                            <div className={`text-[10px] font-bold uppercase tracking-[0.18em] ${config.color}`}>{config.label}</div>
                            <div className="mt-2 text-[10px] text-white/30 leading-relaxed">{config.description}</div>
                        </div>
                    ))}
                </div>

                <div className="cyber-panel hud-corners p-5">
                    <div className="text-[9px] uppercase tracking-[0.2em] text-matrix/40 mb-4">Utilisateurs</div>
                    {isLoading ? (
                        <div className="text-[10px] text-white/25">Chargement des utilisateurs...</div>
                    ) : users.length === 0 ? (
                        <div className="text-[10px] text-white/25">Aucun utilisateur disponible.</div>
                    ) : (
                        <div className="space-y-3">
                            {users.map((user) => (
                                <div key={user.username} className="border border-white/[0.05] bg-white/[0.02] p-4">
                                    <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
                                        <div>
                                            <div className="text-[11px] font-bold text-white/75">{user.display_name || user.username}</div>
                                            <div className="text-[9px] text-white/25">{user.username}</div>
                                        </div>
                                        <div className="flex items-center gap-2 flex-wrap">
                                            <span className={`px-2 py-0.5 text-[8px] uppercase tracking-[0.18em] ${ROLE_CONFIG[user.role].color}`}>{ROLE_CONFIG[user.role].label}</span>
                                            <span className={`px-2 py-0.5 text-[8px] uppercase tracking-[0.18em] border ${user.is_active ? 'border-matrix/20 bg-matrix/10 text-matrix/70' : 'border-cyber-pink/20 bg-cyber-pink/10 text-cyber-pink/70'}`}>
                                                {user.is_active ? 'actif' : 'inactif'}
                                            </span>
                                        </div>
                                    </div>

                                    <div className="mt-4 flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
                                        <div className="flex items-center gap-2 flex-wrap">
                                            {editingUser === user.username ? (
                                                <>
                                                    <select value={editRole} onChange={(e) => setEditRole(e.target.value as UserRole)} className="cyber-input min-w-[150px]">
                                                        <option value="viewer">Lecture</option>
                                                        <option value="operator">Operateur</option>
                                                        <option value="admin">Admin</option>
                                                    </select>
                                                    <button onClick={() => void handleUpdateRole(user.username, editRole)} className="cyber-btn cyber-btn-cyan py-2">Valider</button>
                                                    <button onClick={() => setEditingUser(null)} className="cyber-btn py-2">Annuler</button>
                                                </>
                                            ) : (
                                                <button
                                                    onClick={() => {
                                                        setEditingUser(user.username)
                                                        setEditRole(user.role)
                                                    }}
                                                    className="cyber-btn py-2"
                                                >
                                                    Changer le role
                                                </button>
                                            )}
                                        </div>
                                        <div className="flex gap-2 flex-wrap">
                                            <button onClick={() => void handleToggleActive(user.username, user.is_active)} className="cyber-btn py-2">
                                                {user.is_active ? 'Desactiver' : 'Activer'}
                                            </button>
                                            {user.username !== 'admin' && (
                                                <button onClick={() => void handleDelete(user.username)} className="cyber-btn cyber-btn-danger py-2">
                                                    Supprimer
                                                </button>
                                            )}
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            </div>
        </div>
    )
}
