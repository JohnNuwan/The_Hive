/**
 * THE HIVE — Auth Store (Zustand)
 * Gestion de l'authentification JWT avec RBAC
 */
import { create } from 'zustand'

// ═══ TYPES ═══
export type UserRole = 'admin' | 'operator' | 'viewer'

export interface User {
    username: string
    role: UserRole
    display_name: string
    created_at: string
    is_active: boolean
}

interface AuthState {
    user: User | null
    token: string | null
    isAuthenticated: boolean
    isLoading: boolean
    error: string | null
    login: (username: string, password: string) => Promise<boolean>
    logout: () => void
    checkAuth: () => Promise<boolean>
    clearError: () => void
}

const TOKEN_KEY = 'hive-auth-token'
const USER_KEY = 'hive-auth-user'

export const useAuthStore = create<AuthState>((set) => ({
    user: (() => {
        try { return JSON.parse(localStorage.getItem(USER_KEY) || 'null') }
        catch { return null }
    })(),
    token: localStorage.getItem(TOKEN_KEY),
    isAuthenticated: !!localStorage.getItem(TOKEN_KEY),
    isLoading: true,
    error: null,

    login: async (username: string, password: string) => {
        set({ error: null, isLoading: true })
        try {
            const res = await fetch('/api/core/auth/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, password }),
            })

            if (!res.ok) {
                const detail = res.status === 401
                    ? 'Identifiants invalides'
                    : res.status === 503
                        ? 'Service d\'authentification indisponible'
                        : `Erreur serveur (${res.status})`
                set({ error: detail, isLoading: false })
                return false
            }

            const data = await res.json()
            const { access_token, user } = data

            localStorage.setItem(TOKEN_KEY, access_token)
            localStorage.setItem(USER_KEY, JSON.stringify(user))
            set({ user, token: access_token, isAuthenticated: true, isLoading: false, error: null })
            return true
        } catch {
            set({
                error: 'Connexion au serveur impossible. Vérifiez que EVA Core est actif.',
                isLoading: false,
            })
            return false
        }
    },

    logout: () => {
        localStorage.removeItem(TOKEN_KEY)
        localStorage.removeItem(USER_KEY)
        set({ user: null, token: null, isAuthenticated: false, error: null })
    },

    checkAuth: async () => {
        const token = localStorage.getItem(TOKEN_KEY)
        if (!token) {
            set({ isLoading: false, isAuthenticated: false })
            return false
        }

        try {
            const res = await fetch('/api/core/auth/me', {
                headers: { 'Authorization': `Bearer ${token}` },
            })

            if (res.ok) {
                const user = await res.json()
                localStorage.setItem(USER_KEY, JSON.stringify(user))
                set({ user, token, isAuthenticated: true, isLoading: false })
                return true
            }
        } catch {
            // Server unreachable — use cached user data
            const cached = localStorage.getItem(USER_KEY)
            if (cached) {
                try {
                    const user = JSON.parse(cached)
                    set({ user, token, isAuthenticated: true, isLoading: false })
                    return true
                } catch { /* fall through */ }
            }
        }

        // Token invalid
        localStorage.removeItem(TOKEN_KEY)
        localStorage.removeItem(USER_KEY)
        set({ user: null, token: null, isAuthenticated: false, isLoading: false })
        return false
    },

    clearError: () => set({ error: null }),
}))

/**
 * Helper: Retourne les headers d'authentification pour les appels API.
 */
export function getAuthHeaders(): Record<string, string> {
    const token = localStorage.getItem(TOKEN_KEY)
    return token ? { 'Authorization': `Bearer ${token}` } : {}
}
