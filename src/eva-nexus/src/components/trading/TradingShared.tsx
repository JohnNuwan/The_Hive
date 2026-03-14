import type { ReactNode } from 'react'

export type AccentTone = 'cyan' | 'emerald' | 'amber' | 'sky' | 'violet' | 'rose' | 'slate'

function accentMap(accent: AccentTone) {
    switch (accent) {
        case 'emerald':
            return {
                border: 'border-emerald-500/20',
                title: 'text-emerald-300',
                glow: 'bg-emerald-500/10',
                pill: 'text-emerald-400 border-emerald-500/20 bg-emerald-500/10',
            }
        case 'amber':
            return {
                border: 'border-amber-500/20',
                title: 'text-amber-300',
                glow: 'bg-amber-500/10',
                pill: 'text-amber-400 border-amber-500/20 bg-amber-500/10',
            }
        case 'sky':
            return {
                border: 'border-sky-500/20',
                title: 'text-sky-300',
                glow: 'bg-sky-500/10',
                pill: 'text-sky-400 border-sky-500/20 bg-sky-500/10',
            }
        case 'violet':
            return {
                border: 'border-violet-500/20',
                title: 'text-violet-300',
                glow: 'bg-violet-500/10',
                pill: 'text-violet-400 border-violet-500/20 bg-violet-500/10',
            }
        case 'rose':
            return {
                border: 'border-rose-500/20',
                title: 'text-rose-300',
                glow: 'bg-rose-500/10',
                pill: 'text-rose-400 border-rose-500/20 bg-rose-500/10',
            }
        case 'slate':
            return {
                border: 'border-white/10',
                title: 'text-slate-300',
                glow: 'bg-white/[0.05]',
                pill: 'text-slate-300 border-white/10 bg-white/[0.04]',
            }
        default:
            return {
                border: 'border-cyan-500/20',
                title: 'text-cyan-300',
                glow: 'bg-cyan-500/10',
                pill: 'text-cyan-400 border-cyan-500/20 bg-cyan-500/10',
            }
    }
}

export function PanelShell({
    title,
    subtitle,
    accent = 'cyan',
    aside,
    children,
}: {
    title: string
    subtitle?: string
    accent?: AccentTone
    aside?: ReactNode
    children: ReactNode
}) {
    const styles = accentMap(accent)
    return (
        <div className={`glass rounded-[2rem] p-6 border ${styles.border} shadow-xl relative overflow-hidden`}>
            <div className={`absolute top-0 right-0 h-24 w-24 ${styles.glow} blur-3xl rounded-full`} />
            <div className="relative z-10 flex items-start justify-between gap-4 mb-5">
                <div>
                    <h4 className={`text-[10px] font-black uppercase tracking-[0.3em] ${styles.title}`}>{title}</h4>
                    {subtitle ? (
                        <p className="mt-2 text-[10px] text-slate-500 uppercase font-bold tracking-[0.18em]">{subtitle}</p>
                    ) : null}
                </div>
                {aside ? <div className="shrink-0">{aside}</div> : null}
            </div>
            <div className="relative z-10">{children}</div>
        </div>
    )
}

export function StatusBadge({ label, tone = 'slate' }: { label: string; tone?: AccentTone }) {
    const styles = accentMap(tone)
    return (
        <span className={`inline-flex items-center rounded-xl border px-3 py-1 text-[9px] font-black uppercase tracking-[0.2em] ${styles.pill}`}>
            {label}
        </span>
    )
}

export function MetricTile({
    label,
    value,
    meta,
    accent = 'slate',
}: {
    label: string
    value: string
    meta?: string
    accent?: AccentTone
}) {
    const styles = accentMap(accent)
    return (
        <div className={`rounded-2xl border p-4 ${styles.border} bg-black/20`}>
            <div className="text-[9px] font-black uppercase tracking-[0.2em] text-slate-500">{label}</div>
            <div className={`mt-2 text-xl font-black tracking-tight ${styles.title}`}>{value}</div>
            {meta ? <div className="mt-2 text-[10px] text-slate-500">{meta}</div> : null}
        </div>
    )
}

export function MetricPill({ label, value }: { label: string; value: string }) {
    return (
        <div className="rounded-xl border border-white/5 bg-white/[0.02] px-3 py-2">
            <div className="text-[8px] text-slate-500 uppercase font-black tracking-[0.18em]">{label}</div>
            <div className="mt-1 text-[11px] font-black text-white/80 truncate">{value}</div>
        </div>
    )
}

export function formatUsd(value: number) {
    const amount = Number(value || 0)
    const prefix = amount > 0 ? '+' : ''
    return `${prefix}${amount.toFixed(2)} $`
}

export function formatPercent(value: number, digits = 2) {
    return `${Number(value || 0).toFixed(digits)}%`
}

export function formatElapsed(startedAt?: string | null, finishedAt?: string | null) {
    if (!startedAt) return '--'
    const start = new Date(startedAt)
    if (Number.isNaN(start.getTime())) return '--'
    const end = finishedAt ? new Date(finishedAt) : new Date()
    const durationMs = Math.max(end.getTime() - start.getTime(), 0)
    const minutes = Math.floor(durationMs / 60000)
    const hours = Math.floor(minutes / 60)
    const remainingMinutes = minutes % 60
    if (hours > 0) return `${hours}h${String(remainingMinutes).padStart(2, '0')}`
    return `${remainingMinutes}m`
}

export function formatDateLabel(value?: string | null) {
    if (!value) return '--'
    const date = new Date(value)
    if (Number.isNaN(date.getTime())) return '--'
    return date.toLocaleString('fr-FR', {
        day: '2-digit',
        month: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
    })
}

export function compactList(values: string[], limit = 6) {
    if (!values.length) return '--'
    return values.slice(0, limit).join(', ')
}
