import { useCallback, useEffect, useMemo, useState } from 'react'
import {
    Bot,
    CheckCircle2,
    Hammer,
    RefreshCw,
    Rocket,
    Search,
    Wrench,
} from 'lucide-react'

import {
    type BuilderBuildRequest,
    type BuilderHealth,
    type BuilderHistoryEntry,
    type BuilderPublicApiEntry,
    getBuilderHealth,
    getBuilderHistory,
    searchBuilderPublicApis,
} from '../services/api'

const INITIAL_BUILD_FORM: BuilderBuildRequest = {
    prompt: "Construis un micro-SaaS d'analyse et de veille avec API, tableau de bord et documentation.",
    filename: 'app.py',
    language: 'python',
    auto_validate: true,
    use_public_api_catalog: true,
    api_context_query: 'analytics dashboard finance automation api',
    api_context_limit: 5,
}

const INITIAL_HEALTH: BuilderHealth = {
    status: 'offline',
    service: 'builder',
    active_pipelines: 0,
    builds_completed: 0,
    forge_runs: 0,
    public_api_entries: 0,
    mutation_enabled: false,
    deploy_enabled: false,
}

const EXECUTION_LOCKED = true

function StatusPill({ active, label }: { active: boolean; label: string }) {
    return (
        <span
            className={`px-2 py-1 text-[8px] uppercase tracking-[0.2em] border ${
                active
                    ? 'border-matrix/30 bg-matrix/10 text-matrix'
                    : 'border-white/10 bg-white/[0.02] text-white/25'
            }`}
        >
            {label}
        </span>
    )
}

function ResultBadge({ status }: { status?: string }) {
    const normalized = (status || 'unknown').toLowerCase()
    const config: Record<string, string> = {
        success: 'border-matrix/30 bg-matrix/10 text-matrix',
        warning: 'border-cyber-amber/30 bg-cyber-amber/10 text-cyber-amber',
        dry_run: 'border-cyber-cyan/30 bg-cyber-cyan/10 text-cyber-cyan',
        disabled: 'border-white/10 bg-white/[0.02] text-white/30',
        skipped: 'border-white/10 bg-white/[0.02] text-white/30',
        failed: 'border-cyber-pink/30 bg-cyber-pink/10 text-cyber-pink',
        error: 'border-cyber-pink/30 bg-cyber-pink/10 text-cyber-pink',
    }

    return (
        <span className={`px-2 py-1 text-[8px] uppercase tracking-[0.2em] border ${config[normalized] || config.disabled}`}>
            {normalized.replace('_', ' ')}
        </span>
    )
}

export default function BuilderWorkbench() {
    const [health, setHealth] = useState<BuilderHealth>(INITIAL_HEALTH)
    const [history, setHistory] = useState<BuilderHistoryEntry[]>([])
    const [buildForm, setBuildForm] = useState<BuilderBuildRequest>(INITIAL_BUILD_FORM)
    const [deployService, setDeployService] = useState('')
    const [deployTarget, setDeployTarget] = useState<'local' | 'proxmox'>('proxmox')
    const [deployForceRebuild, setDeployForceRebuild] = useState(false)
    const [deployDryRun, setDeployDryRun] = useState(true)
    const [mutationDryRun, setMutationDryRun] = useState(true)
    const [composeFile, setComposeFile] = useState('')
    const [apiQuery, setApiQuery] = useState(INITIAL_BUILD_FORM.api_context_query || '')
    const [apiResults, setApiResults] = useState<BuilderPublicApiEntry[]>([])
    const [loadingAction, setLoadingAction] = useState<string | null>(null)
    const [feedback, setFeedback] = useState<string | null>(null)

    const refreshBuilderState = useCallback(async () => {
        const [builderHealth, historyPayload] = await Promise.all([
            getBuilderHealth(),
            getBuilderHistory(),
        ])
        setHealth(builderHealth)
        setHistory(historyPayload.history.slice().reverse().slice(0, 8))
    }, [])

    useEffect(() => {
        refreshBuilderState()
        const interval = setInterval(refreshBuilderState, 10000)
        return () => clearInterval(interval)
    }, [refreshBuilderState])

    const latestActions = useMemo(() => history.slice(0, 5), [history])

    const updateBuildForm = <K extends keyof BuilderBuildRequest>(key: K, value: BuilderBuildRequest[K]) => {
        setBuildForm((current) => ({ ...current, [key]: value }))
    }

    const handleSearchApis = async () => {
        const query = apiQuery.trim() || buildForm.api_context_query?.trim() || buildForm.prompt
        setLoadingAction('search')
        const result = await searchBuilderPublicApis(query, buildForm.api_context_limit || 5)
        setApiResults(result.results)
        setFeedback(result.total > 0 ? `${result.total} API(s) remontee(s) pour "${query}".` : 'Aucune API pertinente trouvee.')
        setLoadingAction(null)
    }

    return (
        <section className="cyber-panel hud-corners p-4 lg:p-5 space-y-4">
            <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
                <div>
                    <div className="text-[9px] uppercase tracking-[0.2em] text-cyber-cyan/50">Pilotage Builder</div>
                    <h3 className="font-display text-lg font-black tracking-[0.08em] text-white/80 mt-1">Flux Builder en observation</h3>
                    <p className="text-[10px] text-white/25 mt-2 max-w-3xl">
                        Cette vue reste volontairement en observation pendant le run trading. Les briefs, cibles et options restent visibles pour preparer le prochain cycle sans lancer de mutation.
                    </p>
                </div>
                <div className="flex flex-wrap gap-2">
                    <StatusPill active={health.status === 'ok'} label="builder online" />
                    <StatusPill active={false} label="execution verrouillee" />
                    <StatusPill active={false} label="dry-run uniquement" />
                </div>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <MetricCard label="Builds" value={String(health.builds_completed)} tone="matrix" />
                <MetricCard label="Forge Runs" value={String(health.forge_runs)} tone="cyber-cyan" />
                <MetricCard label="API Cache" value={String(health.public_api_entries)} tone="cyber-amber" />
                <MetricCard label="Pipelines" value={String(health.active_pipelines)} tone="cyber-pink" />
            </div>

            <div className="grid grid-cols-1 xl:grid-cols-[1.05fr_1.35fr] gap-4">
                <div className="space-y-4">
                    <div className="border border-white/[0.05] bg-black/30 p-4 space-y-3">
                        <div className="flex items-center justify-between gap-3">
                            <div className="flex items-center gap-2">
                                <Search size={14} className="text-cyber-cyan" />
                                <span className="text-[10px] uppercase tracking-[0.2em] text-cyber-cyan/60">Veille API</span>
                            </div>
                            <button
                                type="button"
                                disabled={true}
                                className="cyber-btn text-[9px] px-3 py-1.5 opacity-40 cursor-not-allowed"
                            >
                                <RefreshCw size={11} />
                                <span>Sync verrouille</span>
                            </button>
                        </div>

                        <div>
                            <label className="text-[8px] text-white/20 uppercase tracking-[0.2em] block mb-2">Recherche API</label>
                            <input
                                value={apiQuery}
                                onChange={(event) => setApiQuery(event.target.value)}
                                placeholder="finance, pricing, forex, analytics..."
                                className="w-full bg-black/50 border border-white/10 px-3 py-2 text-[12px] text-white/80 outline-none focus:border-cyber-cyan/40"
                            />
                        </div>

                        <button
                            onClick={handleSearchApis}
                            disabled={loadingAction !== null}
                            className="cyber-btn text-[9px] px-3 py-2 w-full disabled:opacity-40"
                        >
                            <Search size={11} className={loadingAction === 'search' ? 'animate-pulse' : ''} />
                            <span>Analyser le catalogue</span>
                        </button>

                        <div className="space-y-2 max-h-[320px] overflow-y-auto custom-scrollbar pr-1">
                            {apiResults.length === 0 ? (
                                <div className="text-[10px] text-white/20 border border-dashed border-white/10 p-3">
                                    Aucune suggestion chargee pour l'instant.
                                </div>
                            ) : (
                                apiResults.map((entry) => (
                                    <a
                                        key={`${entry.category}-${entry.name}`}
                                        href={entry.url}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="block border border-white/[0.05] bg-white/[0.02] p-3 hover:border-cyber-cyan/20 transition-colors"
                                    >
                                        <div className="flex items-center justify-between gap-3">
                                            <div className="text-[11px] font-bold text-white/75">{entry.name}</div>
                                            <span className="text-[8px] uppercase tracking-[0.2em] text-cyber-cyan/50">{entry.category}</span>
                                        </div>
                                        <div className="text-[10px] text-white/30 mt-2 leading-relaxed">{entry.description}</div>
                                        <div className="flex flex-wrap gap-2 mt-3 text-[8px] uppercase tracking-[0.15em]">
                                            <span className="text-white/25">auth {entry.auth || 'none'}</span>
                                            <span className={entry.https ? 'text-matrix/60' : 'text-cyber-pink/50'}>
                                                {entry.https ? 'https' : 'http'}
                                            </span>
                                            <span className="text-white/20">cors {entry.cors}</span>
                                        </div>
                                    </a>
                                ))
                            )}
                        </div>
                    </div>

                    <div className="border border-white/[0.05] bg-black/30 p-4 space-y-3">
                        <div className="flex items-center gap-2">
                            <Bot size={14} className="text-matrix" />
                            <span className="text-[10px] uppercase tracking-[0.2em] text-matrix/50">Journal Builder</span>
                        </div>
                        <div className="space-y-2">
                            {latestActions.length === 0 ? (
                                <div className="text-[10px] text-white/20 border border-dashed border-white/10 p-3">
                                    Aucun evenement Builder remonte pour l'instant.
                                </div>
                            ) : (
                                latestActions.map((entry, index) => (
                                    <div key={`${entry.timestamp}-${index}`} className="border border-white/[0.04] bg-white/[0.01] p-3">
                                        <div className="flex items-center justify-between gap-3">
                                            <div className="text-[10px] font-bold text-white/65 uppercase tracking-[0.15em]">{entry.action}</div>
                                            <ResultBadge status={entry.status} />
                                        </div>
                                        <div className="text-[10px] text-white/25 mt-2">{entry.details}</div>
                                        <div className="text-[8px] text-white/15 mt-2 tracking-[0.15em] uppercase">{entry.timestamp}</div>
                                    </div>
                                ))
                            )}
                        </div>
                    </div>
                </div>

                <div className="border border-white/[0.05] bg-black/30 p-4 space-y-4">
                    <div className="flex items-center gap-2">
                        <Hammer size={14} className="text-cyber-amber" />
                        <span className="text-[10px] uppercase tracking-[0.2em] text-cyber-amber/50">Flux Builder</span>
                    </div>

                    <div className="border border-cyber-amber/20 bg-cyber-amber/5 p-3 text-[10px] text-cyber-amber/80 leading-relaxed">
                        Les builds, deploiements et mutations restent verrouilles pendant le run actif. Cet ecran sert uniquement a preparer le brief, la cible et le contexte API du prochain passage.
                    </div>

                    <div>
                        <label className="text-[8px] text-white/20 uppercase tracking-[0.2em] block mb-2">Brief produit</label>
                        <textarea
                            value={buildForm.prompt}
                            onChange={(event) => updateBuildForm('prompt', event.target.value)}
                            rows={5}
                            className="w-full bg-black/50 border border-white/10 px-3 py-3 text-[12px] text-white/80 outline-none focus:border-cyber-amber/40 resize-none"
                        />
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                        <InputField
                            label="Fichier"
                            value={buildForm.filename}
                            onChange={(value) => updateBuildForm('filename', value)}
                            placeholder="app.py"
                        />
                        <InputField
                            label="Langage"
                            value={buildForm.language}
                            onChange={(value) => updateBuildForm('language', value)}
                            placeholder="python"
                        />
                        <InputField
                            label="Query API"
                            value={buildForm.api_context_query || ''}
                            onChange={(value) => updateBuildForm('api_context_query', value)}
                            placeholder="market data analytics"
                        />
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                        <InputField
                            label="Service deploy"
                            value={deployService}
                            onChange={setDeployService}
                            placeholder="builder, muse-nexus, swarm..."
                        />
                        <SelectField
                            label="Cible"
                            value={deployTarget}
                            onChange={(value) => setDeployTarget(value as 'local' | 'proxmox')}
                            options={[
                                { value: 'proxmox', label: 'proxmox' },
                                { value: 'local', label: 'local' },
                            ]}
                        />
                        <InputField
                            label="Compose"
                            value={composeFile}
                            onChange={setComposeFile}
                            placeholder="docker-compose.yml"
                        />
                    </div>

                    <div className="grid grid-cols-2 lg:grid-cols-5 gap-2">
                        <ToggleChip
                            label="Validation auto"
                            active={buildForm.auto_validate}
                            onClick={() => updateBuildForm('auto_validate', !buildForm.auto_validate)}
                        />
                        <ToggleChip
                            label="Catalogue API"
                            active={buildForm.use_public_api_catalog}
                            onClick={() => updateBuildForm('use_public_api_catalog', !buildForm.use_public_api_catalog)}
                        />
                        <ToggleChip
                            label="Deploy dry-run"
                            active={deployDryRun}
                            onClick={() => setDeployDryRun((value) => !value)}
                        />
                        <ToggleChip
                            label="Mutation dry-run"
                            active={mutationDryRun}
                            onClick={() => setMutationDryRun((value) => !value)}
                        />
                        <ToggleChip
                            label="Force rebuild"
                            active={deployForceRebuild}
                            onClick={() => setDeployForceRebuild((value) => !value)}
                        />
                    </div>

                    <div className="flex flex-col sm:flex-row gap-3">
                        <button
                            type="button"
                            disabled={EXECUTION_LOCKED}
                            className="cyber-btn text-[10px] px-4 py-2 flex-1 opacity-40 cursor-not-allowed"
                        >
                            <Wrench size={12} />
                            <span>Build verrouille</span>
                        </button>
                        <button
                            type="button"
                            disabled={EXECUTION_LOCKED}
                            className="cyber-btn text-[10px] px-4 py-2 flex-1 opacity-40 cursor-not-allowed"
                        >
                            <Rocket size={12} />
                            <span>Flux verrouille</span>
                        </button>
                    </div>

                    {feedback && (
                        <div className="flex items-start gap-2 border border-cyber-cyan/20 bg-cyber-cyan/5 p-3 text-[10px] text-cyber-cyan/80">
                            <CheckCircle2 size={14} className="shrink-0 mt-0.5" />
                            <span>{feedback}</span>
                        </div>
                    )}

                </div>
            </div>
        </section>
    )
}

function MetricCard({ label, value, tone }: { label: string; value: string; tone: string }) {
    const toneMap: Record<string, string> = {
        matrix: 'text-matrix',
        'cyber-cyan': 'text-cyber-cyan',
        'cyber-amber': 'text-cyber-amber',
        'cyber-pink': 'text-cyber-pink',
    }

    return (
        <div className="border border-white/[0.05] bg-black/25 p-3">
            <div className="text-[8px] text-white/20 uppercase tracking-[0.2em]">{label}</div>
            <div className={`text-lg font-black mt-2 ${toneMap[tone] || 'text-white/70'}`}>{value}</div>
        </div>
    )
}

function InputField({
    label,
    value,
    onChange,
    placeholder,
}: {
    label: string
    value: string
    onChange: (value: string) => void
    placeholder: string
}) {
    return (
        <div>
            <label className="text-[8px] text-white/20 uppercase tracking-[0.2em] block mb-2">{label}</label>
            <input
                value={value}
                onChange={(event) => onChange(event.target.value)}
                placeholder={placeholder}
                className="w-full bg-black/50 border border-white/10 px-3 py-2 text-[12px] text-white/80 outline-none focus:border-cyber-cyan/40"
            />
        </div>
    )
}

function SelectField({
    label,
    value,
    onChange,
    options,
}: {
    label: string
    value: string
    onChange: (value: string) => void
    options: Array<{ value: string; label: string }>
}) {
    return (
        <div>
            <label className="text-[8px] text-white/20 uppercase tracking-[0.2em] block mb-2">{label}</label>
            <select
                value={value}
                onChange={(event) => onChange(event.target.value)}
                className="w-full bg-black/50 border border-white/10 px-3 py-2 text-[12px] text-white/80 outline-none focus:border-cyber-cyan/40"
            >
                {options.map((option) => (
                    <option key={option.value} value={option.value}>
                        {option.label}
                    </option>
                ))}
            </select>
        </div>
    )
}

function ToggleChip({
    label,
    active,
    onClick,
}: {
    label: string
    active: boolean
    onClick: () => void
}) {
    return (
        <button
            type="button"
            onClick={onClick}
            className={`border px-3 py-2 text-[9px] uppercase tracking-[0.15em] transition-colors ${
                active
                    ? 'border-matrix/30 bg-matrix/10 text-matrix'
                    : 'border-white/10 bg-black/30 text-white/30'
            }`}
        >
            {label}
        </button>
    )
}
