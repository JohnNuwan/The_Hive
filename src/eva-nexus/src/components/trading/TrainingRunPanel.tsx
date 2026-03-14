import type { TrainingRunStatus } from '../../services/api'
import { MetricPill, PanelShell, StatusBadge, compactList, formatDateLabel, formatElapsed } from './TradingShared'

function dependencyTone(state: string) {
    if (state === 'online' || state === 'running') return 'emerald'
    if (state === 'stopped_for_training') return 'amber'
    if (state === 'offline') return 'rose'
    return 'slate'
}

function dependencyLabel(state: string) {
    if (state === 'stopped_for_training') return 'arrete pour entrainement'
    if (!state) return 'inconnu'
    return state
}

export default function TrainingRunPanel({ trainingStatus }: { trainingStatus: TrainingRunStatus | null }) {
    const run = trainingStatus?.run
    const universe = trainingStatus?.universe
    const dependencies = trainingStatus?.dependencies || {}
    const logs = trainingStatus?.logs || []
    const currentStep = run?.current_step || null
    const families = Object.entries(universe?.family_counts || {}).filter(([, value]) => Number(value) > 0)
    const timeframes = Object.entries(universe?.timeframe_counts || {}).filter(([, value]) => Number(value) > 0)

    return (
        <PanelShell
            title="Run d entrainement"
            subtitle="Lecture seule | progression, dependances et univers"
            accent="violet"
            aside={<StatusBadge label={String(run?.status || 'idle').toUpperCase()} tone={run?.active ? 'violet' : 'slate'} />}
        >
            <div className="grid grid-cols-2 gap-3 mb-4">
                <MetricPill label="Strategie" value={String(run?.strategy || 'idle').toUpperCase()} />
                <MetricPill label="Trigger" value={String(run?.trigger || '--')} />
                <MetricPill label="Duree" value={formatElapsed(run?.started_at || null, run?.finished_at || null)} />
                <MetricPill label="Mise a jour" value={formatDateLabel(run?.updated_at || null)} />
            </div>

            <div className="rounded-2xl border border-white/5 bg-black/20 p-4 mb-4">
                <div className="text-[9px] text-slate-500 uppercase font-black tracking-[0.2em]">Etape active</div>
                <div className="mt-2 text-sm font-black text-white/90">{run?.step_label || 'Aucun run actif'}</div>
                <div className="mt-3 grid grid-cols-2 gap-2">
                    <MetricPill label="Symbole" value={String(currentStep?.symbol || '--')} />
                    <MetricPill label="Progression symboles" value={currentStep?.symbol_total ? `${currentStep.symbol_index || 0}/${currentStep.symbol_total}` : '--'} />
                    <MetricPill label="Parties" value={currentStep?.part_total ? `${currentStep.part_index || 0}/${currentStep.part_total}` : '--'} />
                    <MetricPill label="Epochs" value={currentStep?.epoch_total ? `${currentStep.epoch_current || 0}/${currentStep.epoch_total}` : '--'} />
                    <MetricPill label="Steps" value={currentStep?.training_step_total ? `${currentStep.training_step_current || 0}/${currentStep.training_step_total}` : '--'} />
                    <MetricPill label="Skip cron" value={String(run?.skip_reason || 'aucun')} />
                </div>
            </div>

            <div className="mb-4">
                <div className="text-[9px] text-slate-500 uppercase font-black tracking-[0.2em] mb-2">Dependances</div>
                <div className="flex flex-wrap gap-2">
                    {['trainer', 'vllm', 'redis', 'neo4j', 'mosquitto'].map((name) => (
                        <StatusBadge
                            key={name}
                            label={`${name}: ${dependencyLabel(String(dependencies[name]?.state || 'indisponible'))}`}
                            tone={dependencyTone(String(dependencies[name]?.state || 'offline')) as any}
                        />
                    ))}
                </div>
            </div>

            <div className="rounded-2xl border border-white/5 bg-black/20 p-4 mb-4">
                <div className="text-[9px] text-slate-500 uppercase font-black tracking-[0.2em]">Diversite d univers</div>
                <div className="mt-3 grid grid-cols-2 gap-2">
                    {families.length === 0 ? (
                        <MetricPill label="Training" value="Indisponible" />
                    ) : families.map(([family, count]) => <MetricPill key={family} label={family} value={String(count)} />)}
                </div>
                <div className="mt-3 grid grid-cols-3 gap-2">
                    {timeframes.length === 0 ? (
                        <MetricPill label="Timeframes" value="--" />
                    ) : timeframes.map(([timeframe, count]) => <MetricPill key={timeframe} label={timeframe} value={String(count)} />)}
                </div>
                <div className="mt-3 text-[10px] text-slate-400">Echantillon: {compactList(universe?.sample_symbols || [], 8)}</div>
            </div>

            <div>
                <div className="text-[9px] text-slate-500 uppercase font-black tracking-[0.2em] mb-2">Logs recents</div>
                <div className="space-y-2 max-h-52 overflow-y-auto pr-2 custom-scrollbar">
                    {logs.length === 0 ? (
                        <div className="rounded-xl border border-white/5 bg-white/[0.02] px-3 py-2 text-[10px] font-mono text-slate-400">
                            Aucun run actif ou donnees indisponibles.
                        </div>
                    ) : logs.slice().reverse().map((line) => (
                        <div key={line} className="rounded-xl border border-white/5 bg-white/[0.02] px-3 py-2 text-[10px] font-mono text-slate-300 break-words">
                            {line}
                        </div>
                    ))}
                </div>
            </div>
        </PanelShell>
    )
}
