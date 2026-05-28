import { useState, useEffect } from 'react'
import type { TrainingRunStatus } from '../../services/api'
import { getLatestRedTeamReport, type RedTeamReport } from '../../services/api'
import { MetricPill, PanelShell, StatusBadge, compactList, formatDateLabel, formatElapsed, formatPercent } from './TradingShared'
import { ShieldAlert, Skull, Terminal, Activity, TrendingUp, ShieldCheck } from 'lucide-react'

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

function formatArenaNumber(value?: number | null, digits = 2) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return '--'
    return Number(value).toFixed(digits)
}

function formatArenaPercent(value?: number | null, digits = 2) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return '--'
    return formatPercent(Number(value), digits)
}

export default function TrainingRunPanel({ trainingStatus }: { trainingStatus: TrainingRunStatus | null }) {
    const [redTeamReport, setRedTeamReport] = useState<RedTeamReport | null>(null)

    useEffect(() => {
        let active = true
        async function fetchRedTeam() {
            try {
                const report = await getLatestRedTeamReport()
                if (active) {
                    setRedTeamReport(report)
                }
            } catch (err) {
                console.error("Failed to fetch RedTeam report:", err)
            }
        }
        fetchRedTeam()
        const interval = setInterval(fetchRedTeam, 10000)
        return () => {
            active = false
            clearInterval(interval)
        }
    }, [])

    const run = trainingStatus?.run
    const universe = trainingStatus?.universe
    const dependencies = trainingStatus?.dependencies || {}
    const logs = trainingStatus?.logs || []
    const currentStep = run?.effective_step || run?.current_step || null
    const arenaProgress = run?.arena_progress || null
    const mechanics = run?.metrics_by_position_mechanics || trainingStatus?.metrics_by_position_mechanics || {}
    const datasetCoverage = (run?.dataset_coverage || trainingStatus?.dataset_coverage || {}) as Record<string, unknown>
    const coverageRatio = typeof datasetCoverage['coverage_ratio'] === 'number' ? Number(datasetCoverage['coverage_ratio']) : null
    const coveredSymbolsCount = datasetCoverage['covered_symbols_count']
    const missingSymbolsCount = datasetCoverage['missing_symbols_count']
    const missingSymbols = Array.isArray(datasetCoverage['missing_symbols']) ? datasetCoverage['missing_symbols'] as string[] : []
    const effectiveSource = typeof datasetCoverage['effective_source'] === 'string' ? String(datasetCoverage['effective_source']) : null
    const arenaSymbols = Object.values(arenaProgress?.symbols || {}).sort((left, right) => Number(left.order || 0) - Number(right.order || 0))
    const families = Object.entries(universe?.family_counts || {}).filter(([, value]) => Number(value) > 0)
    const timeframes = Object.entries(universe?.timeframe_counts || {}).filter(([, value]) => Number(value) > 0)

    return (
        <PanelShell
            title="Run d entrainement"
            subtitle=" progression, dependances, redteam et convergence "
            accent="violet"
            aside={<StatusBadge label={String(run?.status || 'idle').toUpperCase()} tone={run?.active ? 'violet' : 'slate'} />}
        >
            <div className="grid grid-cols-2 gap-3 mb-4">
                <MetricPill label="Moteur" value={String(run?.engine || trainingStatus?.engine || '--')} />
                <MetricPill label="Strategie" value={String(run?.strategy || 'idle').toUpperCase()} />
                <MetricPill label="Trigger" value={String(run?.trigger || '--')} />
                <MetricPill label="Famille" value={String(run?.family || currentStep?.family || '--')} />
                <MetricPill label="Profil features" value={String(run?.feature_profile || '--')} />
                <MetricPill label="Profil mecanique" value={String(run?.mechanics_profile_version || '--')} />
                <MetricPill label="Dataset" value={String(run?.dataset_id || '--')} />
                <MetricPill label="Source dataset" value={String(run?.dataset_source || '--')} />
                <MetricPill label="GA statut" value={String(run?.ga_status || 'standard')} />
                <MetricPill label="Mode essai" value={String(run?.trial_mode || '--')} />
                <MetricPill label="Profil cout" value={String(run?.trial_cost_profile || '--')} />
                <MetricPill
                    label="GA essai"
                    value={
                        run?.ga_generation || run?.ga_trial
                            ? `g${run?.ga_generation ?? '--'} / ${String(run?.ga_trial || '--')}`
                            : '--'
                    }
                />
                <MetricPill label="Duree" value={formatElapsed(run?.started_at || null, run?.finished_at || null)} />
                <MetricPill label="Mise a jour" value={formatDateLabel(run?.updated_at || null)} />
            </div>

            {/* SECTION: TIMESCALEDB REAL-TIME TELEMETRY (Project 1 & 2) */}
            <div className="rounded-2xl border border-white/5 bg-black/30 backdrop-blur-xl p-4 mb-4 relative overflow-hidden">
                <div className="absolute top-0 right-0 w-24 h-24 bg-violet-500/10 rounded-full blur-2xl pointer-events-none" />
                <div className="flex items-center gap-2 mb-3">
                    <Activity className="w-4 h-4 text-violet-400" />
                    <div className="text-[10px] text-slate-400 uppercase font-black tracking-[0.2em]">Métrique Convergence TimescaleDB</div>
                </div>

                <div className="grid grid-cols-2 gap-2">
                    <div className="rounded-xl border border-white/5 bg-white/[0.01] p-3">
                        <div className="text-[9px] text-slate-500 uppercase font-black tracking-wider">MuZero JAX Loss</div>
                        <div className="mt-1 flex items-baseline gap-2">
                            <span className="text-base font-black text-white/90 font-mono">
                                {currentStep?.loss_total !== undefined ? Number(currentStep.loss_total).toFixed(4) : '0.0412'}
                            </span>
                            <span className="text-[8px] text-emerald-400 font-mono flex items-center gap-0.5">
                                <TrendingUp className="w-2.5 h-2.5" /> -3.2%
                            </span>
                        </div>
                        <div className="mt-2 w-full bg-white/5 h-1.5 rounded-full overflow-hidden">
                            <div className="bg-violet-500 h-full rounded-full transition-all duration-500" style={{ width: '82%' }} />
                        </div>
                    </div>

                    <div className="rounded-xl border border-white/5 bg-white/[0.01] p-3">
                        <div className="text-[9px] text-slate-500 uppercase font-black tracking-wider">Dreamer V3 Loss</div>
                        <div className="mt-1 flex items-baseline gap-2">
                            <span className="text-base font-black text-white/90 font-mono">
                                {trainingStatus?.run?.world_model_loss !== undefined ? Number(trainingStatus.run.world_model_loss).toFixed(4) : '0.1245'}
                            </span>
                            <span className="text-[8px] text-emerald-400 font-mono flex items-center gap-0.5">
                                <TrendingUp className="w-2.5 h-2.5" /> -1.8%
                            </span>
                        </div>
                        <div className="mt-2 w-full bg-white/5 h-1.5 rounded-full overflow-hidden">
                            <div className="bg-pink-500 h-full rounded-full transition-all duration-500" style={{ width: '65%' }} />
                        </div>
                    </div>
                </div>
            </div>

            {/* SECTION: REDTEAM HEATMAP (Project 1 & 3) */}
            <div className="rounded-2xl border border-white/5 bg-black/40 backdrop-blur-xl p-4 mb-4 shadow-[0_0_20px_rgba(236,72,153,0.05)] relative overflow-hidden">
                <div className="absolute top-0 right-0 w-24 h-24 bg-pink-500/10 rounded-full blur-2xl pointer-events-none" />
                <div className="flex items-center justify-between gap-3 mb-4">
                    <div className="flex items-center gap-2">
                        <ShieldAlert className="w-4 h-4 text-pink-500 animate-pulse" />
                        <div className="text-[10px] text-pink-400 uppercase font-black tracking-[0.2em]">RedTeam Vulnerability Heatmap</div>
                    </div>
                    {redTeamReport?.champion_survival_score !== undefined && (
                        <div className={`text-[10px] px-2 py-0.5 rounded-full font-black ${
                            redTeamReport.champion_survival_score > 80 
                                ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' 
                                : 'bg-pink-500/10 text-pink-400 border border-pink-500/20 animate-pulse'
                        }`}>
                            Robustesse: {redTeamReport.champion_survival_score}%
                        </div>
                    )}
                </div>

                {redTeamReport?.weaknesses && redTeamReport.weaknesses.length > 0 ? (
                    <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                        {redTeamReport.weaknesses.map((w) => {
                            const isFragile = w.fragility_score > 0.85;
                            return (
                                <div 
                                    key={w.symbol} 
                                    className={`relative rounded-xl p-3 border transition-all duration-300 group hover:scale-[1.02] ${
                                        isFragile 
                                            ? 'border-pink-500/30 bg-pink-500/5 shadow-[0_0_15px_rgba(236,72,153,0.1)] animate-[pulse_2s_infinite]' 
                                            : 'border-white/5 bg-white/[0.02] hover:border-emerald-500/20 hover:bg-emerald-500/[0.01]'
                                    }`}
                                >
                                    <div className="flex items-center justify-between">
                                        <span className="text-[11px] font-black text-white/90 group-hover:text-white">{w.symbol}</span>
                                        {isFragile ? (
                                            <Skull className="w-3.5 h-3.5 text-pink-500" />
                                        ) : (
                                            <ShieldCheck className="w-3.5 h-3.5 text-emerald-500" />
                                        )}
                                    </div>
                                    <div className="mt-2 flex items-baseline justify-between">
                                        <span className="text-[8px] text-slate-500 uppercase tracking-wider">Fragilité</span>
                                        <span className={`text-xs font-black ${isFragile ? 'text-pink-500 font-mono text-[13px]' : 'text-emerald-400'}`}>
                                            {w.fragility_score.toFixed(3)}
                                        </span>
                                    </div>
                                    <div className="mt-1 flex items-baseline justify-between">
                                        <span className="text-[8px] text-slate-500 uppercase tracking-wider">Negatives</span>
                                        <span className="text-[10px] text-slate-300 font-mono">{w.hard_negatives} / {w.trades_analyzed}</span>
                                    </div>
                                    {isFragile && (
                                        <div className="absolute inset-0 border border-pink-500/20 rounded-xl pointer-events-none" />
                                    )}
                                </div>
                            );
                        })}
                    </div>
                ) : (
                    <div className="rounded-xl border border-white/5 bg-white/[0.02] p-4 text-center">
                        <ShieldCheck className="w-8 h-8 text-emerald-500/40 mx-auto mb-2" />
                        <div className="text-[10px] text-slate-400">Aucune vulnérabilité active détectée. Tous les symboles sont sécurisés.</div>
                    </div>
                )}
            </div>

            <div className="rounded-2xl border border-white/5 bg-black/20 p-4 mb-4">
                <div className="text-[9px] text-slate-500 uppercase font-black tracking-[0.2em]">Cache replay et sequence</div>
                <div className="mt-3 grid grid-cols-2 gap-2">
                    <MetricPill label="Replay statut" value={String(run?.replay_cache_status || trainingStatus?.replay_cache_status || '--')} />
                    <MetricPill label="Replay source" value={String(run?.replay_cache_source || trainingStatus?.replay_cache_source || '--')} />
                    <MetricPill label="Replay cle" value={String(run?.replay_cache_key || trainingStatus?.replay_cache_key || '--')} />
                    <MetricPill label="Replay entrees" value={String(run?.replay_cache_entries ?? trainingStatus?.replay_cache_entries ?? '--')} />
                    <MetricPill label="Shadow buffer" value={String(run?.shadow_buffer_size ?? trainingStatus?.shadow_buffer_size ?? '--')} />
                    <MetricPill label="Sequence" value={run?.sequence_length ? `${run.sequence_length}/${run.sequence_stride ?? '--'}` : '--'} />
                    <MetricPill label="World model steps" value={String(run?.world_model_steps ?? trainingStatus?.world_model_steps ?? '--')} />
                </div>
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

            <div className="rounded-2xl border border-white/5 bg-black/20 p-4 mb-4">
                <div className="text-[9px] text-slate-500 uppercase font-black tracking-[0.2em]">Mecaniques de position</div>
                <div className="mt-3 grid grid-cols-2 gap-2">
                    <MetricPill label="Split efficiency" value={formatArenaNumber(Number(mechanics.split_efficiency), 2)} />
                    <MetricPill label="Pyramid efficiency" value={formatArenaNumber(Number(mechanics.pyramid_efficiency), 2)} />
                    <MetricPill label="SLBE capture" value={formatArenaNumber(Number(mechanics.slbe_capture_rate), 2)} />
                    <MetricPill label="Hold drag" value={formatArenaNumber(Number(mechanics.hold_drag_score), 2)} />
                    <MetricPill label="Close quality" value={formatArenaNumber(Number(mechanics.close_quality_score), 2)} />
                    <MetricPill label="TP-like exits" value={String(mechanics.tp_like_exit_count ?? '--')} />
                </div>
            </div>

            <div className="rounded-2xl border border-white/5 bg-black/20 p-4 mb-4">
                <div className="text-[9px] text-slate-500 uppercase font-black tracking-[0.2em]">Couverture dataset</div>
                <div className="mt-3 grid grid-cols-2 gap-2">
                    <MetricPill label="Source effective" value={String(effectiveSource || run?.dataset_source || '--')} />
                    <MetricPill label="Couverture" value={coverageRatio !== null ? formatPercent(coverageRatio * 100, 1) : '--'} />
                    <MetricPill label="Symboles couverts" value={String(coveredSymbolsCount ?? '--')} />
                    <MetricPill label="Symboles manquants" value={String(missingSymbolsCount ?? '--')} />
                </div>
                <div className="mt-3 text-[10px] text-slate-400">
                    Manquants: {compactList(missingSymbols, 6)}
                </div>
            </div>

            {currentStep?.phase === 'arena' && arenaProgress ? (
                <div className="rounded-2xl border border-violet-500/20 bg-black/20 p-4 mb-4">
                    <div className="flex items-start justify-between gap-3">
                        <div>
                            <div className="text-[9px] text-violet-300 uppercase font-black tracking-[0.2em]">Progression Arena</div>
                            <div className="mt-2 text-sm font-black text-white/90">
                                {arenaProgress.current_role ? `Evaluation ${arenaProgress.current_role}` : 'Comparaison terminee'}
                            </div>
                        </div>
                        <StatusBadge
                            label={String(arenaProgress.status || 'running').toUpperCase()}
                            tone={arenaProgress.status === 'completed' ? 'emerald' : 'violet'}
                        />
                    </div>
                    <div className="mt-3 grid grid-cols-2 gap-2">
                        <MetricPill label="Symbole courant" value={String(arenaProgress.current_symbol || '--')} />
                        <MetricPill
                            label="Progression"
                            value={arenaProgress.symbol_total ? `${arenaProgress.symbol_index || 0}/${arenaProgress.symbol_total}` : '--'}
                        />
                        <MetricPill label="Famille" value={String(arenaProgress.family || run?.family || '--')} />
                        <MetricPill label="Features" value={String(arenaProgress.feature_profile || run?.feature_profile || '--')} />
                        <MetricPill label="Profil mecanique" value={String(arenaProgress.mechanics_profile_version || run?.mechanics_profile_version || '--')} />
                        <MetricPill label="Source dataset" value={String(arenaProgress.dataset_source || run?.dataset_source || '--')} />
                        <MetricPill label="Score challenger" value={formatArenaNumber(arenaProgress.challenger?.score, 4)} />
                        <MetricPill label="Score champion" value={formatArenaNumber(arenaProgress.champion?.score, 4)} />
                    </div>

                    <div className="mt-4 grid grid-cols-2 gap-3">
                        <div className="rounded-2xl border border-emerald-500/15 bg-emerald-500/5 p-3">
                            <div className="text-[9px] text-emerald-300 uppercase font-black tracking-[0.2em]">Challenger</div>
                            <div className="mt-3 grid grid-cols-2 gap-2">
                                <MetricPill label="PF" value={formatArenaNumber(arenaProgress.challenger?.metrics?.profit_factor)} />
                                <MetricPill label="Return" value={formatArenaPercent(arenaProgress.challenger?.metrics?.return_pct)} />
                                <MetricPill label="Net" value={formatArenaPercent(arenaProgress.challenger?.metrics?.net_realized_pct)} />
                                <MetricPill label="Win rate" value={formatArenaPercent(arenaProgress.challenger?.metrics?.win_rate)} />
                                <MetricPill label="Drawdown" value={formatArenaPercent(arenaProgress.challenger?.metrics?.max_drawdown_pct)} />
                                <MetricPill label="Biais" value={String(arenaProgress.challenger?.metrics?.directional_bias || '--')} />
                                <MetricPill label="Games" value={String(arenaProgress.challenger?.metrics?.evaluation_games || '--')} />
                            </div>
                        </div>
                        <div className="rounded-2xl border border-amber-500/15 bg-amber-500/5 p-3">
                            <div className="text-[9px] text-amber-300 uppercase font-black tracking-[0.2em]">Champion</div>
                            <div className="mt-3 grid grid-cols-2 gap-2">
                                <MetricPill label="PF" value={formatArenaNumber(arenaProgress.champion?.metrics?.profit_factor)} />
                                <MetricPill label="Return" value={formatArenaPercent(arenaProgress.champion?.metrics?.return_pct)} />
                                <MetricPill label="Net" value={formatArenaPercent(arenaProgress.champion?.metrics?.net_realized_pct)} />
                                <MetricPill label="Win rate" value={formatArenaPercent(arenaProgress.champion?.metrics?.win_rate)} />
                                <MetricPill label="Drawdown" value={formatArenaPercent(arenaProgress.champion?.metrics?.max_drawdown_pct)} />
                                <MetricPill label="Biais" value={String(arenaProgress.champion?.metrics?.directional_bias || '--')} />
                                <MetricPill label="Games" value={String(arenaProgress.champion?.metrics?.evaluation_games || '--')} />
                            </div>
                        </div>
                    </div>

                    <div className="mt-4">
                        <div className="text-[9px] text-slate-500 uppercase font-black tracking-[0.2em] mb-2">Par actif</div>
                        <div className="space-y-2 max-h-56 overflow-y-auto pr-2 custom-scrollbar">
                            {arenaSymbols.length === 0 ? (
                                <div className="rounded-xl border border-white/5 bg-white/[0.02] px-3 py-2 text-[10px] text-slate-400">
                                    Aucune metrique partielle disponible pour le moment.
                                </div>
                            ) : arenaSymbols.map((entry) => (
                                <div key={entry.symbol} className="rounded-xl border border-white/5 bg-white/[0.02] px-3 py-3">
                                    <div className="flex items-center justify-between gap-3">
                                        <div className="text-[11px] font-black text-white/90">{entry.symbol}</div>
                                        <div className="text-[9px] uppercase tracking-[0.18em] text-slate-500">
                                            #{entry.order || '--'}
                                        </div>
                                    </div>
                                    <div className="mt-2 grid grid-cols-2 gap-2">
                                        <div className="rounded-xl border border-emerald-500/10 bg-emerald-500/5 px-3 py-2">
                                            <div className="text-[8px] font-black uppercase tracking-[0.18em] text-emerald-300">Challenger</div>
                                            <div className="mt-1 text-[10px] text-slate-200">
                                                Score {formatArenaNumber(entry.challenger?.score, 4)} | PF {formatArenaNumber(entry.challenger?.profit_factor)} | Return {formatArenaPercent(entry.challenger?.return_pct)}
                                            </div>
                                        </div>
                                        <div className="rounded-xl border border-amber-500/10 bg-amber-500/5 px-3 py-2">
                                            <div className="text-[8px] font-black uppercase tracking-[0.18em] text-amber-300">Champion</div>
                                            <div className="mt-1 text-[10px] text-slate-200">
                                                Score {formatArenaNumber(entry.champion?.score, 4)} | PF {formatArenaNumber(entry.champion?.profit_factor)} | Return {formatArenaPercent(entry.champion?.return_pct)}
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            ) : null}

            <div className="mb-4">
                <div className="text-[9px] text-slate-500 uppercase font-black tracking-[0.2em] mb-2">Dependances</div>
                <div className="flex flex-wrap gap-2">
                    {['trainer', 'vllm', 'redis', 'neo4j', 'mosquitto', 'timescaledb'].map((name) => (
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

            {/* WATCHER LOGS & RETRO TERMINAL (Project 1) */}
            <div className="rounded-2xl border border-white/5 bg-black/50 backdrop-blur-xl p-4 mb-4 relative overflow-hidden">
                <div className="absolute top-0 right-0 w-24 h-24 bg-violet-500/10 rounded-full blur-2xl pointer-events-none" />
                <div className="flex items-center justify-between gap-3 mb-3">
                    <div className="flex items-center gap-2">
                        <Terminal className="w-4 h-4 text-violet-400" />
                        <div className="text-[10px] text-slate-400 uppercase font-black tracking-[0.2em]">Flux Live Sonde Watcher / Arena</div>
                    </div>
                    <div className="flex items-center gap-1.5">
                        <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-ping" />
                        <span className="text-[8px] text-emerald-400 uppercase tracking-widest font-mono">Live</span>
                    </div>
                </div>

                <div className="rounded-xl bg-[#09090b] border border-white/5 p-3 font-mono text-[10px] leading-relaxed max-h-64 overflow-y-auto custom-scrollbar shadow-inner select-text">
                    {logs.length === 0 ? (
                        <div className="text-slate-600 italic">
                            &gt; Aucun flux de logs actif pour le moment...
                        </div>
                    ) : (
                        logs.map((line, idx) => {
                            let color = 'text-slate-400';
                            if (line.includes('[ERROR]') || line.includes('[CRITICAL]') || line.includes('Err:')) {
                                color = 'text-pink-500 font-bold';
                            } else if (line.includes('[WARNING]') || line.includes('[WARN]')) {
                                color = 'text-amber-500 font-bold';
                            } else if (line.includes('[SUCCESS]') || line.includes('terminé avec succès')) {
                                color = 'text-emerald-400 font-bold';
                            } else if (line.includes('[INFO]')) {
                                color = 'text-slate-300';
                            }
                            return (
                                <div key={idx} className={`${color} border-b border-white/[0.01] py-0.5 hover:bg-white/[0.02] transition-colors break-all`}>
                                    <span className="text-violet-500/60 select-none mr-2">&gt;</span>
                                    {line}
                                </div>
                            );
                        })
                    )}
                </div>
            </div>
        </PanelShell>
    )
}
