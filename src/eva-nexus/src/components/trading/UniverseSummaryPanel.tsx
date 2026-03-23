import type { LabChampionStatus, TradingStatusResponse, TrainingRunStatus } from '../../services/api'
import { MetricPill, PanelShell, compactList } from './TradingShared'

function collectLiveUniverseSamples(championStatus: LabChampionStatus | null, tradingData: TradingStatusResponse | null) {
    const liveSymbols = Array.isArray(tradingData?.universe?.lab_live?.symbols)
        ? (tradingData?.universe?.lab_live?.symbols as string[])
        : []
    if (liveSymbols.length > 0) {
        return liveSymbols
    }
    const symbols = new Set<string>()
    Object.values(championStatus?.horizons || {}).forEach((status) => {
        const horizonSymbols = status?.live_universe?.symbols || []
        horizonSymbols.forEach((symbol) => symbols.add(symbol))
    })
    return Array.from(symbols)
}

function pickCfdSamples(values: string[]) {
    return values.filter((value) => value.includes('.cash') || value.includes('US30') || value.includes('US500') || value.includes('GER40'))
}

export default function UniverseSummaryPanel({
    trainingStatus,
    championStatus,
    tradingData,
}: {
    trainingStatus: TrainingRunStatus | null
    championStatus: LabChampionStatus | null
    tradingData: TradingStatusResponse | null
}) {
    const universe = trainingStatus?.universe
    const liveSymbols = collectLiveUniverseSamples(championStatus, tradingData)
    const cfdSamples = pickCfdSamples(universe?.sample_symbols || [])
    const liveMeta = tradingData?.universe?.lab_live

    return (
        <PanelShell title="Univers" subtitle="Univers d entrainement vs univers live" accent="amber">
            <div className="grid grid-cols-2 gap-3 mb-4">
                <MetricPill label="Training" value={String(universe?.total_symbols || 0)} />
                <MetricPill label="Live" value={liveMeta?.count ? String(liveMeta.count) : String(liveSymbols.length)} />
                <MetricPill label="Source live" value={String(liveMeta?.source || 'indisponible')} />
                <MetricPill label="Restriction" value={liveMeta?.restricted ? 'oui' : 'non'} />
            </div>

            <div className="rounded-2xl border border-white/5 bg-black/20 p-4 mb-4">
                <div className="text-[9px] text-slate-500 uppercase font-black tracking-[0.2em]">Familles training</div>
                <div className="mt-3 grid grid-cols-2 gap-2">
                    {Object.entries(universe?.family_counts || {}).map(([family, count]) => (
                        <MetricPill key={family} label={family} value={String(count)} />
                    ))}
                </div>
            </div>

            <div className="rounded-2xl border border-white/5 bg-black/20 p-4 mb-4">
                <div className="text-[9px] text-slate-500 uppercase font-black tracking-[0.2em]">CFD visibles</div>
                <div className="mt-3 text-[10px] text-slate-300">{compactList(cfdSamples, 8)}</div>
            </div>

            <div className="rounded-2xl border border-white/5 bg-black/20 p-4">
                <div className="text-[9px] text-slate-500 uppercase font-black tracking-[0.2em]">Symboles live</div>
                <div className="mt-3 text-[10px] text-slate-300">{compactList(liveSymbols, 10)}</div>
            </div>
        </PanelShell>
    )
}
