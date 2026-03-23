export type HiveTabId =
    | 'dashboard'
    | 'chat'
    | 'trading'
    | 'graph'
    | 'memory'
    | 'monitoring'
    | 'osint'
    | 'factories'
    | 'admin'
    | 'settings'
    | 'muse'
    | 'knowledge'
    | 'agentfeed'

export type HiveNavigationDetail = {
    tab: HiveTabId
}

const HIVE_NAVIGATE_EVENT = 'hive:navigate'

export function navigateToHiveTab(tab: HiveTabId): void {
    window.dispatchEvent(new CustomEvent<HiveNavigationDetail>(HIVE_NAVIGATE_EVENT, {
        detail: { tab },
    }))
}

export function onHiveNavigate(listener: (detail: HiveNavigationDetail) => void): () => void {
    const handler = (event: Event) => {
        const customEvent = event as CustomEvent<HiveNavigationDetail>
        if (!customEvent.detail?.tab) {
            return
        }
        listener(customEvent.detail)
    }

    window.addEventListener(HIVE_NAVIGATE_EVENT, handler)
    return () => window.removeEventListener(HIVE_NAVIGATE_EVENT, handler)
}
