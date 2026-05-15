"""
Pipeline d'ingestion des historiques MT5 multi-comptes.

Le pipeline consomme les endpoints Banker deja actifs afin d'eviter de
reprendre la main sur les terminaux MT5 depuis un second processus. Il produit
des positions normalisees, des transitions Shadow Learning et des slices
Nemesis utilisables par les prochaines Arena.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


JsonFetcher = Callable[[str, dict[str, Any] | None], dict[str, Any]]


@dataclass(slots=True)
class BankerHistorySource:
    """
    Decrit une instance Banker a interroger pour l'historique MT5.

    Args:
        name (str): Nom lisible de l'instance.
        base_url (str): URL racine HTTP de l'instance Banker.
        role (str): Role fonctionnel, par exemple ``master`` ou ``follower``.
        login (int | None): Login MT5 associe si connu.
        server (str): Serveur MT5 associe.
        broker (str): Courtier ou famille de compte.
        enabled (bool): Indique si la cible est active dans le routeur.
        symbol_map (dict[str, str]): Mapping symbole canonique vers symbole broker.
        phase (str): Phase du compte, par exemple ``challenge`` ou ``funded``.
    """

    name: str
    base_url: str
    role: str = "follower"
    login: int | None = None
    server: str = ""
    broker: str = ""
    enabled: bool = True
    symbol_map: dict[str, str] = field(default_factory=dict)
    phase: str = ""

    @property
    def account_key(self) -> str:
        """
        Retourne une cle stable pour dedupliquer le compte.

        Returns:
            str: Cle ``server:login`` si possible, sinon URL de fallback.
        """

        if self.login is not None and self.server:
            return f"{self.server}:{self.login}"
        if self.login is not None:
            return str(self.login)
        return self.base_url.rstrip("/")


@dataclass(slots=True)
class IngestionConfig:
    """
    Configure une execution du pipeline d'ingestion.

    Args:
        master_url (str): URL du Banker maitre.
        days (int): Fenetre d'historique a importer.
        output_root (Path): Racine des artefacts d'ingestion.
        shadow_output_dir (Path): Dossier JSONL Shadow Learning.
        state_file (Path): Fichier d'etat des positions deja importees.
        include_disabled (bool): Inclut les targets de copie desactivees.
        force (bool): Reimporte les positions deja vues.
        timeout_seconds (float): Timeout HTTP par requete.
        max_deals_per_account (int): Limite defensive par compte, 0 pour illimite.
    """

    master_url: str = "http://127.0.0.1:8100"
    days: int = 30
    output_root: Path = Path("data/mt5_history_ingestion")
    shadow_output_dir: Path = Path("data/shadow_learning/mt5_fleet")
    state_file: Path = Path("data/mt5_history_ingestion/state.json")
    include_disabled: bool = False
    force: bool = False
    timeout_seconds: float = 8.0
    max_deals_per_account: int = 0


def default_json_fetcher(timeout_seconds: float = 8.0) -> JsonFetcher:
    """
    Construit un fetcher HTTP JSON minimal base sur la bibliotheque standard.

    Args:
        timeout_seconds (float): Timeout HTTP applique a chaque requete.

    Returns:
        JsonFetcher: Fonction de lecture JSON.
    """

    def fetch(url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        query = urlencode({key: value for key, value in (params or {}).items() if value is not None})
        target_url = f"{url}?{query}" if query else url
        request = Request(target_url, headers={"Accept": "application/json"})
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                raw_body = response.read().decode("utf-8")
        except HTTPError as exc:
            raise RuntimeError(f"Requete HTTP refusee par {target_url}: {exc.code}") from exc
        except URLError as exc:
            raise RuntimeError(f"Instance Banker injoignable {target_url}: {exc.reason}") from exc
        return json.loads(raw_body)

    return fetch


def discover_sources(
    master_url: str,
    *,
    include_disabled: bool = False,
    fetcher: JsonFetcher | None = None,
) -> list[BankerHistorySource]:
    """
    Decouvre le master et ses followers depuis les endpoints Banker.

    Args:
        master_url (str): URL du Banker maitre.
        include_disabled (bool): Inclut les cibles desactivees.
        fetcher (JsonFetcher | None): Fetcher substituable pour les tests.

    Returns:
        list[BankerHistorySource]: Sources ordonnees a interroger.
    """

    fetch = fetcher or default_json_fetcher()
    normalized_master_url = master_url.rstrip("/")
    sources = [
        BankerHistorySource(
            name="Master",
            base_url=normalized_master_url,
            role="master",
            enabled=True,
        )
    ]

    try:
        status = fetch(f"{normalized_master_url}/trading/status", None)
    except RuntimeError as exc:
        logger.warning("Statut master indisponible pour la decouverte: %s", exc)
        status = {}

    copy_targets = list(dict(status.get("copy_trading") or {}).get("targets") or [])
    if not copy_targets:
        try:
            copy_status = fetch(f"{normalized_master_url}/copy-trading/status", None)
            copy_targets = list(copy_status.get("targets") or [])
        except RuntimeError as exc:
            logger.warning("Statut copy-trading indisponible pour la decouverte: %s", exc)

    seen = {normalized_master_url}
    for target in copy_targets:
        enabled = bool(target.get("enabled", True))
        if not enabled and not include_disabled:
            continue
        base_url = str(target.get("banker_base_url") or "").strip().rstrip("/")
        if not base_url or base_url in seen:
            continue
        seen.add(base_url)
        sources.append(
            BankerHistorySource(
                name=str(target.get("name") or target.get("terminal_label") or base_url),
                base_url=base_url,
                role="follower",
                login=_safe_int(target.get("login")),
                server=str(target.get("server") or ""),
                broker=str(target.get("broker") or ""),
                enabled=enabled,
                symbol_map=dict(target.get("symbol_map") or {}),
                phase=str(target.get("phase") or ""),
            )
        )

    return sources


def ingest_mt5_fleet_history(
    config: IngestionConfig,
    *,
    fetcher: JsonFetcher | None = None,
    sources: list[BankerHistorySource] | None = None,
) -> dict[str, Any]:
    """
    Execute l'ingestion complete des historiques MT5 de la flotte.

    Args:
        config (IngestionConfig): Configuration de l'execution.
        fetcher (JsonFetcher | None): Fetcher HTTP substituable.
        sources (list[BankerHistorySource] | None): Sources forcees pour tests ou batchs dedies.

    Returns:
        dict[str, Any]: Rapport d'ingestion et chemins des artefacts.
    """

    fetch = fetcher or default_json_fetcher(config.timeout_seconds)
    discovered_sources = sources or discover_sources(
        config.master_url,
        include_disabled=config.include_disabled,
        fetcher=fetch,
    )
    state = _load_state(config.state_file)
    imported_keys = set(state.get("imported_position_keys") or [])

    raw_deals: list[dict[str, Any]] = []
    normalized_positions: list[dict[str, Any]] = []
    open_positions: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for source in discovered_sources:
        if not source.enabled and not config.include_disabled:
            continue
        try:
            payload = fetch(
                f"{source.base_url}/history/deals",
                {
                    "days": max(1, int(config.days)),
                    "closed_only": "false",
                    "limit": max(0, int(config.max_deals_per_account)),
                },
            )
            deals = [dict(item) for item in payload.get("deals") or []]
            account_payload = dict(payload.get("account") or {})
            source = _merge_source_account(source, account_payload)
            raw_deals.extend(_decorate_deals(deals, source))
            normalized_positions.extend(group_deals_into_positions(deals, source))
        except RuntimeError as exc:
            failures.append({"source": source.name, "base_url": source.base_url, "reason": str(exc)})
            logger.warning("Ingestion ignoree pour %s: %s", source.name, exc)

        try:
            open_payload = fetch(f"{source.base_url}/positions", None)
            if isinstance(open_payload, list):
                open_positions.extend(_decorate_open_positions(open_payload, source))
        except RuntimeError:
            pass

    linked_positions = link_copy_groups(normalized_positions)
    new_positions = [
        position
        for position in linked_positions
        if config.force or str(position.get("position_key")) not in imported_keys
    ]
    shadow_transitions = build_shadow_transitions(new_positions)
    nemesis_records = build_nemesis_records(new_positions)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    artefacts = write_ingestion_artifacts(
        config=config,
        timestamp=timestamp,
        raw_deals=raw_deals,
        positions=new_positions,
        open_positions=open_positions,
        shadow_transitions=shadow_transitions,
        nemesis_records=nemesis_records,
    )

    imported_keys.update(str(position.get("position_key")) for position in new_positions)
    state.update(
        {
            "last_import_at": datetime.now(timezone.utc).isoformat(),
            "last_window_days": int(config.days),
            "imported_position_keys": sorted(imported_keys),
            "last_summary": {
                "sources": len(discovered_sources),
                "raw_deals": len(raw_deals),
                "positions": len(new_positions),
                "shadow_transitions": len(shadow_transitions),
                "nemesis_records": len(nemesis_records),
                "failures": len(failures),
            },
        }
    )
    _save_state(config.state_file, state)

    report = {
        "status": "ok",
        "timestamp": timestamp,
        "window_days": int(config.days),
        "sources": [asdict(source) for source in discovered_sources],
        "failures": failures,
        "raw_deals": len(raw_deals),
        "positions_imported": len(new_positions),
        "open_positions_seen": len(open_positions),
        "shadow_transitions": len(shadow_transitions),
        "nemesis_records": len(nemesis_records),
        "copy_groups": _summarize_copy_groups(new_positions),
        "artefacts": artefacts,
        "state_file": str(config.state_file),
    }
    latest_report = config.output_root / "reports" / "latest.json"
    latest_report.parent.mkdir(parents=True, exist_ok=True)
    latest_report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def group_deals_into_positions(
    deals: list[dict[str, Any]],
    source: BankerHistorySource,
) -> list[dict[str, Any]]:
    """
    Regroupe les deals MT5 en positions normalisees.

    Args:
        deals (list[dict[str, Any]]): Deals bruts exposes par le Banker.
        source (BankerHistorySource): Source du compte.

    Returns:
        list[dict[str, Any]]: Positions fermees deduites des deals.
    """

    grouped: dict[int, list[dict[str, Any]]] = {}
    for deal in deals:
        position_id = _safe_int(deal.get("position_id")) or _safe_int(deal.get("order"))
        if position_id is None or position_id <= 0:
            continue
        grouped.setdefault(position_id, []).append(dict(deal))

    positions: list[dict[str, Any]] = []
    for position_id, items in grouped.items():
        ordered = sorted(items, key=lambda item: _parse_datetime(item.get("time")))
        entries = [item for item in ordered if _safe_int(item.get("entry"), -1) in {0, 2}]
        exits = [item for item in ordered if _safe_int(item.get("entry"), -1) in {1, 2, 3}]
        if not entries or not exits:
            continue

        first_entry = entries[0]
        first_exit = exits[0]
        last_exit = exits[-1]
        symbol = str(first_entry.get("symbol") or last_exit.get("symbol") or "")
        canonical_symbol = canonicalize_symbol(symbol, source.symbol_map)
        action = str(first_entry.get("type") or "UNKNOWN").upper()
        entry_time = _parse_datetime(first_entry.get("time"))
        exit_time = _parse_datetime(last_exit.get("time"))
        gross_profit = sum(_safe_float(item.get("profit")) for item in exits)
        swap = sum(_safe_float(item.get("swap")) for item in exits)
        commission = sum(_safe_float(item.get("commission")) for item in exits)
        net_pnl = gross_profit + swap + commission
        entry_volume = sum(_safe_float(item.get("volume")) for item in entries)
        exit_volume = sum(_safe_float(item.get("volume")) for item in exits)
        volume = max(entry_volume, exit_volume, _safe_float(first_entry.get("volume")))
        entry_price = _safe_float(first_entry.get("price"))
        exit_price = _safe_float(last_exit.get("price"))
        partial_exit_count = max(0, len(exits) - 1)
        eva_close_count = sum(1 for item in exits if "eva close" in str(item.get("comment") or "").lower())
        first_exit_profit = _safe_float(first_exit.get("profit"))
        final_exit_profit = _safe_float(last_exit.get("profit"))
        close_ratio = min(1.0, exit_volume / max(volume, 1e-12))
        duration_seconds = max(0.0, (exit_time - entry_time).total_seconds())
        position_key = f"{source.account_key}:{position_id}"

        positions.append(
            {
                "position_key": position_key,
                "position_id": position_id,
                "account_key": source.account_key,
                "account_name": source.name,
                "account_role": source.role,
                "account_login": source.login,
                "account_server": source.server,
                "broker": source.broker,
                "phase": source.phase,
                "symbol": symbol,
                "canonical_symbol": canonical_symbol,
                "action": action,
                "entry_time": entry_time.isoformat(),
                "exit_time": exit_time.isoformat(),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "volume": volume,
                "closed_volume": exit_volume,
                "close_ratio": close_ratio,
                "gross_profit": gross_profit,
                "swap": swap,
                "commission": commission,
                "net_pnl": net_pnl,
                "duration_seconds": duration_seconds,
                "entry_ticket": _safe_int(first_entry.get("ticket")),
                "exit_ticket": _safe_int(last_exit.get("ticket")),
                "magic": _safe_int(first_entry.get("magic"), 0),
                "entry_comment": str(first_entry.get("comment") or ""),
                "exit_comment": str(last_exit.get("comment") or ""),
                "partial_exit_count": partial_exit_count,
                "eva_close_count": eva_close_count,
                "first_exit_profit": first_exit_profit,
                "final_exit_profit": final_exit_profit,
                "had_profitable_partial": bool(eva_close_count > 0 and first_exit_profit > 0.0),
                "runner_final_negative": bool(eva_close_count > 0 and net_pnl < 0.0),
                "raw_deal_count": len(ordered),
            }
        )

    return sorted(positions, key=lambda item: (item["exit_time"], item["account_key"], item["position_id"]))


def canonicalize_symbol(symbol: str, symbol_map: dict[str, str] | None = None) -> str:
    """
    Ramene un symbole broker vers le symbole canonique du master.

    Args:
        symbol (str): Symbole vu sur le compte.
        symbol_map (dict[str, str] | None): Mapping canonique vers broker.

    Returns:
        str: Symbole canonique normalise.
    """

    raw_symbol = str(symbol or "").strip()
    if not raw_symbol:
        return ""
    reverse_map = {str(remote).upper(): str(master) for master, remote in (symbol_map or {}).items()}
    mapped = reverse_map.get(raw_symbol.upper())
    if mapped:
        return mapped

    upper_symbol = raw_symbol.upper()
    suffixes = (".M", ".E", ".CASH", ".PRO", ".RAW")
    for suffix in suffixes:
        if upper_symbol.endswith(suffix):
            upper_symbol = upper_symbol[: -len(suffix)]
            break
    aliases = {
        "DE40": "GER40.cash",
        "GER40": "GER40.cash",
        "US30": "US30.cash",
        "DJ30": "US30.cash",
        "USTEC": "US100.cash",
        "UT100": "US100.cash",
        "NAS100": "US100.cash",
        "US500": "US500.cash",
        "SP500": "US500.cash",
        "XAUUSD": "XAUUSD",
        "BTCUSD": "BTCUSD",
    }
    return aliases.get(upper_symbol, upper_symbol)


def link_copy_groups(positions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Ajoute une cle de groupe pour rapprocher master et followers copies.

    Args:
        positions (list[dict[str, Any]]): Positions normalisees.

    Returns:
        list[dict[str, Any]]: Positions enrichies avec groupe de copie.
    """

    grouped: dict[str, list[dict[str, Any]]] = {}
    enriched: list[dict[str, Any]] = []
    for position in positions:
        entry_time = _parse_datetime(position.get("entry_time"))
        group_time = int(entry_time.timestamp() // 300) * 300
        copy_group_id = (
            f"{position.get('canonical_symbol')}:{position.get('action')}:{group_time}"
        )
        item = dict(position)
        item["copy_group_id"] = copy_group_id
        grouped.setdefault(copy_group_id, []).append(item)
        enriched.append(item)

    group_summary = {
        group_id: {
            "accounts": sorted({str(item.get("account_key")) for item in items}),
            "master_present": any(str(item.get("account_role")) == "master" for item in items),
            "followers": sum(1 for item in items if str(item.get("account_role")) != "master"),
        }
        for group_id, items in grouped.items()
    }
    for item in enriched:
        item["copy_group"] = group_summary.get(str(item.get("copy_group_id")), {})
    return enriched


def build_shadow_transitions(positions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Transforme les positions normalisees en transitions Shadow Learning.

    Args:
        positions (list[dict[str, Any]]): Positions fermees normalisees.

    Returns:
        list[dict[str, Any]]: Transitions JSONL compatibles avec ``shadow_dataset``.
    """

    transitions: list[dict[str, Any]] = []
    for position in positions:
        episode_id = f"mt5-fleet:{position.get('position_key')}"
        base_metadata = {
            "episode_id": episode_id,
            "source": "mt5_fleet_history",
            "position_id": position.get("position_id"),
            "position_key": position.get("position_key"),
            "account_key": position.get("account_key"),
            "account_role": position.get("account_role"),
            "broker": position.get("broker"),
            "symbol": position.get("canonical_symbol"),
            "raw_symbol": position.get("symbol"),
            "profit": position.get("net_pnl"),
            "net_pnl": position.get("net_pnl"),
            "close_ratio": position.get("close_ratio"),
            "slbe": position.get("eva_close_count", 0) > 0,
            "copy_group_id": position.get("copy_group_id"),
        }
        entry_action = str(position.get("action") or "HOLD").upper()
        reward = _scaled_reward(position)
        transitions.append(
            {
                "timestamp": position.get("entry_time"),
                "observation": _build_trade_observation(position, at_exit=False),
                "action": {
                    "type": entry_action if entry_action in {"BUY", "SELL"} else "HOLD",
                    "symbol": position.get("canonical_symbol"),
                    "volume": position.get("volume"),
                },
                "reward": 0.0,
                "next_observation": _build_trade_observation(position, at_exit=True),
                "metadata": {**base_metadata, "step_index": 0, "steps_total": 2},
                "done": False,
            }
        )
        if int(position.get("eva_close_count", 0) or 0) > 0:
            transitions.append(
                {
                    "timestamp": position.get("exit_time"),
                    "observation": _build_trade_observation(position, at_exit=True),
                    "action": {
                        "type": "SPLIT",
                        "symbol": position.get("canonical_symbol"),
                        "volume": position.get("closed_volume"),
                        "close_ratio": position.get("close_ratio"),
                        "slbe": True,
                    },
                    "reward": max(0.0, reward * 0.70),
                    "next_observation": _build_trade_observation(position, at_exit=True),
                    "metadata": {**base_metadata, "step_index": 1, "steps_total": 3},
                    "done": False,
                }
            )
        transitions.append(
            {
                "timestamp": position.get("exit_time"),
                "observation": _build_trade_observation(position, at_exit=True),
                "action": {
                    "type": "CLOSE",
                    "symbol": position.get("canonical_symbol"),
                    "volume": position.get("closed_volume"),
                },
                "reward": reward,
                "next_observation": _build_trade_observation(position, at_exit=True),
                "metadata": {**base_metadata, "step_index": 2, "steps_total": 3},
                "done": True,
            }
        )
    return transitions


def build_nemesis_records(positions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Extrait les pertes et anomalies live en slices Nemesis.

    Args:
        positions (list[dict[str, Any]]): Positions fermees normalisees.

    Returns:
        list[dict[str, Any]]: Records Nemesis exploitables en stress Arena.
    """

    records: list[dict[str, Any]] = []
    for position in positions:
        net_pnl = _safe_float(position.get("net_pnl"))
        tags = _classify_nemesis_tags(position)
        if net_pnl >= 0.0 and not tags:
            continue
        records.append(
            {
                "id": f"nemesis:{position.get('position_key')}",
                "source": "mt5_fleet_history",
                "tags": tags or ["loss"],
                "nemesis_type": (tags or ["LOSS"])[0].upper(),
                "symbol": position.get("canonical_symbol"),
                "account_key": position.get("account_key"),
                "account_role": position.get("account_role"),
                "entry_time": position.get("entry_time"),
                "exit_time": position.get("exit_time"),
                "net_pnl": net_pnl,
                "profit": position.get("gross_profit"),
                "duration_seconds": position.get("duration_seconds"),
                "copy_group_id": position.get("copy_group_id"),
                "metadata": {
                    "position_key": position.get("position_key"),
                    "broker": position.get("broker"),
                    "action": position.get("action"),
                    "had_profitable_partial": position.get("had_profitable_partial"),
                    "runner_final_negative": position.get("runner_final_negative"),
                    "copy_group": position.get("copy_group"),
                },
            }
        )
    return records


def write_ingestion_artifacts(
    *,
    config: IngestionConfig,
    timestamp: str,
    raw_deals: list[dict[str, Any]],
    positions: list[dict[str, Any]],
    open_positions: list[dict[str, Any]],
    shadow_transitions: list[dict[str, Any]],
    nemesis_records: list[dict[str, Any]],
) -> dict[str, str | None]:
    """
    Ecrit les artefacts JSONL produits par l'ingestion.

    Args:
        config (IngestionConfig): Configuration de l'execution.
        timestamp (str): Horodatage stable du batch.
        raw_deals (list[dict[str, Any]]): Deals bruts decores.
        positions (list[dict[str, Any]]): Positions normalisees.
        open_positions (list[dict[str, Any]]): Positions ouvertes observees.
        shadow_transitions (list[dict[str, Any]]): Transitions Shadow.
        nemesis_records (list[dict[str, Any]]): Slices Nemesis.

    Returns:
        dict[str, str | None]: Chemins des fichiers ecrits.
    """

    paths = {
        "raw_deals": config.output_root / "raw_deals" / f"mt5_fleet_deals_{timestamp}.jsonl",
        "positions": config.output_root / "positions" / f"mt5_fleet_positions_{timestamp}.jsonl",
        "open_positions": config.output_root / "open_positions" / f"mt5_fleet_open_{timestamp}.jsonl",
        "shadow": config.shadow_output_dir / f"mt5_fleet_shadow_{timestamp}.jsonl",
        "nemesis": config.output_root / "nemesis" / f"mt5_fleet_nemesis_{timestamp}.jsonl",
    }
    _write_jsonl(paths["raw_deals"], raw_deals)
    _write_jsonl(paths["positions"], positions)
    _write_jsonl(paths["open_positions"], open_positions)
    _write_jsonl(paths["shadow"], shadow_transitions)
    _write_jsonl(paths["nemesis"], nemesis_records)
    return {key: str(path) for key, path in paths.items()}


def _decorate_deals(deals: list[dict[str, Any]], source: BankerHistorySource) -> list[dict[str, Any]]:
    """Ajoute les metadonnees de compte aux deals bruts."""

    decorated: list[dict[str, Any]] = []
    for deal in deals:
        item = dict(deal)
        item.update(
            {
                "account_key": source.account_key,
                "account_name": source.name,
                "account_role": source.role,
                "account_login": source.login,
                "account_server": source.server,
                "broker": source.broker,
            }
        )
        decorated.append(item)
    return decorated


def _decorate_open_positions(
    positions: list[dict[str, Any]],
    source: BankerHistorySource,
) -> list[dict[str, Any]]:
    """Ajoute les metadonnees de compte aux positions ouvertes."""

    decorated: list[dict[str, Any]] = []
    for position in positions:
        item = dict(position)
        raw_symbol = str(item.get("symbol") or "")
        item.update(
            {
                "account_key": source.account_key,
                "account_name": source.name,
                "account_role": source.role,
                "account_login": source.login,
                "account_server": source.server,
                "broker": source.broker,
                "canonical_symbol": canonicalize_symbol(raw_symbol, source.symbol_map),
            }
        )
        decorated.append(item)
    return decorated


def _merge_source_account(
    source: BankerHistorySource,
    account_payload: dict[str, Any],
) -> BankerHistorySource:
    """Complete une source avec les metadonnees renvoyees par l'endpoint."""

    login = source.login or _safe_int(account_payload.get("login"))
    server = source.server or str(account_payload.get("server") or "")
    broker = source.broker or str(account_payload.get("broker") or "")
    name = source.name if source.name != "Master" else str(account_payload.get("instance_name") or source.name)
    return BankerHistorySource(
        name=name,
        base_url=source.base_url,
        role=source.role,
        login=login,
        server=server,
        broker=broker,
        enabled=source.enabled,
        symbol_map=source.symbol_map,
        phase=source.phase,
    )


def _build_trade_observation(position: dict[str, Any], *, at_exit: bool) -> dict[str, Any]:
    """Construit une observation minimale depuis une position MT5."""

    price = _safe_float(position.get("exit_price" if at_exit else "entry_price"))
    timestamp = str(position.get("exit_time" if at_exit else "entry_time") or "")
    return {
        "symbol": position.get("canonical_symbol"),
        "horizon": "scalp",
        "price": price,
        "timestamp": timestamp,
        "latest_candle": {
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "tick_volume": 0.0,
            "spread": 0.0,
        },
        "indicators": {
            "Return_1": _price_return(position),
            "Spread_Norm": 0.0,
        },
    }


def _scaled_reward(position: dict[str, Any]) -> float:
    """Borne la recompense issue du PnL pour eviter les outliers broker."""

    net_pnl = _safe_float(position.get("net_pnl"))
    volume = max(_safe_float(position.get("volume")), 1e-6)
    return float(max(-5.0, min(5.0, net_pnl / max(volume * 100.0, 1.0))))


def _price_return(position: dict[str, Any]) -> float:
    """Calcule le retour directionnel brut entre entree et sortie."""

    entry = _safe_float(position.get("entry_price"))
    exit_price = _safe_float(position.get("exit_price"))
    if entry <= 0:
        return 0.0
    direction = 1.0 if str(position.get("action")).upper() == "BUY" else -1.0
    return direction * ((exit_price - entry) / entry)


def _classify_nemesis_tags(position: dict[str, Any]) -> list[str]:
    """Classe une position perdante ou anormale en familles Nemesis."""

    tags: list[str] = []
    net_pnl = _safe_float(position.get("net_pnl"))
    exit_comment = str(position.get("exit_comment") or "").lower()
    entry_comment = str(position.get("entry_comment") or "").lower()
    duration_seconds = _safe_float(position.get("duration_seconds"))

    if net_pnl < 0.0:
        tags.append("loss")
    if "spread" in exit_comment or "spread" in entry_comment:
        tags.append("spread_fail")
    if bool(position.get("runner_final_negative")):
        tags.append("bad_runner")
    if int(position.get("eva_close_count", 0) or 0) <= 0 and net_pnl < 0.0:
        tags.append("bad_close")
    if duration_seconds <= 900 and net_pnl < 0.0:
        tags.append("whiplash")
    if abs(_price_return(position)) > 0.006 and net_pnl < 0.0:
        tags.append("trend_reversal")
    if not bool(dict(position.get("copy_group") or {}).get("master_present", True)):
        tags.append("copy_orphan")
    return list(dict.fromkeys(tags))


def _summarize_copy_groups(positions: list[dict[str, Any]]) -> dict[str, Any]:
    """Resume les groupes de copie produits par le batch."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for position in positions:
        grouped.setdefault(str(position.get("copy_group_id")), []).append(position)
    incomplete = [
        group_id
        for group_id, items in grouped.items()
        if not any(str(item.get("account_role")) == "master" for item in items)
        or sum(1 for item in items if str(item.get("account_role")) != "master") == 0
    ]
    return {
        "total": len(grouped),
        "incomplete": len(incomplete),
        "incomplete_ids": incomplete[:25],
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Ecrit un fichier JSONL meme si le batch est vide."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            json.dump(row, file_obj, ensure_ascii=False)
            file_obj.write("\n")


def _load_state(path: Path) -> dict[str, Any]:
    """Charge l'etat d'ingestion existant."""

    if not path.exists():
        return {"imported_position_keys": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Etat d'ingestion illisible, reinitialisation: %s", exc)
        return {"imported_position_keys": []}


def _save_state(path: Path, state: dict[str, Any]) -> None:
    """Sauvegarde l'etat d'ingestion."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def _parse_datetime(value: Any) -> datetime:
    """Convertit une valeur arbitraire en datetime UTC-naif comparable."""

    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, (int, float)):
        parsed = datetime.fromtimestamp(float(value), tz=timezone.utc)
    elif isinstance(value, str) and value.strip():
        raw_value = value.strip().replace("Z", "+00:00")
        parsed = datetime.fromisoformat(raw_value)
    else:
        return datetime.fromtimestamp(0, tz=timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _safe_float(value: Any, fallback: float = 0.0) -> float:
    """Convertit une valeur en flottant fini."""

    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(result):
        return fallback
    return result


def _safe_int(value: Any, fallback: int | None = None) -> int | None:
    """Convertit une valeur en entier optionnel."""

    try:
        if value is None or value == "":
            return fallback
        return int(value)
    except (TypeError, ValueError):
        return fallback


def resolve_path(value: str | os.PathLike[str]) -> Path:
    """
    Resolve un chemin relatif depuis le dossier courant.

    Args:
        value (str | os.PathLike[str]): Chemin source.

    Returns:
        Path: Chemin normalise.
    """

    return Path(value).expanduser().resolve()
