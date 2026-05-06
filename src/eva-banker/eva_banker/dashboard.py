"""Dashboard Rich pour superviser plusieurs instances Banker."""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from rich.columns import Columns
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


DEFAULT_ENV_GLOB = ".env.banker*.local"
DEFAULT_REFRESH_SECONDS = 2.0
DEFAULT_MAX_EVENTS = 20


@dataclass(slots=True)
class BankerInstanceConfig:
    """Decrit une instance Banker a superviser."""

    name: str
    env_file: Path
    base_url: str
    log_file: Path


@dataclass(slots=True)
class BankerSnapshot:
    """Capture l'etat instantane d'une instance Banker."""

    instance: BankerInstanceConfig
    health: dict[str, Any] | None
    trading_status: dict[str, Any] | None
    copy_status: dict[str, Any] | None
    accounts: list[dict[str, Any]]
    recent_log_lines: list[str]
    fetched_at: datetime
    error: str | None = None


def parse_env_file(path: Path) -> dict[str, str]:
    """Lit un fichier `.env` simple et retourne ses clefs.

    Args:
        path (Path): Fichier d'environnement a parser.

    Returns:
        dict[str, str]: Dictionnaire des variables detectees.
    """

    payload: dict[str, str] = {}
    if not path.exists():
        return payload
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        payload[key.strip()] = value.strip().strip("\"'")
    return payload


def _normalize_bind_host(bind_host: str) -> str:
    """Normalise l'hote de bind pour les appels HTTP locaux.

    Args:
        bind_host (str): Hote de bind tel que lu depuis le `.env`.

    Returns:
        str: Hote atteignable depuis le dashboard local.
    """

    normalized = str(bind_host or "").strip() or "127.0.0.1"
    if normalized in {"0.0.0.0", "::", "[::]"}:
        return "127.0.0.1"
    return normalized


def _slugify_name(value: str) -> str:
    """Construit un slug stable pour les fichiers lies a une instance.

    Args:
        value (str): Nom libre de l'instance.

    Returns:
        str: Nom compact compatible fichier.
    """

    normalized = "".join(char.lower() if char.isalnum() else "_" for char in str(value or "").strip())
    compact = "_".join(segment for segment in normalized.split("_") if segment)
    return compact or "banker"


def resolve_instance_log_file(root_dir: Path, env_path: Path, env_data: dict[str, str], name: str) -> Path:
    """Construit le chemin de log d'une instance Banker.

    Args:
        root_dir (Path): Racine du workspace.
        env_path (Path): Fichier `.env` de l'instance.
        env_data (dict[str, str]): Variables chargees depuis le `.env`.
        name (str): Nom lisible de l'instance.

    Returns:
        Path: Chemin du fichier de log associe.
    """

    explicit_path = str(env_data.get("BANKER_LOG_FILE") or "").strip()
    if explicit_path:
        resolved = Path(explicit_path)
        return resolved if resolved.is_absolute() else root_dir / resolved
    return root_dir / "logs" / f"{_slugify_name(name)}.log"


def read_log_tail(path: Path, max_lines: int = 8) -> list[str]:
    """Lit les dernieres lignes d'un fichier de log.

    Args:
        path (Path): Fichier a lire.
        max_lines (int): Nombre maximal de lignes a retourner.

    Returns:
        list[str]: Dernieres lignes nettoyees.
    """

    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return [line.rstrip() for line in lines[-max(1, max_lines):]]


def discover_banker_instances(root_dir: Path, env_glob: str = DEFAULT_ENV_GLOB) -> list[BankerInstanceConfig]:
    """Decouvre les instances Banker a partir des fichiers `.env` locaux.

    Args:
        root_dir (Path): Racine du workspace.
        env_glob (str): Motif de recherche des fichiers `.env`.

    Returns:
        list[BankerInstanceConfig]: Instances detectees et ordonnees.
    """

    instances: list[BankerInstanceConfig] = []
    for env_path in sorted(root_dir.glob(env_glob)):
        env_data = parse_env_file(env_path)
        api_port = int(str(env_data.get("BANKER_API_PORT") or "8100").strip())
        bind_host = _normalize_bind_host(str(env_data.get("BANKER_BIND_HOST") or "127.0.0.1"))
        raw_name = str(env_data.get("BANKER_INSTANCE_NAME") or "").strip()
        name = raw_name or env_path.stem.replace(".env.", "").replace(".local", "")
        log_file = resolve_instance_log_file(root_dir=root_dir, env_path=env_path, env_data=env_data, name=name)
        instances.append(
            BankerInstanceConfig(
                name=name,
                env_file=env_path,
                base_url=f"http://{bind_host}:{api_port}",
                log_file=log_file,
            )
        )
    instances.sort(key=lambda item: ("master" not in item.name.lower(), item.name.lower()))
    return instances


def _fetch_json(url: str, timeout_seconds: float = 3.0) -> tuple[Any | None, str | None]:
    """Lit une ressource JSON et retourne soit le contenu, soit une erreur.

    Args:
        url (str): URL de l'endpoint cible.
        timeout_seconds (float): Timeout HTTP.

    Returns:
        tuple[Any | None, str | None]: Payload JSON et message d'erreur eventuel.
    """

    request = Request(url=url, headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
            if not body.strip():
                return None, "reponse_vide"
            return json.loads(body), None
    except HTTPError as exc:
        return None, f"http_{exc.code}"
    except URLError as exc:
        return None, str(exc.reason)
    except Exception as exc:  # pragma: no cover - garde-fou IO.
        return None, str(exc)


def collect_snapshot(instance: BankerInstanceConfig) -> BankerSnapshot:
    """Interroge les endpoints d'une instance et construit un snapshot.

    Args:
        instance (BankerInstanceConfig): Instance a sonder.

    Returns:
        BankerSnapshot: Snapshot assemble pour l'affichage.
    """

    fetched_at = datetime.now()
    health, health_error = _fetch_json(f"{instance.base_url}/health")
    trading_status, trading_error = _fetch_json(f"{instance.base_url}/trading/status")
    copy_status, copy_error = _fetch_json(f"{instance.base_url}/copy-trading/status")
    accounts_payload, accounts_error = _fetch_json(f"{instance.base_url}/accounts/propfirm")

    errors = [msg for msg in (health_error, trading_error, copy_error, accounts_error) if msg]
    return BankerSnapshot(
        instance=instance,
        health=health if isinstance(health, dict) else None,
        trading_status=trading_status if isinstance(trading_status, dict) else None,
        copy_status=copy_status if isinstance(copy_status, dict) else None,
        accounts=accounts_payload if isinstance(accounts_payload, list) else [],
        recent_log_lines=read_log_tail(instance.log_file),
        fetched_at=fetched_at,
        error=" | ".join(errors) or None,
    )


def _safe_get(payload: dict[str, Any] | None, *keys: str, default: Any = None) -> Any:
    """Lit une clef imbriquee sans lever d'exception.

    Args:
        payload (dict[str, Any] | None): Structure source.
        *keys (str): Chemin de lecture.
        default (Any): Valeur de repli.

    Returns:
        Any: Valeur lue ou repli.
    """

    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def _format_float(value: Any, digits: int = 2) -> str:
    """Formate une valeur numerique de maniere defensive.

    Args:
        value (Any): Valeur a formater.
        digits (int): Nombre de decimales.

    Returns:
        str: Texte lisible.
    """

    try:
        return f"{float(value):,.{digits}f}".replace(",", " ")
    except (TypeError, ValueError):
        return "-"


def _build_status_signature(snapshot: BankerSnapshot) -> dict[str, Any]:
    """Construit une signature compacte pour detecter les changements.

    Args:
        snapshot (BankerSnapshot): Etat courant de l'instance.

    Returns:
        dict[str, Any]: Signature simplifiee exploitable par le journal.
    """

    trading = snapshot.trading_status or {}
    runtime = _safe_get(trading, "runtime", default={}) or {}
    account = _safe_get(trading, "account", default={}) or {}
    risk = _safe_get(trading, "risk", default={}) or {}
    mechanics = _safe_get(trading, "execution_mechanics", default={}) or {}
    return {
        "health_status": _safe_get(snapshot.health, "status"),
        "mt5_connected": _safe_get(snapshot.health, "mt5_connected"),
        "runtime_mode": runtime.get("runtime_mode"),
        "runtime_profile": runtime.get("runtime_profile"),
        "active_live_engine": mechanics.get("active_live_engine"),
        "live_champion_id": mechanics.get("live_champion_id"),
        "open_positions": risk.get("open_positions"),
        "equity": account.get("equity"),
        "copy_enabled": _safe_get(snapshot.copy_status, "enabled", default=False),
        "targets_count": len(_safe_get(snapshot.copy_status, "targets", default=[]) or []),
    }


def build_change_events(
    previous_snapshot: BankerSnapshot | None,
    current_snapshot: BankerSnapshot,
) -> list[str]:
    """Genere un journal court a partir des changements de statut.

    Args:
        previous_snapshot (BankerSnapshot | None): Snapshot precedent si disponible.
        current_snapshot (BankerSnapshot): Snapshot courant.

    Returns:
        list[str]: Evenements textuels a ajouter au journal.
    """

    if previous_snapshot is None:
        return [f"{current_snapshot.instance.name}: supervision initialisee"]

    previous = _build_status_signature(previous_snapshot)
    current = _build_status_signature(current_snapshot)
    events: list[str] = []

    for key, label in (
        ("health_status", "sante"),
        ("mt5_connected", "connexion MT5"),
        ("runtime_mode", "mode runtime"),
        ("active_live_engine", "moteur live"),
        ("live_champion_id", "champion live"),
        ("open_positions", "positions ouvertes"),
        ("copy_enabled", "copy trading"),
        ("targets_count", "nombre de cibles"),
    ):
        if previous.get(key) != current.get(key):
            events.append(
                f"{current_snapshot.instance.name}: {label} {previous.get(key)} -> {current.get(key)}"
            )

    previous_equity = previous.get("equity")
    current_equity = current.get("equity")
    if previous_equity != current_equity and previous_equity is not None and current_equity is not None:
        if abs(float(current_equity) - float(previous_equity)) >= 0.01:
            events.append(
                f"{current_snapshot.instance.name}: equity {float(previous_equity):.2f} -> {float(current_equity):.2f}"
            )

    if previous_snapshot.error != current_snapshot.error:
        events.append(
            f"{current_snapshot.instance.name}: erreur endpoint {previous_snapshot.error} -> {current_snapshot.error}"
        )

    return events


def _render_instance_panel(snapshot: BankerSnapshot) -> Panel:
    """Construit le panneau Rich d'une instance Banker.

    Args:
        snapshot (BankerSnapshot): Snapshot a afficher.

    Returns:
        Panel: Panneau formate.
    """

    trading = snapshot.trading_status or {}
    runtime = _safe_get(trading, "runtime", default={}) or {}
    mechanics = _safe_get(trading, "execution_mechanics", default={}) or {}
    account = _safe_get(trading, "account", default={}) or {}
    risk = _safe_get(trading, "risk", default={}) or {}
    universe = _safe_get(trading, "universe", "lab_live", default={}) or {}
    targets = _safe_get(snapshot.copy_status, "targets", default=[]) or []
    accounts = snapshot.accounts or []

    role = str(accounts[0].get("copy_role") or "-") if accounts else "-"
    server = str(accounts[0].get("server") or "-") if accounts else "-"
    phase = str(accounts[0].get("phase") or "-") if accounts else "-"
    status = str(_safe_get(snapshot.health, "status", default="unknown")).lower()
    mt5_connected = bool(_safe_get(snapshot.health, "mt5_connected", default=False))
    status_color = "green" if status == "ok" and mt5_connected else "yellow" if mt5_connected else "red"
    title = f"[bold]{snapshot.instance.name}[/bold] [{snapshot.instance.base_url}]"

    table = Table.grid(padding=(0, 1))
    table.add_column(style="cyan", no_wrap=True)
    table.add_column()
    table.add_row("Etat", f"[{status_color}]{status}[/{status_color}]")
    table.add_row("MT5", "connecte" if mt5_connected else "hors ligne")
    table.add_row("Role", role)
    table.add_row("Serveur", server)
    table.add_row("Phase", phase)
    table.add_row("Runtime", str(runtime.get("runtime_mode") or "-"))
    table.add_row("Profil", str(runtime.get("runtime_profile") or "-"))
    table.add_row("Moteur live", str(mechanics.get("active_live_engine") or "-"))
    table.add_row("Champion", str(mechanics.get("live_champion_id") or mechanics.get("live_champion_id_muzero") or "-"))
    table.add_row("Entrees live", "oui" if bool(universe.get("live_entries_allowed", False)) else "non")
    table.add_row("Gate", str(universe.get("gate_reason") or "-"))
    table.add_row("Balance", _format_float(account.get("balance")))
    table.add_row("Equity", _format_float(account.get("equity")))
    table.add_row("Positions", str(risk.get("open_positions") or 0))
    table.add_row("Trading", "autorise" if bool(risk.get("trading_allowed", False)) else "bloque")
    table.add_row("Cibles copy", ", ".join(str(target.get("name") or "-") for target in targets) or "-")
    table.add_row("Log", str(snapshot.instance.log_file.name))
    if snapshot.error:
        table.add_row("Erreur", f"[red]{snapshot.error}[/red]")

    footer = Text(
        f"Dernier refresh: {snapshot.fetched_at.strftime('%H:%M:%S')} | env: {snapshot.instance.env_file.name}",
        style="dim",
    )
    return Panel(Group(table, footer), title=title, border_style=status_color)


def _render_events_panel(events: deque[str]) -> Panel:
    """Construit le panneau de journal d'evenements.

    Args:
        events (deque[str]): Evenements les plus recents.

    Returns:
        Panel: Panneau de journal.
    """

    table = Table.grid(expand=True)
    table.add_column()
    if not events:
        table.add_row("[dim]Aucun evenement pour l'instant.[/dim]")
    else:
        for event in reversed(events):
            table.add_row(event)
    return Panel(table, title="Journal Banker", border_style="blue")


def _render_logs_panel(snapshots: list[BankerSnapshot]) -> Panel:
    """Construit un panneau de tail de logs pour toutes les instances.

    Args:
        snapshots (list[BankerSnapshot]): Snapshots courants.

    Returns:
        Panel: Panneau combine des tails de logs.
    """

    tables: list[Panel] = []
    for snapshot in snapshots:
        table = Table.grid(expand=True)
        table.add_column()
        if snapshot.recent_log_lines:
            for line in snapshot.recent_log_lines:
                table.add_row(Text(line, overflow="fold"))
        else:
            table.add_row("[dim]Aucun log persistant pour cette instance.[/dim]")
        tables.append(
            Panel(
                table,
                title=f"Logs {snapshot.instance.name}",
                border_style="magenta",
            )
        )
    return Panel(Columns(tables, expand=True, equal=True), title="Flux logs Banker", border_style="magenta")


def extract_recent_decision_lines(snapshot: BankerSnapshot, max_lines: int = 6) -> list[str]:
    """Construit un resume lisible du flux decisionnel du champion.

    Args:
        snapshot (BankerSnapshot): Snapshot courant a analyser.
        max_lines (int): Nombre maximal de lignes a retourner.

    Returns:
        list[str]: Lignes de synthese du flux decisionnel live.
    """

    trading = snapshot.trading_status or {}
    mechanics = _safe_get(trading, "execution_mechanics", default={}) or {}
    active_engine = str(mechanics.get("active_live_engine") or trading.get("active_live_engine") or "").strip().lower()
    recent = list((_safe_get(trading, "decision_audit", "recent", default=[]) or []))
    if recent:
        lines: list[str] = []
        for event in recent[-max(1, max_lines):]:
            timestamp = str(event.get("timestamp") or "").strip()
            hhmmss = timestamp[11:19] if len(timestamp) >= 19 else "--:--:--"
            symbol = str(event.get("symbol") or "-")
            action = str(event.get("post_veto_action") or event.get("raw_model_action") or "-")
            engine = str(event.get("engine_name") or active_engine or "-")
            champion = str(event.get("model_version") or event.get("checkpoint") or "-")
            selection = str(event.get("selection") or "-")
            lines.append(
                f"[{hhmmss}] {symbol} -> {action} | moteur={engine} | selection={selection} | modele={champion}"
            )
        return lines

    decisions = trading.get("decisions")
    if isinstance(decisions, dict) and decisions:
        lines = []
        for symbol, decision in list(decisions.items())[: max(1, max_lines)]:
            action = str(decision.get("post_veto_action") or decision.get("action") or "-")
            comment = str(decision.get("comment") or "-")
            champion = str(decision.get("model_version") or decision.get("checkpoint") or "-")
            lines.append(
                f"{symbol} -> {action} | modele={champion} | {comment}"
            )
        return lines

    if active_engine:
        champion = str(
            mechanics.get("live_champion_id")
            or mechanics.get("live_champion_id_muzero")
            or trading.get("live_champion_id_muzero")
            or "-"
        )
        return [f"Moteur live actif: {active_engine} | champion={champion} | attente des prochaines decisions."]

    return ["Aucun flux champion disponible pour cette instance."]


def _render_decisions_panel(snapshots: list[BankerSnapshot]) -> Panel:
    """Construit un panneau de flux decisionnel live par instance.

    Args:
        snapshots (list[BankerSnapshot]): Snapshots courants.

    Returns:
        Panel: Panneau combine des decisions recentes.
    """

    tables: list[Panel] = []
    for snapshot in snapshots:
        table = Table.grid(expand=True)
        table.add_column()
        for line in extract_recent_decision_lines(snapshot):
            table.add_row(Text(line, overflow="fold"))
        tables.append(
            Panel(
                table,
                title=f"Flux champion {snapshot.instance.name}",
                border_style="yellow",
            )
        )
    return Panel(Columns(tables, expand=True, equal=True), title="Flux decisionnel live", border_style="yellow")


def build_dashboard_renderable(
    snapshots: list[BankerSnapshot],
    events: deque[str],
    refresh_seconds: float,
) -> Group:
    """Assemble la vue complete du dashboard.

    Args:
        snapshots (list[BankerSnapshot]): Etats courants des instances.
        events (deque[str]): Journal des changements detectes.
        refresh_seconds (float): Frequence de refresh.

    Returns:
        Group: Bloc Rich pret a etre affiche.
    """

    summary = Table.grid(expand=True)
    summary.add_column()
    summary.add_row(
        f"[bold]Dashboard Banker[/bold] | instances={len(snapshots)} | refresh={refresh_seconds:.1f}s | horodatage={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    columns = Columns([_render_instance_panel(snapshot) for snapshot in snapshots], expand=True, equal=True)
    return Group(
        Panel(summary, border_style="white"),
        columns,
        _render_events_panel(events),
        _render_decisions_panel(snapshots),
        _render_logs_panel(snapshots),
    )


def run_dashboard(
    root_dir: Path,
    env_glob: str = DEFAULT_ENV_GLOB,
    refresh_seconds: float = DEFAULT_REFRESH_SECONDS,
    max_events: int = DEFAULT_MAX_EVENTS,
) -> None:
    """Lance la supervision temps reel des instances Banker.

    Args:
        root_dir (Path): Racine du workspace.
        env_glob (str): Motif de decouverte des `.env` Banker.
        refresh_seconds (float): Delai entre deux refresh.
        max_events (int): Taille maximale du journal.

    Raises:
        RuntimeError: Si aucune instance Banker n'est decouverte.
    """

    instances = discover_banker_instances(root_dir, env_glob=env_glob)
    if not instances:
        raise RuntimeError("Aucune instance Banker n'a ete decouverte.")

    history: dict[str, BankerSnapshot] = {}
    events: deque[str] = deque(maxlen=max_events)
    initial_snapshots = [collect_snapshot(instance) for instance in instances]
    for snapshot in initial_snapshots:
        events.extend(build_change_events(None, snapshot))
        history[snapshot.instance.name] = snapshot

    with Live(
        build_dashboard_renderable(initial_snapshots, events, refresh_seconds),
        refresh_per_second=max(1, int(round(1 / max(refresh_seconds, 0.1)))),
        screen=True,
    ) as live:
        while True:
            time.sleep(refresh_seconds)
            snapshots = [collect_snapshot(instance) for instance in instances]
            for snapshot in snapshots:
                previous = history.get(snapshot.instance.name)
                events.extend(build_change_events(previous, snapshot))
                history[snapshot.instance.name] = snapshot
            live.update(build_dashboard_renderable(snapshots, events, refresh_seconds))


def _build_argument_parser() -> argparse.ArgumentParser:
    """Construit le parseur CLI du dashboard.

    Returns:
        argparse.ArgumentParser: Parseur configure.
    """

    parser = argparse.ArgumentParser(description="Dashboard Rich pour plusieurs Bankers locaux.")
    parser.add_argument(
        "--refresh-seconds",
        type=float,
        default=DEFAULT_REFRESH_SECONDS,
        help="Delai entre deux refresh du dashboard.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=DEFAULT_MAX_EVENTS,
        help="Nombre maximal d'evenements conserves dans le journal.",
    )
    parser.add_argument(
        "--env-glob",
        type=str,
        default=DEFAULT_ENV_GLOB,
        help="Motif glob des fichiers `.env` Banker a surveiller.",
    )
    return parser


def main() -> None:
    """Point d'entree CLI du dashboard Banker."""

    parser = _build_argument_parser()
    args = parser.parse_args()
    run_dashboard(
        root_dir=Path.cwd(),
        env_glob=str(args.env_glob),
        refresh_seconds=max(0.5, float(args.refresh_seconds)),
        max_events=max(5, int(args.max_events)),
    )


if __name__ == "__main__":
    main()
