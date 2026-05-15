"""Exporte un inventaire CSV des comptes Banker et terminaux MT5 locaux."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "data" / "account_fleet"


def parse_env_file(path: Path) -> dict[str, str]:
    """
    Lit un fichier `.env` Banker sans evaluer les secrets.

    Args:
        path (Path): Chemin du fichier `.env` a analyser.

    Returns:
        dict[str, str]: Variables utiles du fichier, avec guillemets retires.
    """
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key.strip()] = value
    return values


def request_json(url: str, timeout: float) -> dict[str, Any] | None:
    """
    Recupere un JSON local avec un timeout court.

    Args:
        url (str): URL HTTP locale a interroger.
        timeout (float): Delai maximal en secondes.

    Returns:
        dict[str, Any] | None: Payload JSON si disponible, sinon None.
    """
    try:
        with urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8", "replace"))
    except (OSError, URLError, TimeoutError, json.JSONDecodeError):
        return None


def infer_broker(env: dict[str, str], target: dict[str, Any] | None) -> str:
    """
    Deduit le broker a partir du serveur MT5 et du nom d'instance.

    Args:
        env (dict[str, str]): Variables de configuration Banker.
        target (dict[str, Any] | None): Cible copy-trading correspondante.

    Returns:
        str: Broker normalise, par exemple `FTMO` ou `FTUK`.
    """
    if target and target.get("broker"):
        return str(target["broker"])
    source = " ".join(
        [
            env.get("MT5_SERVER", ""),
            env.get("BANKER_INSTANCE_NAME", ""),
            env.get("MT5_TERMINAL_PATH", ""),
        ]
    ).upper()
    if "FTUK" in source:
        return "FTUK"
    if "FTMO" in source:
        return "FTMO"
    return ""


def parse_owner_from_mt5_name(account_name: str) -> str:
    """
    Extrait un nom de proprietaire depuis le libelle MT5.

    Les comptes FTUK remontent souvent un libelle de type
    `FTUK - Flex Challenge - Prenom Nom`. On conserve le nom final, et on
    evite de transformer les libelles FTMO qui ne contiennent pas de
    proprietaire humain.

    Args:
        account_name (str): Nom brut remonte par `mt5.account_info()`.

    Returns:
        str: Nom humain extrait si disponible, sinon chaine vide.
    """
    normalized = str(account_name or "").strip()
    if not normalized:
        return ""

    parts = [part.strip() for part in re.split(r"\s+-\s+", normalized) if part.strip()]
    if len(parts) < 2:
        return ""

    candidate = parts[-1]
    lowered = candidate.lower()
    blocked_terms = {
        "challenge",
        "funded",
        "flex",
        "swing",
        "step",
        "2-step",
        "ftmo",
        "ftuk",
    }
    if any(term in lowered for term in blocked_terms):
        return ""
    if not re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ]", candidate):
        return ""
    return candidate


def infer_owner(env: dict[str, str], account: dict[str, Any] | None = None) -> tuple[str, str]:
    """
    Deduit le proprietaire depuis MT5, la config ou le chemin du terminal.

    Args:
        env (dict[str, str]): Variables de configuration Banker.
        account (dict[str, Any] | None): Informations compte exposees par MT5.

    Returns:
        tuple[str, str]: Nom du proprietaire et source utilisee.
    """
    if account:
        mt5_name = str(account.get("name") or "").strip()
        mt5_owner = parse_owner_from_mt5_name(mt5_name)
        if mt5_owner:
            return mt5_owner, "mt5_account_name"

    explicit_owner = env.get("BANKER_ACCOUNT_OWNER") or env.get("ACCOUNT_OWNER")
    if explicit_owner:
        return explicit_owner.strip(), "env"

    return "", "unavailable"


def infer_terminal_folder_owner(env: dict[str, str]) -> str:
    """
    Deduit un proprietaire probable depuis le dossier du terminal.

    Cette valeur reste un indice d'exploitation, pas la source officielle du
    proprietaire. Le CSV l'expose donc dans une colonne separee.

    Args:
        env (dict[str, str]): Variables de configuration Banker.

    Returns:
        str: Nom probable deduit du dossier, sinon chaine vide.
    """
    terminal_path = env.get("MT5_TERMINAL_PATH", "")
    normalized = terminal_path.replace("\\", "/")
    for segment in normalized.split("/"):
        if not segment:
            continue
        lowered = segment.lower()
        if lowered == "john" or lowered.startswith("john_"):
            return "John"
        if lowered == "robin" or lowered.startswith("robin_"):
            return "Robin"
    return ""


def infer_role(env: dict[str, str]) -> str:
    """
    Deduit le role operationnel du compte.

    Args:
        env (dict[str, str]): Variables de configuration Banker.

    Returns:
        str: `master` ou `follower`.
    """
    follower_mode = env.get("BANKER_FOLLOWER_MODE", "").strip().lower()
    return "follower" if follower_mode in {"1", "true", "yes", "on"} else "master"


def infer_challenge_type(
    env: dict[str, str],
    target: dict[str, Any] | None,
    role: str,
) -> str:
    """
    Deduit le type de compte/challenge.

    Args:
        env (dict[str, str]): Variables de configuration Banker.
        target (dict[str, Any] | None): Cible copy-trading correspondante.
        role (str): Role operationnel deja deduit.

    Returns:
        str: Type normalise (`master`, `challenge`, `funded`, etc.).
    """
    if role == "master":
        return "master"
    if target and target.get("phase"):
        return str(target["phase"])
    name = env.get("BANKER_INSTANCE_NAME", "").lower()
    if "funded" in name:
        return "funded"
    if "challenge" in name:
        return "challenge"
    return ""


def to_float(value: Any) -> float | None:
    """
    Convertit une valeur JSON ou texte en float tolerant.

    Args:
        value (Any): Valeur brute.

    Returns:
        float | None: Nombre converti, sinon None.
    """
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_copy_targets(master_env: dict[str, str]) -> dict[int, dict[str, Any]]:
    """
    Charge les targets copy-trading du master par login.

    Args:
        master_env (dict[str, str]): Variables du fichier `.env` master.

    Returns:
        dict[int, dict[str, Any]]: Cibles indexees par login MT5.
    """
    raw = master_env.get("BANKER_COPY_TARGETS_JSON", "").strip()
    if not raw:
        return {}
    try:
        targets = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return {
        int(target["login"]): target
        for target in targets
        if isinstance(target, dict) and str(target.get("login", "")).isdigit()
    }


def collect_rows(timeout: float) -> list[dict[str, Any]]:
    """
    Construit les lignes CSV a partir des envs et des API Banker locales.

    Args:
        timeout (float): Timeout HTTP par API locale.

    Returns:
        list[dict[str, Any]]: Lignes pretes a ecrire.
    """
    env_files = sorted(ROOT.glob(".env.banker*.local"))
    master_env = parse_env_file(ROOT / ".env.banker.master.local") if (ROOT / ".env.banker.master.local").exists() else {}
    targets_by_login = load_copy_targets(master_env)
    generated_at = datetime.now().isoformat(timespec="seconds")
    rows: list[dict[str, Any]] = []

    for env_file in env_files:
        env = parse_env_file(env_file)
        login_raw = env.get("MT5_LOGIN", "")
        login = int(login_raw) if login_raw.isdigit() else None
        target = targets_by_login.get(login) if login is not None else None
        port = env.get("BANKER_API_PORT", "")
        api_url = f"http://127.0.0.1:{port}" if port else ""
        status = request_json(f"{api_url}/trading/status", timeout) if api_url else None
        account = status.get("account", {}) if status else {}
        risk = status.get("risk", {}) if status else {}
        positions = status.get("positions", []) if status else []
        role = infer_role(env)
        broker = infer_broker(env, target)
        balance = to_float(account.get("balance"))
        equity = to_float(account.get("equity"))
        floating_profit = (equity - balance) if equity is not None and balance is not None else None
        open_profit = sum(to_float(position.get("profit")) or 0.0 for position in positions)
        owner_name, owner_source = infer_owner(env, account)

        rows.append(
            {
                "generated_at": generated_at,
                "config_file": env_file.name,
                "instance_name": env.get("BANKER_INSTANCE_NAME", ""),
                "owner_name": owner_name,
                "owner_source": owner_source,
                "terminal_folder_owner_guess": infer_terminal_folder_owner(env),
                "mt5_account_name": str(account.get("name") or "") if account else "",
                "mt5_company": str(account.get("company") or "") if account else "",
                "role": role,
                "broker": broker,
                "challenge_type": infer_challenge_type(env, target, role),
                "login": login or "",
                "server": env.get("MT5_SERVER", ""),
                "api_port": port,
                "api_status": status.get("status", "down") if status else "down",
                "mt5_connected": status.get("connection", {}).get("mt5_connected", False) if status else False,
                "copy_target_present": bool(target),
                "copy_target_enabled": bool(target.get("enabled", False)) if target else False,
                "balance": balance,
                "equity": equity,
                "floating_profit": floating_profit,
                "open_positions_profit": open_profit if status else None,
                "open_positions": risk.get("open_positions"),
                "open_positions_total": risk.get("open_positions_total"),
                "hold_positions": risk.get("hold_positions"),
                "ignored_positions": risk.get("ignored_positions"),
                "trading_allowed": risk.get("trading_allowed"),
                "terminal_path": env.get("MT5_TERMINAL_PATH", ""),
            }
        )
    return rows


def write_csv(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    """
    Ecrit le snapshot CSV et met a jour le lien logique `latest`.

    Args:
        rows (list[dict[str, Any]]): Lignes a ecrire.
        output_dir (Path): Repertoire de sortie.

    Returns:
        Path: Chemin du fichier timestamp cree.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_path = output_dir / f"banker_accounts_{timestamp}.csv"
    latest_path = output_dir / "banker_accounts_latest.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    for path in (snapshot_path, latest_path):
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return snapshot_path


def main() -> int:
    """
    Point d'entree CLI de l'export CSV.

    Returns:
        int: Code de sortie processus.
    """
    parser = argparse.ArgumentParser(description="Exporte les comptes Banker locaux en CSV.")
    parser.add_argument("--timeout", type=float, default=3.0, help="Timeout HTTP par compte.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Repertoire de sortie du CSV.",
    )
    args = parser.parse_args()
    rows = collect_rows(timeout=args.timeout)
    if not rows:
        print("Aucun fichier .env.banker*.local trouve.", file=sys.stderr)
        return 1
    output_path = write_csv(rows, args.output_dir)
    print(output_path)
    print(args.output_dir / "banker_accounts_latest.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
