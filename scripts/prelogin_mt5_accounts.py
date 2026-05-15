"""Pre-connecte les comptes MT5 avant le demarrage des Bankers.

Ce script evite les fenetres interactives de connexion MetaTrader 5 en
forcant une initialisation Python explicite pour chaque terminal dedie.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_env_file(path: Path) -> dict[str, str]:
    """Charge un fichier `.env` simple.

    Args:
        path (Path): Chemin du fichier d'environnement a lire.

    Returns:
        dict[str, str]: Variables chargees sous forme cle-valeur.
    """

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key.strip()] = value
    return values


def as_bool(value: str | None) -> bool:
    """Convertit une valeur texte en booleen.

    Args:
        value (str | None): Valeur a interpreter.

    Returns:
        bool: True si la valeur represente un booleen actif.
    """

    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def prelogin_account(env_path: Path) -> bool:
    """Initialise un terminal MT5 avec les identifiants du fichier fourni.

    Args:
        env_path (Path): Fichier `.env` de l'instance Banker.

    Returns:
        bool: True si la session MT5 est ouverte, False sinon.
    """

    import MetaTrader5 as mt5

    env = parse_env_file(env_path)
    login = int(env.get("MT5_LOGIN") or "0")
    password = env.get("MT5_PASSWORD") or ""
    server = env.get("MT5_SERVER") or ""
    terminal_path = env.get("MT5_TERMINAL_PATH") or ""
    portable = as_bool(env.get("MT5_TERMINAL_PORTABLE"))
    timeout = int(env.get("MT5_TERMINAL_TIMEOUT_MS") or "120000")

    if not login or not password or not server or not terminal_path:
        print(f"{env_path.name}: identifiants incomplets, ignore.")
        return False

    kwargs = {
        "path": terminal_path,
        "login": login,
        "password": password,
        "server": server,
        "timeout": timeout,
        "portable": portable,
    }

    mt5.shutdown()
    if not mt5.initialize(**kwargs):
        print(f"{env_path.name}: ECHEC {mt5.last_error()}")
        mt5.shutdown()
        return False

    account = mt5.account_info()
    if not account or int(account.login) != login:
        seen = f"{getattr(account, 'login', None)}@{getattr(account, 'server', None)}"
        print(f"{env_path.name}: mauvais compte connecte ({seen})")
        mt5.shutdown()
        return False

    print(f"{env_path.name}: OK {account.login}@{account.server} balance={account.balance}")
    mt5.shutdown()
    return True


def main(argv: list[str] | None = None) -> int:
    """Point d'entree CLI.

    Args:
        argv (list[str] | None): Arguments optionnels.

    Returns:
        int: Code retour du processus.
    """

    parser = argparse.ArgumentParser(description="Pre-connecte des comptes MT5.")
    parser.add_argument("env_files", nargs="+", help="Fichiers .env Banker a pre-connecter.")
    args = parser.parse_args(argv)

    ok = True
    for item in args.env_files:
        ok = prelogin_account(Path(item)) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
