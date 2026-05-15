"""Interface CustomTkinter pour piloter un agent follower local."""

from __future__ import annotations

import asyncio
import queue
import threading
from pathlib import Path

try:
    from tkinter import BooleanVar, END, StringVar, messagebox
except ImportError:
    BooleanVar = None  # type: ignore[assignment]
    StringVar = None  # type: ignore[assignment]
    END = "end"  # type: ignore[assignment]
    messagebox = None  # type: ignore[assignment]

from eva_banker.follower.agent import FollowerAgent
from eva_banker.follower.config import (
    FollowerAccountConfig,
    FollowerAgentConfig,
    FollowerFleetConfig,
    load_follower_fleet_config,
    load_follower_config,
    save_follower_fleet_config,
    save_follower_config,
)
from eva_banker.follower.fleet import FollowerFleetManager


def run_ui(config_path: str | Path) -> None:
    """Lance l'interface graphique follower.

    Args:
        config_path (str | Path): Fichier de configuration local.

    Raises:
        RuntimeError: Si CustomTkinter n'est pas installe.
    """

    try:
        import customtkinter as ctk
    except ImportError as exc:
        raise RuntimeError("Installez customtkinter pour lancer l'interface follower.") from exc
    _require_tkinter()

    app = FollowerApp(ctk, Path(config_path))
    app.mainloop()


def run_fleet_ui(config_path: str | Path) -> None:
    """Lance l'interface graphique multi-comptes follower.

    Args:
        config_path (str | Path): Fichier de configuration de flotte.

    Raises:
        RuntimeError: Si CustomTkinter n'est pas installe.
    """

    try:
        import customtkinter as ctk
    except ImportError as exc:
        raise RuntimeError("Installez customtkinter pour lancer l'interface follower.") from exc
    _require_tkinter()

    app = FollowerFleetApp(ctk, Path(config_path))
    app.mainloop()


def _require_tkinter() -> None:
    """Verifie que Tkinter est disponible avant de construire l'UI.

    Raises:
        RuntimeError: Si Python n'inclut pas Tkinter dans l'environnement courant.
    """

    if StringVar is None or BooleanVar is None or messagebox is None:
        raise RuntimeError("Installez Tkinter pour lancer l'interface graphique follower.")


class FollowerApp:
    """Application CustomTkinter de supervision follower."""

    def __init__(self, ctk_module, config_path: Path) -> None:
        """Initialise la fenetre principale.

        Args:
            ctk_module: Module CustomTkinter charge dynamiquement.
            config_path (Path): Fichier de configuration a lire et ecrire.
        """

        self.ctk = ctk_module
        self.config_path = config_path
        self.config = load_follower_config(config_path)
        self.events: "queue.Queue[str]" = queue.Queue()
        self.agent: FollowerAgent | None = None
        self.agent_thread: threading.Thread | None = None
        self.agent_loop: asyncio.AbstractEventLoop | None = None

        self.ctk.set_appearance_mode("dark")
        self.ctk.set_default_color_theme("blue")
        self.root = self.ctk.CTk()
        self.root.title("THE HIVE - Follower Agent")
        self.root.geometry("1040x680")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build_layout()
        self._refresh_status()
        self._drain_events()

    def mainloop(self) -> None:
        """Demarre la boucle graphique."""

        self.root.mainloop()

    def _build_layout(self) -> None:
        """Construit les panneaux principaux."""

        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=2)
        self.root.grid_rowconfigure(1, weight=1)

        title = self.ctk.CTkLabel(
            self.root,
            text="THE HIVE Follower Agent",
            font=self.ctk.CTkFont(size=24, weight="bold"),
        )
        title.grid(row=0, column=0, columnspan=2, padx=20, pady=(18, 10), sticky="w")

        self._build_config_panel()
        self._build_runtime_panel()

    def _build_config_panel(self) -> None:
        """Construit le panneau de configuration client."""

        frame = self.ctk.CTkFrame(self.root)
        frame.grid(row=1, column=0, padx=(20, 10), pady=10, sticky="nsew")
        frame.grid_columnconfigure(1, weight=1)

        self.client_id = StringVar(value=self.config.client_id)
        self.account_label = StringVar(value=self.config.account_label)
        self.relay_url = StringVar(value=self.config.relay_base_url)
        self.terminal_path = StringVar(value=self.config.mt5_terminal_path)
        self.login = StringVar(value=str(self.config.mt5_login or ""))
        self.server = StringVar(value=self.config.mt5_server)
        self.allocation = StringVar(value=str(self.config.allocation_ratio))
        self.balance_reference = StringVar(value=str(self.config.balance_reference or ""))
        self.master_balance_reference = StringVar(value=str(self.config.master_balance_reference or ""))
        self.dry_run = BooleanVar(value=self.config.dry_run)
        self.mock_mt5 = BooleanVar(value=self.config.mock_mt5)

        fields = [
            ("Client", self.client_id),
            ("Compte", self.account_label),
            ("Relay", self.relay_url),
            ("Terminal MT5", self.terminal_path),
            ("Login", self.login),
            ("Serveur", self.server),
            ("Multiplicateur risque", self.allocation),
            ("Capital compte", self.balance_reference),
            ("Capital master", self.master_balance_reference),
        ]
        for index, (label, variable) in enumerate(fields):
            self.ctk.CTkLabel(frame, text=label).grid(row=index, column=0, padx=12, pady=8, sticky="w")
            self.ctk.CTkEntry(frame, textvariable=variable).grid(row=index, column=1, padx=12, pady=8, sticky="ew")

        offset = len(fields)
        self.ctk.CTkCheckBox(frame, text="Dry-run", variable=self.dry_run).grid(
            row=offset,
            column=0,
            padx=12,
            pady=8,
            sticky="w",
        )
        self.ctk.CTkCheckBox(frame, text="Mock MT5", variable=self.mock_mt5).grid(
            row=offset,
            column=1,
            padx=12,
            pady=8,
            sticky="w",
        )

        self.ctk.CTkButton(frame, text="Sauvegarder", command=self._save_config).grid(
            row=offset + 1,
            column=0,
            padx=12,
            pady=14,
            sticky="ew",
        )
        self.ctk.CTkButton(frame, text="Demarrer agent", command=self._start_agent).grid(
            row=offset + 1,
            column=1,
            padx=12,
            pady=14,
            sticky="ew",
        )
        self.ctk.CTkButton(frame, text="Pause / reprise", command=self._toggle_pause).grid(
            row=offset + 2,
            column=0,
            padx=12,
            pady=8,
            sticky="ew",
        )
        self.ctk.CTkButton(frame, text="Arreter", command=self._stop_agent).grid(
            row=offset + 2,
            column=1,
            padx=12,
            pady=8,
            sticky="ew",
        )

    def _build_runtime_panel(self) -> None:
        """Construit le panneau de logs et statut."""

        frame = self.ctk.CTkFrame(self.root)
        frame.grid(row=1, column=1, padx=(10, 20), pady=10, sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1)

        self.status_label = self.ctk.CTkLabel(frame, text="Statut: agent arrete", anchor="w")
        self.status_label.grid(row=0, column=0, padx=14, pady=12, sticky="ew")

        self.log_box = self.ctk.CTkTextbox(frame, wrap="word")
        self.log_box.grid(row=1, column=0, padx=14, pady=(0, 14), sticky="nsew")
        self._append_log("Interface follower prete.")

    def _save_config(self) -> None:
        """Sauvegarde les valeurs saisies."""

        try:
            current_payload = (
                self.config.model_dump()
                if hasattr(self.config, "model_dump")
                else self.config.dict()
            )
            self.config = FollowerAgentConfig(
                **{
                    **current_payload,
                    "client_id": self.client_id.get().strip(),
                    "account_label": self.account_label.get().strip(),
                    "relay_base_url": self.relay_url.get().strip(),
                    "mt5_terminal_path": self.terminal_path.get().strip(),
                    "mt5_login": int(self.login.get() or 0),
                    "mt5_server": self.server.get().strip(),
                    "allocation_ratio": float(self.allocation.get() or 1.0),
                    "balance_reference": float(self.balance_reference.get()) if self.balance_reference.get() else None,
                    "master_balance_reference": (
                        float(self.master_balance_reference.get())
                        if self.master_balance_reference.get()
                        else None
                    ),
                    "dry_run": bool(self.dry_run.get()),
                    "mock_mt5": bool(self.mock_mt5.get()),
                }
            )
        except ValueError as exc:
            messagebox.showerror("Configuration invalide", str(exc))
            return
        save_follower_config(self.config, self.config_path)
        self._append_log(f"Configuration sauvegardee: {self.config_path}")

    def _start_agent(self) -> None:
        """Demarre l'agent dans un thread dedie."""

        if self.agent_thread and self.agent_thread.is_alive():
            self._append_log("Agent deja actif.")
            return
        self._save_config()
        self.agent = FollowerAgent(self.config, event_callback=self.events.put)
        self.agent_thread = threading.Thread(target=self._run_agent_thread, daemon=True)
        self.agent_thread.start()
        self._append_log("Demarrage agent demande.")

    def _run_agent_thread(self) -> None:
        """Execute la boucle asyncio de l'agent."""

        self.agent_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.agent_loop)
        if self.agent is None:
            return
        self.agent_loop.run_until_complete(self.agent.run_forever())

    def _toggle_pause(self) -> None:
        """Inverse l'etat pause/reprise de l'agent."""

        if self.agent is None:
            return
        self.agent.paused = not self.agent.paused
        self._append_log("Agent en pause." if self.agent.paused else "Agent repris.")

    def _stop_agent(self) -> None:
        """Demande l'arret de l'agent."""

        if self.agent is not None:
            self.agent.running = False
            self._append_log("Arret agent demande.")

    def _refresh_status(self) -> None:
        """Rafraichit le statut visible."""

        if self.agent is None:
            text = "Statut: agent arrete"
        else:
            status = self.agent.get_status()
            text = (
                f"Statut: running={status.running} | pause={status.paused} | "
                f"relay={status.relay_connected} | mt5={status.mt5_connected} | "
                f"cmd={status.processed_commands} | liens={status.linked_positions}"
            )
        self.status_label.configure(text=text)
        self.root.after(1000, self._refresh_status)

    def _drain_events(self) -> None:
        """Affiche les evenements produits par l'agent."""

        while True:
            try:
                message = self.events.get_nowait()
            except queue.Empty:
                break
            self._append_log(message)
        self.root.after(300, self._drain_events)

    def _append_log(self, message: str) -> None:
        """Ajoute une ligne au panneau de logs."""

        self.log_box.insert(END, f"{message}\n")
        self.log_box.see(END)

    def _on_close(self) -> None:
        """Ferme la fenetre sans tuer brutalement le processus."""

        self._stop_agent()
        self.root.destroy()


class FollowerFleetApp:
    """Interface de gestion multi-comptes follower."""

    def __init__(self, ctk_module, config_path: Path) -> None:
        """Initialise la fenetre de flotte.

        Args:
            ctk_module: Module CustomTkinter charge dynamiquement.
            config_path (Path): Fichier JSON de flotte.
        """

        self.ctk = ctk_module
        self.config_path = config_path
        self.config = load_follower_fleet_config(config_path)
        self.selected_client_id = self.config.accounts[0].client_id if self.config.accounts else ""
        self.events: "queue.Queue[str]" = queue.Queue()
        self.manager: FollowerFleetManager | None = None
        self.manager_thread: threading.Thread | None = None
        self.manager_loop: asyncio.AbstractEventLoop | None = None
        self.account_buttons: dict[str, object] = {}

        self.ctk.set_appearance_mode("dark")
        self.ctk.set_default_color_theme("green")
        self.root = self.ctk.CTk()
        self.root.title("THE HIVE - Follower Fleet Manager")
        self.root.geometry("1220x760")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build_layout()
        self._load_selected_account()
        self._refresh_accounts()
        self._refresh_status()
        self._drain_events()

    def mainloop(self) -> None:
        """Demarre la boucle graphique."""

        self.root.mainloop()

    def _build_layout(self) -> None:
        """Construit la grille principale de l'interface."""

        self.root.grid_columnconfigure(0, weight=0)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_columnconfigure(2, weight=1)
        self.root.grid_rowconfigure(1, weight=1)

        title = self.ctk.CTkLabel(
            self.root,
            text="THE HIVE Follower Fleet Manager",
            font=self.ctk.CTkFont(size=24, weight="bold"),
        )
        title.grid(row=0, column=0, columnspan=3, padx=20, pady=(18, 10), sticky="w")

        self._build_accounts_panel()
        self._build_editor_panel()
        self._build_runtime_panel()

    def _build_accounts_panel(self) -> None:
        """Construit la liste des comptes followers."""

        frame = self.ctk.CTkFrame(self.root, width=280)
        frame.grid(row=1, column=0, padx=(20, 10), pady=10, sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1)

        self.ctk.CTkLabel(frame, text="Comptes").grid(row=0, column=0, padx=12, pady=10, sticky="w")
        self.accounts_list = self.ctk.CTkScrollableFrame(frame, width=250)
        self.accounts_list.grid(row=1, column=0, padx=12, pady=8, sticky="nsew")

        self.ctk.CTkButton(frame, text="Ajouter", command=self._add_account).grid(
            row=2,
            column=0,
            padx=12,
            pady=(10, 4),
            sticky="ew",
        )
        self.ctk.CTkButton(frame, text="Supprimer", command=self._delete_account).grid(
            row=3,
            column=0,
            padx=12,
            pady=4,
            sticky="ew",
        )
        self.ctk.CTkButton(frame, text="Start comptes actifs", command=self._start_fleet).grid(
            row=4,
            column=0,
            padx=12,
            pady=(14, 4),
            sticky="ew",
        )
        self.ctk.CTkButton(frame, text="Stop tout", command=self._stop_fleet).grid(
            row=5,
            column=0,
            padx=12,
            pady=(4, 12),
            sticky="ew",
        )

    def _build_editor_panel(self) -> None:
        """Construit le formulaire du compte selectionne."""

        frame = self.ctk.CTkFrame(self.root)
        frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
        frame.grid_columnconfigure(1, weight=1)

        self.enabled = BooleanVar(value=True)
        self.client_id = StringVar()
        self.account_label = StringVar()
        self.relay_url = StringVar()
        self.api_token = StringVar()
        self.terminal_path = StringVar()
        self.login = StringVar()
        self.password = StringVar()
        self.server = StringVar()
        self.allocation = StringVar()
        self.balance_reference = StringVar()
        self.master_balance_reference = StringVar()
        self.dry_run = BooleanVar(value=True)
        self.mock_mt5 = BooleanVar(value=True)

        fields = [
            ("Client ID", self.client_id, False),
            ("Nom compte", self.account_label, False),
            ("Relay URL", self.relay_url, False),
            ("Token API", self.api_token, True),
            ("Terminal MT5", self.terminal_path, False),
            ("Login", self.login, False),
            ("Password", self.password, True),
            ("Serveur", self.server, False),
            ("Multiplicateur risque", self.allocation, False),
            ("Capital compte", self.balance_reference, False),
            ("Capital master", self.master_balance_reference, False),
        ]
        self.ctk.CTkCheckBox(frame, text="Compte actif", variable=self.enabled).grid(
            row=0,
            column=0,
            columnspan=2,
            padx=12,
            pady=8,
            sticky="w",
        )
        for index, (label, variable, masked) in enumerate(fields, start=1):
            self.ctk.CTkLabel(frame, text=label).grid(row=index, column=0, padx=12, pady=7, sticky="w")
            self.ctk.CTkEntry(frame, textvariable=variable, show="*" if masked else "").grid(
                row=index,
                column=1,
                padx=12,
                pady=7,
                sticky="ew",
            )

        offset = len(fields) + 1
        self.ctk.CTkCheckBox(frame, text="Dry-run", variable=self.dry_run).grid(
            row=offset,
            column=0,
            padx=12,
            pady=8,
            sticky="w",
        )
        self.ctk.CTkCheckBox(frame, text="Mock MT5", variable=self.mock_mt5).grid(
            row=offset,
            column=1,
            padx=12,
            pady=8,
            sticky="w",
        )
        self.ctk.CTkButton(frame, text="Sauvegarder compte", command=self._save_selected_account).grid(
            row=offset + 1,
            column=0,
            padx=12,
            pady=(16, 6),
            sticky="ew",
        )
        self.ctk.CTkButton(frame, text="Start compte", command=self._start_selected_account).grid(
            row=offset + 1,
            column=1,
            padx=12,
            pady=(16, 6),
            sticky="ew",
        )
        self.ctk.CTkButton(frame, text="Stop compte", command=self._stop_selected_account).grid(
            row=offset + 2,
            column=0,
            columnspan=2,
            padx=12,
            pady=6,
            sticky="ew",
        )

    def _build_runtime_panel(self) -> None:
        """Construit le panneau de statut et logs."""

        frame = self.ctk.CTkFrame(self.root)
        frame.grid(row=1, column=2, padx=(10, 20), pady=10, sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(2, weight=1)

        self.status_label = self.ctk.CTkLabel(frame, text="Flotte arretee", anchor="w", justify="left")
        self.status_label.grid(row=0, column=0, padx=14, pady=12, sticky="ew")
        self.config_label = self.ctk.CTkLabel(frame, text=f"Config: {self.config_path}", anchor="w")
        self.config_label.grid(row=1, column=0, padx=14, pady=(0, 8), sticky="ew")

        self.log_box = self.ctk.CTkTextbox(frame, wrap="word")
        self.log_box.grid(row=2, column=0, padx=14, pady=(0, 14), sticky="nsew")
        self._append_log("Interface fleet prete.")

    def _refresh_accounts(self) -> None:
        """Reconstruit la liste visible des comptes."""

        for widget in self.accounts_list.winfo_children():
            widget.destroy()
        self.account_buttons.clear()
        for index, account in enumerate(self.config.accounts):
            marker = "ON" if account.enabled else "OFF"
            label = f"{marker} | {account.account_label}\n{account.client_id}"
            button = self.ctk.CTkButton(
                self.accounts_list,
                text=label,
                anchor="w",
                command=lambda client_id=account.client_id: self._select_account(client_id),
            )
            button.grid(row=index, column=0, padx=8, pady=6, sticky="ew")
            self.account_buttons[account.client_id] = button

    def _select_account(self, client_id: str) -> None:
        """Selectionne un compte dans l'editeur."""

        self._save_selected_account(silent=True)
        self.selected_client_id = client_id
        self._load_selected_account()

    def _load_selected_account(self) -> None:
        """Charge le compte selectionne dans le formulaire."""

        account = self._selected_account()
        if account is None:
            return
        self.enabled.set(account.enabled)
        self.client_id.set(account.client_id)
        self.account_label.set(account.account_label)
        self.relay_url.set(account.relay_base_url)
        self.api_token.set(account.api_token)
        self.terminal_path.set(account.mt5_terminal_path)
        self.login.set(str(account.mt5_login or ""))
        self.password.set(account.mt5_password)
        self.server.set(account.mt5_server)
        self.allocation.set(str(account.allocation_ratio))
        self.balance_reference.set(str(account.balance_reference or ""))
        self.master_balance_reference.set(str(account.master_balance_reference or ""))
        self.dry_run.set(account.dry_run)
        self.mock_mt5.set(account.mock_mt5)

    def _selected_account(self) -> FollowerAccountConfig | None:
        """Retourne le compte selectionne."""

        for account in self.config.accounts:
            if account.client_id == self.selected_client_id:
                return account
        return self.config.accounts[0] if self.config.accounts else None

    def _save_selected_account(self, silent: bool = False) -> None:
        """Sauvegarde le formulaire dans la configuration de flotte."""

        account = self._selected_account()
        if account is None:
            return
        old_client_id = account.client_id
        try:
            payload = account.model_dump() if hasattr(account, "model_dump") else account.dict()
            updated = FollowerAccountConfig(
                **{
                    **payload,
                    "enabled": bool(self.enabled.get()),
                    "client_id": self.client_id.get().strip(),
                    "account_label": self.account_label.get().strip(),
                    "relay_base_url": self.relay_url.get().strip(),
                    "api_token": self.api_token.get(),
                    "mt5_terminal_path": self.terminal_path.get().strip(),
                    "mt5_login": int(self.login.get() or 0),
                    "mt5_password": self.password.get(),
                    "mt5_server": self.server.get().strip(),
                    "allocation_ratio": float(self.allocation.get() or 1.0),
                    "balance_reference": float(self.balance_reference.get()) if self.balance_reference.get() else None,
                    "master_balance_reference": (
                        float(self.master_balance_reference.get())
                        if self.master_balance_reference.get()
                        else None
                    ),
                    "dry_run": bool(self.dry_run.get()),
                    "mock_mt5": bool(self.mock_mt5.get()),
                }
            )
        except ValueError as exc:
            messagebox.showerror("Configuration invalide", str(exc))
            return
        for index, current in enumerate(self.config.accounts):
            if current.client_id == old_client_id:
                self.config.accounts[index] = updated
                break
        self.selected_client_id = updated.client_id
        save_follower_fleet_config(self.config, self.config_path)
        self._refresh_accounts()
        if not silent:
            self._append_log(f"Compte sauvegarde: {updated.account_label}")

    def _add_account(self) -> None:
        """Ajoute un compte vide dans la flotte."""

        next_index = len(self.config.accounts) + 1
        client_id = f"client-{next_index}"
        account = FollowerAccountConfig(
            client_id=client_id,
            account_label=f"Compte {next_index}",
            state_path=f"data/follower_agent/{client_id}.state.json",
            log_path=f"logs/follower_agent_{client_id}.log",
        )
        self.config.accounts.append(account)
        self.selected_client_id = client_id
        save_follower_fleet_config(self.config, self.config_path)
        self._refresh_accounts()
        self._load_selected_account()
        self._append_log(f"Compte ajoute: {client_id}")

    def _delete_account(self) -> None:
        """Supprime le compte selectionne de la flotte."""

        account = self._selected_account()
        if account is None:
            return
        if self.manager and account.client_id in self.manager.agents:
            messagebox.showerror("Compte actif", "Arretez le compte avant suppression.")
            return
        self.config.accounts = [item for item in self.config.accounts if item.client_id != account.client_id]
        self.selected_client_id = self.config.accounts[0].client_id if self.config.accounts else ""
        save_follower_fleet_config(self.config, self.config_path)
        self._refresh_accounts()
        self._load_selected_account()
        self._append_log(f"Compte supprime: {account.client_id}")

    def _start_fleet(self) -> None:
        """Demarre la flotte dans un thread dedie."""

        if self.manager_thread and self.manager_thread.is_alive():
            self._append_log("Flotte deja active.")
            return
        self._save_selected_account(silent=True)
        self.manager = FollowerFleetManager(self.config, event_callback=self.events.put)
        self.manager_thread = threading.Thread(target=self._run_manager_thread, daemon=True)
        self.manager_thread.start()
        self._append_log("Demarrage flotte demande.")

    def _run_manager_thread(self) -> None:
        """Execute la boucle asyncio de la flotte."""

        self.manager_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.manager_loop)
        if self.manager is None:
            return
        self.manager_loop.run_until_complete(self.manager.run_forever())

    def _stop_fleet(self) -> None:
        """Demande l'arret de toute la flotte."""

        if self.manager is not None:
            self.manager.running = False
            self._append_log("Arret flotte demande.")

    def _start_selected_account(self) -> None:
        """Demarre uniquement le compte selectionne."""

        self._save_selected_account(silent=True)
        if self.manager is None or self.manager_loop is None:
            self._start_fleet()
            return
        account = self._selected_account()
        if account is None:
            return
        asyncio.run_coroutine_threadsafe(self.manager.start_account(account.client_id), self.manager_loop)
        self._append_log(f"Demarrage compte demande: {account.account_label}")

    def _stop_selected_account(self) -> None:
        """Arrete uniquement le compte selectionne."""

        if self.manager is None or self.manager_loop is None:
            return
        account = self._selected_account()
        if account is None:
            return
        asyncio.run_coroutine_threadsafe(self.manager.stop_account(account.client_id), self.manager_loop)
        self._append_log(f"Arret compte demande: {account.account_label}")

    def _refresh_status(self) -> None:
        """Rafraichit le statut de tous les comptes."""

        if self.manager is None:
            active_count = 0
            lines = ["Flotte arretee"]
        else:
            statuses = self.manager.get_statuses()
            active_count = sum(1 for status in statuses if status.running)
            lines = [
                f"{status.account_label}: run={status.running} mt5={status.mt5_connected} "
                f"relay={status.relay_connected} cmd={status.processed_commands} liens={status.linked_positions}"
                for status in statuses
            ]
        self.status_label.configure(text=f"Comptes actifs: {active_count}\n" + "\n".join(lines[:8]))
        self.root.after(1000, self._refresh_status)

    def _drain_events(self) -> None:
        """Affiche les evenements produits par la flotte."""

        while True:
            try:
                message = self.events.get_nowait()
            except queue.Empty:
                break
            self._append_log(message)
        self.root.after(300, self._drain_events)

    def _append_log(self, message: str) -> None:
        """Ajoute une ligne au panneau de logs."""

        self.log_box.insert(END, f"{message}\n")
        self.log_box.see(END)

    def _on_close(self) -> None:
        """Ferme la fenetre apres demande d'arret."""

        self._stop_fleet()
        self.root.destroy()


def main() -> None:
    """Point d'entree direct de l'interface."""

    run_fleet_ui(Path("data/follower_agent/fleet.config.json"))


if __name__ == "__main__":
    main()
