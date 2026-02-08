"""
Sentiment & Security Engine — THE HIVE
Analyse de sentiment basique + surveillance d'intégrité système.
"""

import asyncio
import hashlib
import logging
import os
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


class SecurityEngine:
    """
    Moteur de sécurité pour The Sentinel.
    - Vérifie l'intégrité des fichiers critiques (Constitution, Kernel)
    - Surveille les connexions réseau suspectes
    - Maintient un journal d'alertes
    """

    def __init__(self):
        self.alerts: list[dict[str, Any]] = []
        self.integrity_hashes: dict[str, str] = {}
        self.max_alerts = 100

    def _hash_file(self, filepath: str) -> str | None:
        """Calcule le hash SHA256 d'un fichier"""
        try:
            with open(filepath, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception:
            return None

    async def check_integrity(self, files: list[str] | None = None) -> list[dict[str, Any]]:
        """
        Vérifie l'intégrité des fichiers critiques.
        Retourne les alertes si un fichier a changé.
        """
        if files is None:
            files = [
                "mnt/tablet/constitution.toml",
                "src/eva-kernel/src/main.rs",
                "src/eva-kernel/src/laws.rs",
            ]

        results = []
        for filepath in files:
            current_hash = self._hash_file(filepath)
            if current_hash is None:
                results.append({
                    "file": filepath,
                    "status": "NOT_FOUND",
                    "severity": "warning"
                })
                continue

            if filepath in self.integrity_hashes:
                if self.integrity_hashes[filepath] != current_hash:
                    alert = {
                        "id": f"integrity-{len(self.alerts):04d}",
                        "type": "INTEGRITY_VIOLATION",
                        "severity": "critical",
                        "message": f"Fichier modifié: {filepath}",
                        "file": filepath,
                        "old_hash": self.integrity_hashes[filepath][:16],
                        "new_hash": current_hash[:16],
                        "timestamp": datetime.now().isoformat()
                    }
                    self._add_alert(alert)
                    results.append({"file": filepath, "status": "MODIFIED", "severity": "critical"})
                else:
                    results.append({"file": filepath, "status": "OK", "severity": "info"})
            else:
                results.append({"file": filepath, "status": "BASELINE_SET", "severity": "info"})

            self.integrity_hashes[filepath] = current_hash

        return results

    async def check_network(self) -> list[dict[str, Any]]:
        """Vérifie les connexions réseau (basique)"""
        alerts = []
        try:
            import psutil
            connections = psutil.net_connections(kind='inet')
            
            # Compter les connexions par état
            states = {}
            for conn in connections:
                state = conn.status
                states[state] = states.get(state, 0) + 1
            
            # Alerte si trop de connexions ESTABLISHED
            established = states.get("ESTABLISHED", 0)
            if established > 200:
                alert = {
                    "id": f"network-{len(self.alerts):04d}",
                    "type": "NETWORK_ANOMALY",
                    "severity": "warning",
                    "message": f"Nombre élevé de connexions: {established}",
                    "timestamp": datetime.now().isoformat()
                }
                self._add_alert(alert)
                alerts.append(alert)
        except Exception as e:
            logger.error(f"Erreur surveillance réseau: {e}")

        return alerts

    def _add_alert(self, alert: dict[str, Any]):
        """Ajoute une alerte et maintient la taille du journal"""
        self.alerts.append(alert)
        if len(self.alerts) > self.max_alerts:
            self.alerts = self.alerts[-self.max_alerts:]

    def get_alerts(self, limit: int = 20) -> list[dict[str, Any]]:
        """Retourne les alertes récentes"""
        return self.alerts[-limit:]

    async def run_full_scan(self) -> dict[str, Any]:
        """Exécute un scan de sécurité complet"""
        integrity = await self.check_integrity()
        network = await self.check_network()

        return {
            "scan_time": datetime.now().isoformat(),
            "integrity_checks": integrity,
            "network_alerts": network,
            "total_alerts": len(self.alerts),
            "status": "secure" if not any(
                a["severity"] == "critical" for a in integrity
            ) else "compromised"
        }
