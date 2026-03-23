"""
OpenClaw RLM Benchmark — Module de Self-Benchmarking
Part of Sovereign Stack V3.0 — Sprint 9 (RLM Auto-Patching)

Ce module permet à THE HIVE d'évaluer l'impact d'un patch sur le système.
Il capture des métriques de santé "Avant" (Pre-Patch) et "Après" (Post-Patch),
puis décide s'il faut approuver la "mutation" ou déclencher un rollback.

Métriques observées via les composants Core proxy :
- Santé globale (Status 200 vs 500)
- Métriques système Sentinel (CPU, RAM, Température GPU)
- Stabilité financière Banker (Drawdown, WinRate, Exposition)
- Télémétrie Core (Taux d'erreurs global)

Usage :
    benchmark = RLMBenchmark()
    pre_metrics = await benchmark.capture_metrics()
    # ... appliquer le patch, redémarrer le service ...
    # Attendre la stabilisation (ex: 5 minutes)
    post_metrics = await benchmark.capture_metrics()
    
    analysis = benchmark.compare(pre_metrics, post_metrics)
    if not analysis.is_healthy:
        # Trigger Rollback
"""

import asyncio
import logging
import httpx
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """Capture d'un état global du système à l'instant T."""
    timestamp: datetime = field(default_factory=datetime.now)
    core_errors: int = 0
    cpu_usage: float = 0.0
    ram_usage: float = 0.0
    gpu_temp: float = 0.0
    banker_drawdown: float = 0.0
    banker_winrate: float = 0.0
    offline_services: int = 0

@dataclass
class BenchmarkAnalysis:
    """Résultat de la comparaison Pre/Post Patch."""
    is_healthy: bool = True
    degradation_score: float = 0.0 # 0.0 = parfait, > 5.0 = critique (rollback)
    reasons: list[str] = field(default_factory=list)
    summary: str = "Benchmark Analysis Complete"

class RLMBenchmark:
    """Evaluateur d'intelligence et de stabilité post-patch."""
    
    def __init__(self, core_url: str = "http://localhost:8000"):
        """
        Initialise le Benchmarker connectable à l'API Core.
        Args:
            core_url: URL de l'Orchestrateur Core (qui sert de proxy de metrics).
        """
        self.core_url = core_url
        self.client = httpx.AsyncClient(timeout=10.0)

    async def capture_metrics(self) -> Optional[SystemMetrics]:
        """
        Capture une empreinte (snapshot) des métriques actuelles.
        Si l'API est injoignable, on retourne None (ce qui est critique post-patch).
        """
        try:
            # 1. Telemetry API (Core errors)
            telemetry_res = await self.client.get(f"{self.core_url}/telemetry")
            telemetry = telemetry_res.json() if telemetry_res.status_code == 200 else {}
            
            # 2. System API (Hardware usage proxying Sentinel)
            system_res = await self.client.get(f"{self.core_url}/system/status")
            system = system_res.json() if system_res.status_code == 200 else {}
            hardware = system.get("metrics", {})
            
            # 3. Agent Status (Offline count)
            agents_res = await self.client.get(f"{self.core_url}/agents/status")
            agents = agents_res.json() if agents_res.status_code == 200 else {}
            offline_count = sum(1 for agent, data in agents.items() if data.get("status") in ["offline", "stale"])
            
            # 4. Trading Status (Drawdown proxying Banker)
            trading_res = await self.client.get(f"{self.core_url}/trading/status")
            trading = trading_res.json() if trading_res.status_code == 200 else {}
            risk = trading.get("risk", {})
            drawdown = float(risk.get("daily_drawdown_percent", 0.0))
            
            return SystemMetrics(
                core_errors=telemetry.get("errors_total", 0),
                cpu_usage=hardware.get("cpu", {}).get("percent", 0.0),
                ram_usage=hardware.get("ram", {}).get("percent", 0.0),
                gpu_temp=hardware.get("gpu", [{}])[0].get("temperature", 0.0) if hardware.get("gpu") else 0.0,
                banker_drawdown=drawdown,
                offline_services=offline_count
            )
            
        except Exception as e:
            logger.error(f"[RLM:Benchmark] Impossible de capturer les métriques: {e}")
            return None # Snapshot failure indicates severe instability

    def compare(self, pre: SystemMetrics, post: Optional[SystemMetrics]) -> BenchmarkAnalysis:
        """
        Compare les métriques Avant/Après le patch.
        Un score de dégradation élevé déclenche un rollback.
        
        Args:
            pre: Métriques avant l'application du patch.
            post: Métriques après l'application et la période de grâce.
        """
        analysis = BenchmarkAnalysis()
        
        # Cas 1 : Crash critique (API injoignable post-patch)
        if post is None:
            analysis.is_healthy = False
            analysis.degradation_score = 10.0
            analysis.reasons.append("CRITICAL: Le système est totalement injoignable post-patch (Crash Fatal).")
            analysis.summary = "Échec Catastrophique du Patch"
            return analysis
            
        # Comparaison différentielle
        error_diff = post.core_errors - pre.core_errors
        offline_diff = post.offline_services - pre.offline_services
        dd_diff = post.banker_drawdown - pre.banker_drawdown
        cpu_diff = post.cpu_usage - pre.cpu_usage
        
        # Heuristiques de punition (Score)
        if offline_diff > 0:
            analysis.degradation_score += (offline_diff * 3.0)
            analysis.reasons.append(f"Services tombés hors-ligne: +{offline_diff}")
            
        if error_diff > 10:
            analysis.degradation_score += ((error_diff / 10) * 1.5)
            analysis.reasons.append(f"Spike d'erreurs Core (Traceback explosion): +{error_diff} erreurs.")
            
        if dd_diff > 1.0: # 1% de drawdown en plus post patch
            analysis.degradation_score += (dd_diff * 2.0)
            analysis.reasons.append(f"Perte de capital brusque: {dd_diff}% Drawdown ajouté.")
            
        if cpu_diff > 25.0: # Spike CPU permanent suspect
            analysis.degradation_score += 1.0
            analysis.reasons.append(f"Surcharge CPU détectée: +{cpu_diff}% d'usage.")
            
        # Décision de Rollback
        if analysis.degradation_score >= 5.0:
            analysis.is_healthy = False
            analysis.summary = f"Qualité dégradée (Score {analysis.degradation_score:.1f}). ROLLBACK REQUIS."
        else:
            analysis.is_healthy = True
            analysis.summary = f"Patch Stable (Score {analysis.degradation_score:.1f}). MUTATION APPROUVÉE."
            
        return analysis

    async def close(self):
        """Nettoyage des ressources du client HTTP."""
        await self.client.aclose()
