import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

class SovereignAuditor:
    """
    Générateur de Rapports d'Audit Souverains.
    Synthétise la performance, le risque et la sincérité pour le Maître.
    """
    def __init__(self, output_dir: str = "Audits"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    async def generate_daily_audit(self, stats: Dict[str, Any]):
        """
        Génère un rapport d'audit complet en Markdown (convertible en PDF).
        """
        date_str = datetime.now().strftime("%Y-%m-%d")
        report_path = self.output_dir / f"Audit_Souverain_{date_str}.md"
        
        content = f"""# 🐝 EVA SOVEREIGN AUDIT - {date_str}

## 📊 Performance Financière
- **PnL Journalier** : {stats.get('pnl', '0.00')} €
- **Volume Traité** : {stats.get('volume', '0')} lots
- **Win Rate** : {stats.get('win_rate', '0')}%

## 🛡️ Conformité Constitutionnelle (Loi 2)
- **Drawdown Max** : {stats.get('max_dd', '0')}% (Limite: 4.0%)
- **Statut Risque** : {"✅ CONFORME" if stats.get('risk_ok') else "⚠️ ALERTE"}
- **Interceptions Kernel** : {stats.get('kernel_blocks', '0')}

## 🧠 Sincérité Cognitive (Linear Probes)
- **Score de Sincérité LLM** : {stats.get('sincerity_avg', '100')}%
- **Hallucinations Détectées** : {stats.get('hallucinations', '0')}
- **Statut** : {"✨ SINCÈRE" if stats.get('sincerity_avg', 0) > 90 else "☢️ RISQUE COGNITIF"}

## 🐝 Swarm Health (Self-Healing)
- **Experts actifs** : {stats.get('active_experts', '0')}
- **Drones auto-réparés** : {stats.get('healed_drones', '0')}

---
*Rapport généré automatiquement par l'Expert Researcher - THE HIVE Sovereign OS.*
"""
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(content)
            
        logger.info(f"Daily Audit generated: {report_path}")
        return str(report_path)
