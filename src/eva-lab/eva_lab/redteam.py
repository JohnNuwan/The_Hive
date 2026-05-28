"""
Red Team — Analyse adversarial du champion live.

Charge les trades fermes (hard negatives) du LiveTradeReviewService,
genere des scenarios de marche adverses synthetiques bases sur les
echecs reels, et evalue le champion contre ces scenarios.

Peut etre lance en autonome ou integre a la nightly stack.
"""

from __future__ import annotations

import json
import logging
import os
import random
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RedTeamScenario:
    """Un scenario de marche adverse genere a partir de donnees reelles.

    Attributes:
        name: Nom du scenario.
        symbol: Symbole concerne.
        failure_type: Type d'echec (hard_stop, liquidity_trap, etc.).
        prices: Prix synthetiques generes.
        indicators: Indicateurs associes.
        severity: Niveau de severite (0.0 a 1.0).
        source_trade_id: Ticket du trade reel a l'origine du scenario.
    """
    name: str
    symbol: str
    failure_type: str
    prices: list[float] = field(default_factory=list)
    indicators: dict[str, list[float]] = field(default_factory=dict)
    severity: float = 0.5
    source_trade_id: int | None = None


@dataclass
class RedTeamReport:
    """Rapport complet d'une session Red Team.

    Attributes:
        generated_at: Horodatage ISO.
        champion_id: Identifiant du champion evalue.
        total_trades_analyzed: Nombre de trades live analyses.
        hard_negatives_found: Nombre de hard negatifs detectes.
        scenarios_generated: Nombre de scenarios adverses generes.
        weaknesses: Liste des faiblesses detectees.
        failure_type_distribution: Repartition des types d'echec.
        symbol_weakness_score: Score de faiblesse par symbole.
        champion_survival_score: Score global de robustesse (0-100).
    """
    generated_at: str = ""
    champion_id: str = ""
    total_trades_analyzed: int = 0
    hard_negatives_found: int = 0
    scenarios_generated: int = 0
    weaknesses: list[dict[str, Any]] = field(default_factory=list)
    failure_type_distribution: dict[str, int] = field(default_factory=dict)
    symbol_weakness_score: dict[str, float] = field(default_factory=dict)
    champion_survival_score: float = 100.0


class RedTeam:
    """Analyse les trades live et genere des scenarios adverses pour tester
    la robustesse du champion.

    Le Red Team exploite les hard negatives collectes par le
    LiveTradeReviewService pour creer des scenarios synthetiques
    qui reproduisent les conditions de marche ayant cause des pertes.

    Args:
        trade_review_dir: Repertoire des donnees LiveTradeReview.
        champion_id: Identifiant du champion a evaluer.
        seed: Graine de randomisation pour la reproductibilite.
    """

    FAILURE_TYPES = [
        "hard_stop_exit",
        "liquidity_trap",
        "bad_runner_exit",
        "bad_pyramid_exit",
        "range_entry_loss",
        "bad_trade",
    ]

    def __init__(
        self,
        trade_review_dir: str = "data/live_trade_reviews",
        champion_id: str | None = None,
        seed: int = 42,
    ):
        self.trade_review_dir = Path(trade_review_dir)
        self.champion_id = champion_id or os.getenv("REDTEAM_CHAMPION_ID", "unknown")
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self._records: list[dict[str, Any]] = []
        self._hard_negatives: list[dict[str, Any]] = []

    def load_trade_data(self, window_days: int = 30) -> int:
        """Charge les trades live de la fenetre donnee.

        Args:
            window_days: Fenetre de retrospection en jours.

        Returns:
            int: Nombre de trades charges.
        """
        cutoff = datetime.now() - timedelta(days=window_days)
        records: list[dict[str, Any]] = []
        pattern = "live_trade_review_*.jsonl"
        for path in sorted(self.trade_review_dir.glob(pattern)):
            try:
                with path.open("r", encoding="utf-8") as handle:
                    for raw_line in handle:
                        line = raw_line.strip()
                        if not line:
                            continue
                        payload = json.loads(line)
                        closed_at = payload.get("closed_at", "")
                        if closed_at:
                            try:
                                if datetime.fromisoformat(closed_at) < cutoff:
                                    continue
                            except ValueError:
                                pass
                        records.append(payload)
            except Exception as exc:
                logger.warning("RedTeam: echec de lecture %s: %s", path, exc)

        self._records = records
        self._hard_negatives = [
            rec for rec in records
            if rec.get("trade_bucket") == "hard_negative"
        ]
        logger.info(
            "RedTeam: %d trades charges (%d hard negatifs)",
            len(records), len(self._hard_negatives),
        )
        return len(records)

    def analyze_failure_distribution(self) -> dict[str, int]:
        """Analyse la repartition des types d'echec parmi les hard negatifs.

        Returns:
            dict[str, int]: Comptage par type d'echec.
        """
        counter: Counter[str] = Counter()
        for rec in self._hard_negatives:
            quality = rec.get("quality_label", "bad_trade")
            counter[quality] += 1
        return dict(counter)

    def compute_symbol_fragility(self) -> dict[str, float]:
        """Calcule un score de fragilite (0=sain, 1=tres fragile) par symbole.

        Returns:
            dict[str, float]: Score par symbole.
        """
        symbol_stats: dict[str, dict[str, int]] = {}
        for rec in self._records:
            sym = rec.get("symbol", "UNKNOWN")
            bucket = rec.get("trade_bucket", "neutral")
            entry = symbol_stats.setdefault(sym, {"total": 0, "hard_neg": 0})
            entry["total"] += 1
            if bucket == "hard_negative":
                entry["hard_neg"] += 1

        fragility: dict[str, float] = {}
        for sym, stats in symbol_stats.items():
            if stats["total"] < 3:
                fragility[sym] = 0.5
            else:
                fragility[sym] = stats["hard_neg"] / max(1, stats["total"])
        return fragility

    def generate_scenarios(self, max_scenarios: int = 20) -> list[RedTeamScenario]:
        """Genere des scenarios adverses a partir des hard negatifs reels.

        Pour chaque hard negatif, cree un scenario synthetique qui
        reproduit les conditions d'echec avec une perturbation aleatoire.

        Args:
            max_scenarios: Nombre maximum de scenarios a generer.

        Returns:
            list[RedTeamScenario]: Scenarios generes.
        """
        if not self._hard_negatives:
            logger.warning("RedTeam: aucun hard negatif pour generer des scenarios.")
            return []

        self.rng.shuffle(self._hard_negatives)
        scenarios: list[RedTeamScenario] = []
        used_types: set[str] = set()

        for rec in self._hard_negatives:
            if len(scenarios) >= max_scenarios:
                break

            symbol = rec.get("symbol", "UNKNOWN")
            failure_type = rec.get("quality_label", "bad_trade")
            close_reason = rec.get("close_reason", "unknown")
            ticket = rec.get("ticket")

            metadata: dict[str, Any] = rec.get("metadata", {})
            indicators: dict[str, Any] = metadata.get("indicators", {})

            base_price = float(metadata.get("entry_price", 100.0) or 100.0)
            n_points = 200
            prices = self._generate_adversarial_prices(
                base_price, failure_type, n_points,
            )

            scenario_indicators = self._generate_indicators(
                prices, indicators, failure_type,
            )

            scenario = RedTeamScenario(
                name=f"{failure_type}_{symbol}_{ticket or self.rng.randint(1000, 9999)}",
                symbol=symbol,
                failure_type=failure_type,
                prices=prices,
                indicators=scenario_indicators,
                severity=self._compute_severity(failure_type, close_reason),
                source_trade_id=ticket,
            )
            scenarios.append(scenario)
            used_types.add(failure_type)

        if len(scenarios) < max_scenarios:
            for ftype in self.FAILURE_TYPES:
                if ftype not in used_types and len(scenarios) < max_scenarios:
                    scenario = self._generate_generic_scenario(ftype)
                    if scenario:
                        scenarios.append(scenario)

        logger.info("RedTeam: %d scenarios generes", len(scenarios))
        return scenarios

    def produce_report(self, scenarios: list[RedTeamScenario]) -> RedTeamReport:
        """Produit le rapport final de la session Red Team.

        Args:
            scenarios: Liste des scenarios generes.

        Returns:
            RedTeamReport: Rapport complet.
        """
        failure_dist = self.analyze_failure_distribution()
        symbol_fragility = self.compute_symbol_fragility()

        weaknesses: list[dict[str, Any]] = []
        for sym, frag in sorted(symbol_fragility.items(), key=lambda x: -x[1]):
            if frag >= 0.5:
                weaknesses.append({
                    "symbol": sym,
                    "fragility_score": round(frag, 3),
                    "trades_analyzed": sum(
                        1 for r in self._records if r.get("symbol") == sym
                    ),
                    "hard_negatives": sum(
                        1 for r in self._hard_negatives if r.get("symbol") == sym
                    ),
                })

        total = max(1, len(self._records))
        hard_neg_ratio = len(self._hard_negatives) / total
        avg_fragility = (
            sum(symbol_fragility.values()) / max(1, len(symbol_fragility))
        )
        survival_score = max(
            0.0, 100.0 - (
                hard_neg_ratio * 60.0
                + avg_fragility * 30.0
                + max(0, hard_neg_ratio - 0.3) * 50.0
            )
        )

        return RedTeamReport(
            generated_at=datetime.now().isoformat(),
            champion_id=self.champion_id,
            total_trades_analyzed=len(self._records),
            hard_negatives_found=len(self._hard_negatives),
            scenarios_generated=len(scenarios),
            weaknesses=weaknesses,
            failure_type_distribution=failure_dist,
            symbol_weakness_score={
                sym: round(1.0 - frag, 3)
                for sym, frag in symbol_fragility.items()
            },
            champion_survival_score=round(min(100.0, max(0.0, survival_score)), 1),
        )

    def save_report(self, report: RedTeamReport, output_dir: str = "data/redteam") -> str:
        """Persiste le rapport Red Team sur le disque.

        Args:
            report: Rapport a sauvegarder.
            output_dir: Repertoire de sortie.

        Returns:
            str: Chemin du fichier ecrit.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        file_path = output_path / f"redteam_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_json = json.dumps(asdict(report), ensure_ascii=False, indent=2)
        file_path.write_text(
            report_json,
            encoding="utf-8",
        )
        logger.info("RedTeam: rapport sauvegarde dans %s", file_path)

        # On ecrit aussi la version latest pour acces direct et boucle de self-healing
        latest_file = output_path / "redteam_report_latest.json"
        try:
            latest_file.write_text(report_json, encoding="utf-8")
            logger.info("RedTeam: rapport latest mis a jour dans %s", latest_file)
        except Exception as exc:
            logger.warning("RedTeam: impossible d'ecrire le rapport latest: %s", exc)

        return str(file_path)

    def save_scenarios(
        self,
        scenarios: list[RedTeamScenario],
        output_dir: str = "data/redteam",
    ) -> str:
        """Persiste les scenarios generes sur le disque.

        Args:
            scenarios: Scenarios a sauvegarder.
            output_dir: Repertoire de sortie.

        Returns:
            str: Chemin du fichier ecrit.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        file_path = output_path / f"redteam_scenarios_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        payload = [asdict(s) for s in scenarios]
        file_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("RedTeam: %d scenarios sauvegardes dans %s", len(scenarios), file_path)
        return str(file_path)

    def _generate_adversarial_prices(
        self,
        base_price: float,
        failure_type: str,
        n_points: int,
    ) -> list[float]:
        """Genere une serie de prix adverses selon le type d'echec.

        Args:
            base_price: Prix de depart.
            failure_type: Type d'echec a reproduire.
            n_points: Nombre de points a generer.

        Returns:
            list[float]: Serie de prix synthetiques.
        """
        prices = [base_price]
        if failure_type == "hard_stop_exit":
            decline = self.np_rng.uniform(0.005, 0.03)
            for i in range(1, n_points):
                shock = decline * (1.0 + 4.0 * max(0, 1.0 - i / 30.0))
                change = self.np_rng.normal(-shock, 0.005)
                prices.append(max(prices[-1] * (1.0 + change), base_price * 0.85))

        elif failure_type == "liquidity_trap":
            for i in range(1, n_points):
                if 0.3 * n_points < i < 0.6 * n_points:
                    spike_vol = self.np_rng.uniform(0.02, 0.05)
                    change = self.np_rng.normal(0, spike_vol)
                else:
                    change = self.np_rng.normal(0, 0.003)
                prices.append(max(prices[-1] * (1.0 + change), base_price * 0.7))

        elif failure_type == "bad_runner_exit":
            for i in range(1, n_points):
                drift = 0.002 if i < n_points * 0.6 else -0.005
                change = self.np_rng.normal(drift, 0.004)
                prices.append(max(prices[-1] * (1.0 + change), base_price * 0.8))

        elif failure_type in ("range_entry_loss", "bad_trade"):
            for i in range(1, n_points):
                if i < n_points * 0.3:
                    change = self.np_rng.normal(0.001, 0.004)
                elif i < n_points * 0.7:
                    change = self.np_rng.normal(-0.003, 0.006)
                else:
                    change = self.np_rng.normal(0.0, 0.005)
                prices.append(max(prices[-1] * (1.0 + change), base_price * 0.75))

        else:
            for i in range(1, n_points):
                change = self.np_rng.normal(0, 0.005)
                prices.append(max(prices[-1] * (1.0 + change), base_price * 0.5))

        return prices

    def _generate_indicators(
        self,
        prices: list[float],
        source_indicators: dict[str, Any],
        failure_type: str,
    ) -> dict[str, list[float]]:
        """Genere des indicateurs synthetiques coherents avec les prix.

        Args:
            prices: Serie de prix.
            source_indicators: Indicateurs du trade source.
            failure_type: Type d'echec pour calibrer les indicateurs.

        Returns:
            dict[str, list[float]]: Indicateurs generes.
        """
        indicators: dict[str, list[float]] = {}
        price_arr = np.array(prices)

        indicators["price_norm"] = ((price_arr - price_arr.min())
                                     / max(1e-8, price_arr.max() - price_arr.min())).tolist()

        returns = np.diff(price_arr) / price_arr[:-1]
        rsi_values = [50.0]
        for ret in returns:
            gain = max(0, ret) * 100.0
            loss = max(0, -ret) * 100.0
            avg_gain = 0.1 * gain + 0.9 * rsi_values[-1]
            avg_loss = 0.1 * loss + 0.9 * (100.0 - rsi_values[-1])
            rs = avg_gain / max(1e-8, avg_loss)
            rsi = 100.0 - 100.0 / (1.0 + rs)
            rsi_values.append(rsi)

        if len(rsi_values) > len(prices):
            rsi_values = rsi_values[:len(prices)]
        elif len(rsi_values) < len(prices):
            rsi_values.extend([50.0] * (len(prices) - len(rsi_values)))

        indicators["rsi"] = rsi_values
        indicators["adx"] = [20.0 + self.np_rng.uniform(-5, 5) for _ in prices]

        ema = [prices[0]]
        alpha = 0.01
        for p in prices[1:]:
            ema.append(alpha * p + (1 - alpha) * ema[-1])
        indicators["ema_200"] = ema

        vwap_factor = 1.0 + self.np_rng.uniform(-0.02, 0.02)
        indicators["vwap"] = [p * vwap_factor for p in prices]

        return indicators

    def _compute_severity(self, failure_type: str, close_reason: str) -> float:
        """Calcule un niveau de severite pour le scenario.

        Args:
            failure_type: Type d'echec.
            close_reason: Raison de cloture.

        Returns:
            float: Severite entre 0.0 et 1.0.
        """
        base = {
            "hard_stop_exit": 0.8,
            "liquidity_trap": 0.9,
            "bad_runner_exit": 0.6,
            "bad_pyramid_exit": 0.7,
            "range_entry_loss": 0.5,
            "bad_trade": 0.4,
        }.get(failure_type, 0.5)

        severity_boost = {
            "stop_loss": 0.1,
            "unknown": 0.05,
        }.get(close_reason, 0.0)

        return min(1.0, base + severity_boost)

    def _generate_generic_scenario(self, failure_type: str) -> RedTeamScenario | None:
        """Genere un scenario generique pour un type d'echec non couvert.

        Args:
            failure_type: Type d'echec.

        Returns:
            RedTeamScenario | None: Scenario genere ou None.
        """
        symbols = ["XAUUSD", "EURUSD", "BTCUSD", "GER40.cash", "US30.cash"]
        symbol = self.rng.choice(symbols)
        base_price = self.rng.choice([100.0, 50000.0, 1.2, 2000.0])
        prices = self._generate_adversarial_prices(base_price, failure_type, 200)
        indicators = self._generate_indicators(prices, {}, failure_type)

        return RedTeamScenario(
            name=f"generic_{failure_type}_{symbol}_{self.rng.randint(1000, 9999)}",
            symbol=symbol,
            failure_type=failure_type,
            prices=prices,
            indicators=indicators,
            severity=self._compute_severity(failure_type, "unknown"),
            source_trade_id=None,
        )
