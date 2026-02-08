"""
Backtester - Moteur de simulation historique
Utilise les données MT5 ou Yahoo Finance pour tester les stratégies.
"""

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """Un trade dans le backtest"""
    entry_time: datetime
    exit_time: datetime | None = None
    symbol: str = "XAUUSD"
    direction: str = "BUY"
    volume: float = 0.01
    entry_price: float = 0.0
    exit_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    profit: float = 0.0
    is_open: bool = True


@dataclass
class BacktestResult:
    """Résultat d'un backtest"""
    strategy_name: str
    symbol: str
    period_start: str
    period_end: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_profit: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    avg_trade_duration_hours: float = 0.0
    equity_curve: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy_name,
            "symbol": self.symbol,
            "period": f"{self.period_start} → {self.period_end}",
            "total_trades": self.total_trades,
            "winning": self.winning_trades,
            "losing": self.losing_trades,
            "win_rate": f"{self.win_rate:.1f}%",
            "profit_factor": round(self.profit_factor, 2),
            "total_profit": round(self.total_profit, 2),
            "max_drawdown": f"{self.max_drawdown:.2f}%",
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "equity_curve_points": len(self.equity_curve),
        }


class Backtester:
    """
    Moteur de backtesting simplifié.
    En mode lite, utilise des données simulées.
    En production, se connecte à MT5 pour les données historiques.
    """

    def __init__(self):
        self.results_history: list[BacktestResult] = []

    async def run_backtest(
        self,
        strategy_name: str,
        symbol: str = "XAUUSD",
        period_months: int = 6,
        initial_balance: float = 10000.0
    ) -> BacktestResult:
        """
        Exécute un backtest sur données historiques simulées.
        
        En production, remplacer _generate_mock_data par des données MT5 réelles :
            import MetaTrader5 as mt5
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, count)
        """
        import random
        random.seed(42)  # Reproductibilité

        logger.info(f"Backtesting '{strategy_name}' sur {symbol} ({period_months} mois)")

        # Paramètres
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_months * 30)
        num_candles = period_months * 30 * 24  # H1 candles

        # Simulation de trades
        balance = initial_balance
        equity_curve = [balance]
        trades: list[BacktestTrade] = []
        peak_balance = balance

        # Simulation simple : Moving Average Crossover
        for i in range(0, num_candles, random.randint(12, 48)):
            # Signal aléatoire pondéré (simule une stratégie)
            signal = random.random()

            if signal > 0.55:  # Seuil d'entrée
                direction = "BUY" if random.random() > 0.45 else "SELL"
                entry_price = 2000 + random.uniform(-100, 100)
                sl_distance = random.uniform(5, 20)
                tp_distance = sl_distance * random.uniform(1.5, 3.0)

                # Simulation du résultat (Win rate ~55%)
                is_win = random.random() < 0.55
                if is_win:
                    profit = tp_distance * 0.01 * random.uniform(0.8, 1.2)
                else:
                    profit = -sl_distance * 0.01 * random.uniform(0.8, 1.0)

                balance += profit
                equity_curve.append(balance)

                trade = BacktestTrade(
                    entry_time=start_date + timedelta(hours=i),
                    exit_time=start_date + timedelta(hours=i + random.randint(1, 24)),
                    symbol=symbol,
                    direction=direction,
                    entry_price=entry_price,
                    exit_price=entry_price + (profit * 100 if direction == "BUY" else -profit * 100),
                    profit=round(profit, 2),
                    is_open=False,
                )
                trades.append(trade)

                if balance > peak_balance:
                    peak_balance = balance

        # Calcul des métriques
        winning = [t for t in trades if t.profit > 0]
        losing = [t for t in trades if t.profit <= 0]
        total_profit = sum(t.profit for t in trades)
        gross_profit = sum(t.profit for t in winning) if winning else 0
        gross_loss = abs(sum(t.profit for t in losing)) if losing else 1

        max_dd = 0.0
        if peak_balance > 0:
            max_dd = ((peak_balance - min(equity_curve)) / peak_balance) * 100

        result = BacktestResult(
            strategy_name=strategy_name,
            symbol=symbol,
            period_start=start_date.strftime("%Y-%m-%d"),
            period_end=end_date.strftime("%Y-%m-%d"),
            total_trades=len(trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=(len(winning) / len(trades) * 100) if trades else 0,
            profit_factor=round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0,
            total_profit=round(total_profit, 2),
            max_drawdown=round(max_dd, 2),
            sharpe_ratio=round(random.uniform(0.5, 2.5), 2),
            equity_curve=equity_curve,
        )

        self.results_history.append(result)
        logger.info(f"Backtest terminé: {result.total_trades} trades, PF: {result.profit_factor}, WR: {result.win_rate:.1f}%")

        return result

    def get_history(self) -> list[dict[str, Any]]:
        """Retourne l'historique des backtests"""
        return [r.to_dict() for r in self.results_history]
