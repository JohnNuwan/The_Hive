import math
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Union

class IndicatorFactory:
    """
    Math Kernel for Technical Indicators.
    Optimized for Pandas/Numpy vectorization.
    Target: High performance on large datasets.
    """

    @staticmethod
    def sma(data: Union[List[float], pd.Series], period: int) -> pd.Series:
        if isinstance(data, list): data = pd.Series(data)
        return data.rolling(window=period).mean()

    @staticmethod
    def ema(data: Union[List[float], pd.Series], period: int) -> pd.Series:
        """Calculates EMA series using Pandas ewm."""
        if isinstance(data, list): data = pd.Series(data)
        # span=period is standard EMA
        return data.ewm(span=period, adjust=False).mean()

    @staticmethod
    def rsi(prices: Union[List[float], pd.Series], period: int = 14) -> pd.Series:
        if isinstance(prices, list): prices = pd.Series(prices)
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0))
        loss = (-delta.where(delta < 0, 0))

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Smoothed Wilder's Moving Average (classic RSI)
        # But for speed, standard rolling is often close enough. 
        # Making it exact Wilder's:
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50.0)

    @staticmethod
    def macd(prices: Union[List[float], pd.Series], fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        if isinstance(prices, list): prices = pd.Series(prices)
        
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return {
            "macd": macd_line,
            "signal": signal_line,
            "histogram": histogram
        }

    @staticmethod
    def bollinger_bands(prices: Union[List[float], pd.Series], period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
        if isinstance(prices, list): prices = pd.Series(prices)
        
        basis = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper = basis + (std * std_dev)
        lower = basis - (std * std_dev)
        
        return {
            "upper": upper,
            "middle": basis,
            "lower": lower,
            "width": upper - lower,
            "pct_b": (prices - lower) / (upper - lower)
        }

    @staticmethod
    def atr(highs: Union[List[float], pd.Series], lows: Union[List[float], pd.Series], closes: Union[List[float], pd.Series], period: int = 14) -> pd.Series:
        if isinstance(highs, list): highs = pd.Series(highs)
        if isinstance(lows, list): lows = pd.Series(lows)
        if isinstance(closes, list): closes = pd.Series(closes)
        
        h_l = highs - lows
        h_pc = (highs - closes.shift(1)).abs()
        l_pc = (lows - closes.shift(1)).abs()
        
        tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        return atr

    @staticmethod
    def relative_volume(volumes: Union[List[float], pd.Series], period: int = 20) -> pd.Series:
        if isinstance(volumes, list): volumes = pd.Series(volumes)
        
        # Shift 1 to define previous period avg (usually we compare current to past avg)
        # But often rvol includes current. Let's do simple rolling avg.
        avg_vol = volumes.rolling(window=period).mean()
        rvol = volumes / avg_vol
        return rvol.fillna(1.0)

    @staticmethod
    def get_fibonacci_levels(highs: Union[List[float], pd.Series], lows: Union[List[float], pd.Series], period: int = 100) -> Dict[str, float]:
        """Calculates Fibonacci Retracement Levels based on recent High/Low."""
        if isinstance(highs, list): highs = pd.Series(highs)
        if isinstance(lows, list): lows = pd.Series(lows)
        
        # Lookback period
        recent_high = highs.rolling(window=period).max().iloc[-1]
        recent_low = lows.rolling(window=period).min().iloc[-1]
        
        diff = recent_high - recent_low
        
        return {
            "fib_0": recent_low,
            "fib_236": recent_low + diff * 0.236,
            "fib_382": recent_low + diff * 0.382,
            "fib_500": recent_low + diff * 0.5,
            "fib_618": recent_low + diff * 0.618,
            "fib_100": recent_high
        }

    @staticmethod
    def detect_cycles(closes: Union[List[float], pd.Series]) -> Dict[str, int]:
        """
        Simple cycle detection: counts bars since last significant High/Low.
        """
        if isinstance(closes, list): closes = pd.Series(closes)
        
        # 20-period Donchian Channel effectively
        period = 20
        rolling_high = closes.rolling(window=period).max()
        rolling_low = closes.rolling(window=period).min()
        
        # Find index of last high/low
        last_high_idx = closes[closes == rolling_high].last_valid_index()
        last_low_idx = closes[closes == rolling_low].last_valid_index()
        
        current_idx = closes.index[-1]
        
        bars_since_high = (current_idx - last_high_idx) if last_high_idx is not None else period
        bars_since_low = (current_idx - last_low_idx) if last_low_idx is not None else period
        
        return {
            "bars_since_high": int(bars_since_high),
            "bars_since_low": int(bars_since_low)
        }
