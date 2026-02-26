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
    def obv(closes: Union[List[float], pd.Series], volumes: Union[List[float], pd.Series]) -> pd.Series:
        if isinstance(closes, list): closes = pd.Series(closes)
        if isinstance(volumes, list): volumes = pd.Series(volumes)
        
        diff = closes.diff()
        direction = pd.Series(0, index=closes.index)
        direction[diff > 0] = 1
        direction[diff < 0] = -1
        
        obv = (volumes * direction).cumsum()
        return obv.fillna(0.0)

    @staticmethod
    def vwap(highs: Union[List[float], pd.Series], lows: Union[List[float], pd.Series], closes: Union[List[float], pd.Series], volumes: Union[List[float], pd.Series]) -> pd.Series:
        """Simple Rolling VWAP over a session (approximate typically with rolling window if intraday without anchor)"""
        if isinstance(highs, list): highs = pd.Series(highs)
        if isinstance(lows, list): lows = pd.Series(lows)
        if isinstance(closes, list): closes = pd.Series(closes)
        if isinstance(volumes, list): volumes = pd.Series(volumes)
        
        typical_price = (highs + lows + closes) / 3
        # Assuming a rolling VWAP for algorithmic consistency without daily anchors
        return (typical_price * volumes).rolling(window=100).sum() / volumes.rolling(window=100).sum()

    @staticmethod
    def relative_volume(volumes: Union[List[float], pd.Series], period: int = 20) -> pd.Series:
        if isinstance(volumes, list): volumes = pd.Series(volumes)
        
        avg_vol = volumes.rolling(window=period).mean()
        rvol = volumes / avg_vol
        return rvol.fillna(1.0)

    @staticmethod
    def momentum(closes: Union[List[float], pd.Series], period: int = 10) -> pd.Series:
        """Rate of Change (ROC) / Momentum"""
        if isinstance(closes, list): closes = pd.Series(closes)
        roc = ((closes - closes.shift(period)) / closes.shift(period)) * 100
        return roc.fillna(0.0)

    @staticmethod
    def trix(closes: Union[List[float], pd.Series], period: int = 15) -> pd.Series:
        if isinstance(closes, list): closes = pd.Series(closes)
        ema1 = closes.ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        ema3 = ema2.ewm(span=period, adjust=False).mean()
        trix_s = ((ema3 - ema3.shift(1)) / ema3.shift(1)) * 100
        return trix_s.fillna(0.0)

    @staticmethod
    def stochastic(highs: Union[List[float], pd.Series], lows: Union[List[float], pd.Series], closes: Union[List[float], pd.Series], period: int = 14, smooth_k: int = 3) -> Dict[str, pd.Series]:
        if isinstance(highs, list): highs = pd.Series(highs)
        if isinstance(lows, list): lows = pd.Series(lows)
        if isinstance(closes, list): closes = pd.Series(closes)
        
        low_min = lows.rolling(window=period).min()
        high_max = highs.rolling(window=period).max()
        
        k_fast = 100 * (closes - low_min) / (high_max - low_min)
        k_slow = k_fast.rolling(window=smooth_k).mean()
        d_slow = k_slow.rolling(window=3).mean()
        
        return {
            "percent_k": k_slow.fillna(50.0),
            "percent_d": d_slow.fillna(50.0)
        }

    @staticmethod
    def cci(highs: Union[List[float], pd.Series], lows: Union[List[float], pd.Series], closes: Union[List[float], pd.Series], period: int = 20) -> pd.Series:
        if isinstance(highs, list): highs = pd.Series(highs)
        if isinstance(lows, list): lows = pd.Series(lows)
        if isinstance(closes, list): closes = pd.Series(closes)
        
        typical_price = (highs + lows + closes) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        
        cci = (typical_price - sma_tp) / (0.015 * mad)
        return cci.fillna(0.0)

    @staticmethod
    def adx(highs: Union[List[float], pd.Series], lows: Union[List[float], pd.Series], closes: Union[List[float], pd.Series], period: int = 14) -> Dict[str, pd.Series]:
        if isinstance(highs, list): highs = pd.Series(highs)
        if isinstance(lows, list): lows = pd.Series(lows)
        if isinstance(closes, list): closes = pd.Series(closes)
        
        up_move = highs - highs.shift(1)
        down_move = lows.shift(1) - lows
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        plus_dm = pd.Series(plus_dm, index=highs.index)
        minus_dm = pd.Series(minus_dm, index=highs.index)
        
        tr1 = highs - lows
        tr2 = (highs - closes.shift(1)).abs()
        tr3 = (lows - closes.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        
        plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
        
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
        adx = dx.ewm(alpha=1/period, adjust=False).mean()
        
        return {
            "adx": adx.fillna(0.0),
            "plus_di": plus_di.fillna(0.0),
            "minus_di": minus_di.fillna(0.0)
        }

    @staticmethod
    def ichimoku(highs: Union[List[float], pd.Series], lows: Union[List[float], pd.Series], closes: Union[List[float], pd.Series], tenkan_period: int = 9, kijun_period: int = 26, senkou_period: int = 52) -> Dict[str, pd.Series]:
        if isinstance(highs, list): highs = pd.Series(highs)
        if isinstance(lows, list): lows = pd.Series(lows)
        if isinstance(closes, list): closes = pd.Series(closes)
        
        tenkan_sen = (highs.rolling(window=tenkan_period).max() + lows.rolling(window=tenkan_period).min()) / 2
        kijun_sen = (highs.rolling(window=kijun_period).max() + lows.rolling(window=kijun_period).min()) / 2
        
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun_period)
        senkou_span_b = ((highs.rolling(window=senkou_period).max() + lows.rolling(window=senkou_period).min()) / 2).shift(kijun_period)
        
        chikou_span = closes.shift(-kijun_period)
        
        return {
            "tenkan_sen": tenkan_sen.fillna(0.0),
            "kijun_sen": kijun_sen.fillna(0.0),
            "senkou_span_a": senkou_span_a.fillna(0.0),
            "senkou_span_b": senkou_span_b.fillna(0.0),
            "chikou_span": chikou_span.fillna(0.0)
        }

    @staticmethod
    def trendlines(closes: Union[List[float], pd.Series], window: int = 20) -> pd.Series:
        """Linear regression slope over a rolling window"""
        if isinstance(closes, list): closes = pd.Series(closes)
        
        def calc_slope(y):
            if len(y) < 2: return 0.0
            x = np.arange(len(y))
            slope, _ = np.polyfit(x, y, 1)
            return slope
            
        slopes = closes.rolling(window=window).apply(calc_slope, raw=True)
        return slopes.fillna(0.0)

    @staticmethod
    def get_fibonacci_levels(highs: Union[List[float], pd.Series], lows: Union[List[float], pd.Series], period: int = 100) -> Dict[str, float]:
        """Calculates Advanced Fibonacci Retracement & Extension Levels based on recent Swing High/Low."""
        if isinstance(highs, list): highs = pd.Series(highs)
        if isinstance(lows, list): lows = pd.Series(lows)
        
        recent_high = highs.rolling(window=period).max().iloc[-1]
        recent_low = lows.rolling(window=period).min().iloc[-1]
        diff = recent_high - recent_low
        
        return {
            "fib_0": recent_low,
            "fib_236": recent_low + diff * 0.236,
            "fib_382": recent_low + diff * 0.382,
            "fib_500": recent_low + diff * 0.5,
            "fib_618": recent_low + diff * 0.618,
            "fib_786": recent_low + diff * 0.786,
            "fib_100": recent_high,
            "fib_ext_1618": recent_high + diff * 0.618,
            "fib_ext_2618": recent_high + diff * 1.618,
        }

    @staticmethod
    def support_resistance(highs: Union[List[float], pd.Series], lows: Union[List[float], pd.Series], closes: Union[List[float], pd.Series], window: int = 20) -> Dict[str, float]:
        """Detects Local Support and Resistance by finding rolling min/max over the window."""
        if isinstance(highs, list): highs = pd.Series(highs)
        if isinstance(lows, list): lows = pd.Series(lows)
        if isinstance(closes, list): closes = pd.Series(closes)
        
        pivot_high = highs.rolling(window=window, center=True).max()
        pivot_low = lows.rolling(window=window, center=True).min()
        
        last_res = pivot_high.dropna().iloc[-1] if not pivot_high.dropna().empty else highs.iloc[-1]
        last_sup = pivot_low.dropna().iloc[-1] if not pivot_low.dropna().empty else lows.iloc[-1]
        
        current_price = closes.iloc[-1]
        
        dist_res = max(0.0, last_res - current_price)
        dist_sup = max(0.0, current_price - last_sup)
        
        return {
            "nearest_resistance": float(last_res),
            "nearest_support": float(last_sup),
            "dist_to_res": float(dist_res),
            "dist_to_sup": float(dist_sup)
        }

    @staticmethod
    def gann_angles(highs: Union[List[float], pd.Series], lows: Union[List[float], pd.Series], period: int = 100) -> Dict[str, float]:
        """Very basic Gann fan mapping from the recent swing low/high."""
        if isinstance(highs, list): highs = pd.Series(highs)
        if isinstance(lows, list): lows = pd.Series(lows)
        
        recent_high = highs.rolling(window=period).max().iloc[-1]
        recent_low = lows.rolling(window=period).min().iloc[-1]
        
        price_diff = recent_high - recent_low
        if period == 0: period = 1
        slope_1x1 = price_diff / period
        
        return {
            "gann_1x1": slope_1x1,
            "gann_1x2": slope_1x1 * 0.5,
            "gann_2x1": slope_1x1 * 2.0
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
