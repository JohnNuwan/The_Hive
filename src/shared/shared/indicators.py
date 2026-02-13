import math
from typing import List, Dict, Tuple

class IndicatorFactory:
    """
    Math Kernel for Technical Indicators.
    Pure Python/Math implementation (no heavy deps if possible, but numpy is standard).
    Target: High performance, low latency for HFT-like decisions.
    """

    @staticmethod
    def sma(data: List[float], period: int) -> float:
        if len(data) < period: return 0.0
        return sum(data[-period:]) / period

    @staticmethod
    def ema(data: List[float], period: int, smoothing: int = 2) -> List[float]:
        """Calculates EMA series. Returns the full series to allow further calcs."""
        if len(data) < period: return []
        
        multiplier = smoothing / (1 + period)
        ema_result = [sum(data[:period]) / period] # Start with SMA
        
        for price in data[period:]:
            ema_val = (price - ema_result[-1]) * multiplier + ema_result[-1]
            ema_result.append(ema_val)
            
        return ema_result

    @staticmethod
    def rsi(prices: List[float], period: int = 14) -> float:
        if len(prices) < period + 1: return 50.0
        
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0.0)
            else:
                gains.append(0.0)
                losses.append(abs(change))
                
        # First Avg Gain/Loss
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        # Smoothed
        for i in range(period, len(prices) - 1):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
        if avg_loss == 0: return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def macd(prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, float]:
        if len(prices) < slow + signal: 
            return {"macd": 0.0, "signal": 0.0, "histogram": 0.0}
            
        ema_fast = IndicatorFactory.ema(prices, fast)
        ema_slow = IndicatorFactory.ema(prices, slow)
        
        # Trim to match lengths (ema_slow is shorter)
        # We need the last N values where N is length of ema_slow
        # Actually logic is: EMA calculation returns list starting from index 'period'.
        # So ema_fast starts at index 'fast', ema_slow at index 'slow'.
        # We need to align them by the original price index.
        
        # Simplification for realtime: Just calculate last value? No, MACD needs history for Signal line.
        # Re-implementation for correct series alignment:
        
        # Full Series calc is better
        # Let's use a simpler iterative approach if we just need the LATEST values
        # But for Signal Line (EMA of MACD), we need history of MACD.
        
        # Let's align lists from the end
        min_len = min(len(ema_fast), len(ema_slow))
        if min_len == 0: return {"macd": 0.0, "signal": 0.0, "histogram": 0.0}
        
        macd_line = []
        # ema_fast[-min_len:] corresponds to ema_slow[-min_len:] time-wise
        f_series = ema_fast[-min_len:]
        s_series = ema_slow[-min_len:]
        
        for f, s in zip(f_series, s_series):
            macd_line.append(f - s)
            
        # Signal Line = EMA of MACD Line
        signal_line_series = IndicatorFactory.ema(macd_line, signal)
        
        if not signal_line_series: 
            return {"macd": macd_line[-1], "signal": 0.0, "histogram": 0.0}
            
        return {
            "macd": macd_line[-1],
            "signal": signal_line_series[-1],
            "histogram": macd_line[-1] - signal_line_series[-1]
        }

    @staticmethod
    def bollinger_bands(prices: List[float], period: int = 20, std_dev: float = 2.0) -> Dict[str, float]:
        if len(prices) < period: return {"upper": 0.0, "middle": 0.0, "lower": 0.0, "width": 0.0}
        
        basis = IndicatorFactory.sma(prices, period)
        
        # Std Dev
        slice_data = prices[-period:]
        variance = sum([pow(x - basis, 2) for x in slice_data]) / period
        std = math.sqrt(variance)
        
        return {
            "upper": basis + (std_dev * std),
            "middle": basis,
            "lower": basis - (std_dev * std),
            "width": (basis + (std_dev * std)) - (basis - (std_dev * std)),
            "pct_b": (prices[-1] - (basis - std_dev * std)) / ((basis + std_dev * std) - (basis - std_dev * std)) if std > 0 else 0.5
        }

    @staticmethod
    def atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
        if len(closes) < period + 1: return 0.0
        
        tr_list = []
        for i in range(1, len(closes)):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i-1])
            lc = abs(lows[i] - closes[i-1])
            tr_list.append(max(hl, hc, lc))
            
        # First ATR is simple average
        curr_atr = sum(tr_list[:period]) / period
        
        # Smoothed
        for i in range(period, len(tr_list)):
            curr_atr = (curr_atr * (period - 1) + tr_list[i]) / period
            
        return curr_atr

    @staticmethod
    def get_fibonacci_levels(highs: List[float], lows: List[float], depth: int = 100) -> Dict[str, float]:
        """Finds recent High/Low in depth and calculates Retracements."""
        if not highs or not lows: return {}
        
        # Lookback depth
        recent_high = max(highs[-depth:]) if len(highs) >= depth else max(highs)
        recent_low = min(lows[-depth:]) if len(lows) >= depth else min(lows)
        
        diff = recent_high - recent_low
        
        return {
            "fib_high": recent_high,
            "fib_low": recent_low,
            "fib_0": recent_low,
            "fib_236": recent_low + diff * 0.236,
            "fib_382": recent_low + diff * 0.382,
            "fib_500": recent_low + diff * 0.500,
            "fib_618": recent_low + diff * 0.618,
            "fib_100": recent_high
        }

    @staticmethod
    def relative_volume(volumes: List[float], period: int = 20) -> float:
        if len(volumes) < period + 1: return 1.0
        current = volumes[-1]
        avg = sum(volumes[-(period+1):-1]) / period 
        if avg == 0: return 1.0
        return current / avg

    @staticmethod
    def detect_cycles(closes: List[float], depth: int = 50) -> Dict[str, int]:
        """Detects simple pivot points to estimate time since last Low/High."""
        if len(closes) < 5: return {"bars_since_low": 0, "bars_since_high": 0}
        
        # Simple Pivot: Higher than neighbours
        #   X
        # X   X
        
        last_high_idx = 0
        last_low_idx = 0
        
        for i in range(2, len(closes) - 2):
            # High pivot
            if closes[i] > closes[i-1] and closes[i] > closes[i-2] and \
               closes[i] > closes[i+1] and closes[i] > closes[i+2]:
                last_high_idx = i
                
            # Low pivot
            if closes[i] < closes[i-1] and closes[i] < closes[i-2] and \
               closes[i] < closes[i+1] and closes[i] < closes[i+2]:
                last_low_idx = i
                
        current_idx = len(closes) - 1
        return {
            "bars_since_high": current_idx - last_high_idx,
            "bars_since_low": current_idx - last_low_idx
        }
