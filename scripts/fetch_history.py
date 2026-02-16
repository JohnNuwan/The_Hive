import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import os
import sys
import time

# Configuration
SYMBOLS = ["XAUUSD", "EURUSD", "BTCUSD", "USDJPY", "GBPUSD"]
# Potential aliases for US30
US30_ALIASES = ["US30", "US30.cash", "DJ30", "WS30", "US30Contract"]

TIMEFRAMES = {
    "M5": (mt5.TIMEFRAME_M5, 50000),  # ~6 months
    "H1": (mt5.TIMEFRAME_H1, 10000)   # ~1.5 years
}
OUTPUT_DIR = "data/history"

def find_us30_symbol():
    """Try to find the correct symbol for US30"""
    for alias in US30_ALIASES:
        if mt5.symbol_select(alias, True):
            return alias
    return None

def fetch_data(symbol, timeframe_name, timeframe_value, count):
    print(f"Fetching {symbol} ({timeframe_name}) - Count: {count}...")
    
    # Check if symbol is available
    if not mt5.symbol_select(symbol, True):
        print(f"Failed to select {symbol}")
        return None

    rates = mt5.copy_rates_from_pos(symbol, timeframe_value, 0, count)
    
    if rates is None:
        error = mt5.last_error()
        print(f"Failed to get rates for {symbol} ({timeframe_name}). Error: {error}")
        return None
        
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Save to CSV
    filename = f"{OUTPUT_DIR}/{symbol}_{timeframe_name}.csv"
    df.to_csv(filename, index=False)
    print(f"Saved {len(df)} rows to {filename}")
    return filename

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        sys.exit()
    
    print(f"MT5 Version: {mt5.version()}")
    
    # 1. Handle standard symbols
    success_count = 0
    total_count = 0

    target_symbols = list(SYMBOLS)
    
    # 2. Handle US30 alias
    us30 = find_us30_symbol()
    if us30:
        print(f"Found US30 symbol: {us30}")
        target_symbols.append(us30)
    else:
        print("Could not find US30 symbol among aliases.")

    for symbol in target_symbols:
        for tf_name, (tf_value, count) in TIMEFRAMES.items():
            total_count += 1
            if fetch_data(symbol, tf_name, tf_value, count):
                success_count += 1
            time.sleep(0.5) 

    mt5.shutdown()
    print(f"\nCompleted: {success_count}/{total_count} files generated.")

if __name__ == "__main__":
    main()
