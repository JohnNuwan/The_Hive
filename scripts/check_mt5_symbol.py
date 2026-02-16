import MetaTrader5 as mt5
import sys

def check_symbol(symbol):
    if not mt5.initialize():
        print(f"MT5 initialize failed: {mt5.last_error()}")
        return
    
    # Check if symbol exists
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"Symbol {symbol} NOT found in MT5.")
        # Try to select it
        if mt5.symbol_select(symbol, True):
            print(f"Symbol {symbol} was found but NOT selected. It is now selected.")
            symbol_info = mt5.symbol_info(symbol)
        else:
            print(f"Symbol {symbol} could not be selected. error: {mt5.last_error()}")
            
            # List some symbols to see suffixes
            symbols = mt5.symbols_get()
            print(f"Total symbols found: {len(symbols)}")
            print("First 10 symbols:")
            for s in symbols[:10]:
                print(f" - {s.name}")
            
            # Try to find something that looks like Gold
            gold_keys = ["GOLD", "XAU", "METAL"]
            for s in symbols:
                if any(k in s.name.upper() for k in gold_keys):
                    print(f"Potential Gold match: {s.name}")
    else:
        print(f"Symbol {symbol} is available and selected.")
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            print(f"Last tick for {symbol}: Bid={tick.bid}, Ask={tick.ask}")
        else:
            print(f"Could not get tick for {symbol}. error: {mt5.last_error()}")

    mt5.shutdown()

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "XAUUSD"
    check_symbol(target)
