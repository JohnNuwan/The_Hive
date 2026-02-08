"""
MT5_Bridge.py — Utilitaire de test de connexion MetaTrader 5.
Ce script doit être exécuté directement sur l'hôte Windows où MT5 est installé.
"""

import MetaTrader5 as mt5
import sys
import time
import json

def test_connection():
    print("--- HYDRA PROTOCOL : MT5 Bridge Test ---")
    
    # Initialisation
    if not mt5.initialize():
        print(f"FAILED: mt5.initialize() failed, error code = {mt5.last_error()}")
        return

    print("SUCCESS: MT5 Initialized")
    
    # Infos terminal
    terminal_info = mt5.terminal_info()
    if terminal_info:
        print(f"Terminal Info: {terminal_info.company} - {terminal_info.name}")
    
    # Infos compte
    account_info = mt5.account_info()
    if account_info:
        print(f"\nAccount Balance: {account_info.balance} {account_info.currency}")
        print(f"Equity: {account_info.equity}")
        print(f"Profit: {account_info.profit}")
        print(f"Leverage: {account_info.leverage}")
    else:
        print("WARNING: Not connected to an account. Use mt5.login() if needed.")

    # Test de Ticks
    symbol = "XAUUSD"
    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info:
        # Essayer EURUSD si XAUUSD absent
        symbol = "EURUSD"
        symbol_info = mt5.symbol_info(symbol)

    if symbol_info:
        print(f"\nTesting Ticks for {symbol}:")
        for _ in range(5):
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                print(f"  [{symbol}] BID: {tick.bid} | ASK: {tick.ask}")
            time.sleep(1)
    else:
        print(f"\nSymbol {symbol} not found.")

    mt5.shutdown()
    print("\n--- Test Complete ---")

if __name__ == "__main__":
    if sys.platform != "win32":
        print("ERROR: This bridge only runs on Windows (MT5 requirement).")
        sys.exit(1)
        
    test_connection()
