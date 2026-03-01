import pandas as pd

# Load MT5 Excel Export (Skip the first 6 header rows)
df = pd.read_excel('C:/Users/nandi/Desktop/eva_trade_test.xlsx', header=6)

# Drop rows where Symbol is missing (this drops the bottom summary rows)
df = df.dropna(subset=['Symbole'])

# Convert Profit to numeric, coaxing errors to NaN, then drop those
df['Profit'] = pd.to_numeric(df['Profit'], errors='coerce')
df = df.dropna(subset=['Profit'])

print("=== GLOBAL PERFORMANCE ===")
total_trades = len(df)
wins = len(df[df['Profit'] > 0])
losses = len(df[df['Profit'] < 0])
win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
total_pnl = df['Profit'].sum()

print(f"Total Trades: {total_trades}")
print(f"Win Rate: {win_rate:.2f}% ({wins} W / {losses} L)")
print(f"Total P&L: ${total_pnl:.2f}\n")

print("=== PERFORMANCE BY SYMBOL ===")
grouped = df.groupby('Symbole')['Profit'].agg(['count', 'sum', lambda x: (x > 0).mean() * 100])
grouped.columns = ['Trades', 'Total P&L', 'Win Rate %']
print(grouped.to_string())

print("\n=== TOP 5 WORST TRADES ===")
print(df.sort_values(by='Profit').head(5)[['Heure', 'Symbole', 'Volume', 'Profit']].to_string(index=False))

print("\n=== TOP 5 BEST TRADES ===")
print(df.sort_values(by='Profit', ascending=False).head(5)[['Heure', 'Symbole', 'Volume', 'Profit']].to_string(index=False))
