# Story 03: The Banker (The Money)

## 📌 Context
The primary revenue generator. It must be robust, fast, and fail-safe.

## 🎯 Objectives
- Connect Python to MT5.
- Implement data ingestion.
- Execute the first "Paper Trade".

## 📋 Epic/Tasks Breakdown

### TASK-03-01: Windows VM & MT5 Setup
- **Role**: SysAdmin
- **Description**: Prepare the "Trading Floor" VM.
- **Acceptance Criteria**:
    - [ ] Windows 10 LTSC running.
    - [ ] MT5 Terminal installed and logged in (Demo Account).
    - [ ] Auto-login on reboot configured.

### TASK-03-02: MT5 Python Bridge
- **Role**: Python Dev (Quant)
- **Description**: Create a service ensuring connectivity.
- **Technical**: Use `MetaTrader5` python package.
- **Acceptance Criteria**:
    - [ ] `mt5.initialize()` returns True.
    - [ ] Can fetch account balance.
    - [ ] Can fetch last 100 candles of XAUUSD.

### TASK-03-03: Data Pipeline (TimescaleDB)
- **Role**: Data Engineer
- **Description**: Store market data for analysis.
- **Acceptance Criteria**:
    - [ ] Service polls MT5 every 1 minute.
    - [ ] Pushes OHLCV data to PostgreSQL (TimescaleDB).

### TASK-03-04: Order Execution Engine (Basic)
- **Role**: Python Dev
- **Description**: Wrapper to send orders safely.
- **Constraints**:
    - MUST check Stop Loss presence.
    - MUST check Max Risk (Lot size calculation).
- **Acceptance Criteria**:
    - [ ] A function `place_order(symbol, direction, sl, tp)` executes a trade in MT5 Demo.
    - [ ] Rejects order if SL is missing.
