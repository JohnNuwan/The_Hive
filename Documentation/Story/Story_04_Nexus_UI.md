# Story 04: The Nexus & UI (The Interface)

## 📌 Context
How the Admin interacts with the Hive.

## 🎯 Objectives
- Create the Admin Dashboard (Panopticon).
- Setup the secure Chat interface (Nexus).

## 📋 Epic/Tasks Breakdown

### TASK-04-01: Dashboard V0 (Streamlit)
- **Role**: Frontend Dev
- **Description**: A quick-and-dirty internal dashboard to monitor system health.
- **Components**:
    - GPU Temp gauge.
    - Recent Log entries.
    - MT5 Account Equity curve.
- **Acceptance Criteria**:
    - [ ] App launches and connects to Redis/DB.
    - [ ] Updates every 5s.

### TASK-04-02: Secure Chat (Nexus Backend)
- **Role**: Backend Dev
- **Description**: WebSocket handling for chat.
- **Acceptance Criteria**:
    - [ ] Client can connect via WS.
    - [ ] Messages are persisted in DB.
    - [ ] Messages are routed to EVA Core.

### TASK-04-03: Tailscale Access
- **Role**: SysAdmin
- **Description**: Setup remote access.
- **Acceptance Criteria**:
    - [ ] Admin can access the Dashboard from their phone via 4G (using Tailscale IP).
    - [ ] No ports opened on the Box router.
