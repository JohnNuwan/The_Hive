# Story 01: The Kernel & Security (The Shield)

## 📌 Context
Security is not an option, especially with real money involved. The Kernel (Rust) provides immutable guarantees that Python logic cannot override.

## 🎯 Objectives
- Implement the "Financial Kill-Switch".
- Secure the Key handling.
- Monitor critical hardware metrics.

## 📋 Epic/Tasks Breakdown

### TASK-01-01: Rust Environment & Skeleton
- **Role**: Rust Developer
- **Description**: Set up the `kernel` crate workspace.
- **Acceptance Criteria**:
    - [ ] `cargo build` produces a binary.
    - [ ] Standard logging implemented.

### TASK-01-02: The Financial Kill-Switch (CRITICAL)
- **Role**: Backend Dev (Rust)
- **Description**: A standalone service that polls MT5 account state.
- **Logic**:
    ```rust
    fn check_health() {
        let equity = mt5_api.get_equity();
        let balance = mt5_api.get_daily_start_balance();
        if equity < balance * 0.96 { // 4% Max Loss
            panic_close_all();
        }
    }
    ```
- **Acceptance Criteria**:
    - [ ] Unit test proving the logic triggers correctly.
    - [ ] Integration test with a Mock MT5 API.
    - [ ] Latency < 100ms.

### TASK-01-03: Hardware Watchdog (Thermal)
- **Role**: System Dev
- **Description**: Monitor GPU temperature.
- **Acceptance Criteria**:
    - [ ] If GPU Temp > 85°C (Warning) -> Notify Admin.
    - [ ] If GPU Temp > 90°C (Critical) -> Trigger `shutdown -h now` on the VM.

### TASK-01-04: The Black Box (Audit)
- **Role**: Backend Dev
- **Description**: Secure Logger.
- **Acceptance Criteria**:
    - [ ] Logs are written to a separate partition/file.
    - [ ] Logs are cryptographically chained (Hash N includes Hash N-1).
    - [ ] Python agents can only *append* logs, never delete.
