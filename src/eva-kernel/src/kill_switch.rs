use pydantic::{BaseModel}; // Simulation, on est en Rust
use std::time::Duration;

pub struct KillSwitch {
    max_daily_drawdown: f64,
    current_drawdown: f64,
    is_halted: bool,
}

impl KillSwitch {
    pub fn new(max_drawdown: f64) -> Self {
        KillSwitch {
            max_daily_drawdown: max_drawdown,
            current_drawdown: 0.0,
            is_halted: false,
        }
    }

    pub fn intercept_request(&mut self, amount: f64, risk: f64) -> bool {
        if self.is_halted {
            println!("🛑 KERNEL: REQUEST REJECTED - SYSTEM HALTED");
            return false;
        }

        if risk > 0.04 {
            println!("⚠️ KERNEL: RISK EXCEEDS 4% LIMIT - BLOCKING");
            return false;
        }

        true
    }

    pub fn force_shutdown(&mut self) {
        self.is_halted = true;
        println!("🚨 KERNEL: EMERGENCY KILL-SWITCH ACTIVATED");
    }
}
