//! Kill-Switch Financier — Coupe-circuit d'urgence (Loi 2)
//!
//! Surveille le drawdown journalier en temps réel.
//! Si le seuil est franchi (défaut 4%), TOUTE activité de trading est bloquée.
//! Seul un reset manuel (ou un timer de 24h) peut rétablir le système.

use chrono::{DateTime, Utc};
use serde::Serialize;
use tracing::{error, info, warn};

/// État du Kill-Switch
#[derive(Debug, Clone, Serialize)]
pub struct KillSwitchStatus {
    pub is_active: bool,
    pub activated_at: Option<DateTime<Utc>>,
    pub reason: Option<String>,
    pub current_drawdown: f64,
    pub max_daily_drawdown: f64,
    pub trades_blocked: u64,
}

/// Kill-Switch financier — composant de sécurité critique
pub struct KillSwitch {
    max_daily_drawdown: f64,
    current_drawdown: f64,
    is_halted: bool,
    activated_at: Option<DateTime<Utc>>,
    reason: Option<String>,
    trades_blocked: u64,
}

impl KillSwitch {
    /// Crée un nouveau Kill-Switch avec un seuil de drawdown max
    pub fn new(max_drawdown: f64) -> Self {
        info!(
            "🛡️ Kill-Switch initialisé (seuil: {:.1}%)",
            max_drawdown
        );
        KillSwitch {
            max_daily_drawdown: max_drawdown,
            current_drawdown: 0.0,
            is_halted: false,
            activated_at: None,
            reason: None,
            trades_blocked: 0,
        }
    }

    /// Vérifie si le Kill-Switch est actif (trading bloqué)
    pub fn is_active(&self) -> bool {
        self.is_halted
    }

    /// Intercepte une requête de trade et vérifie la conformité risque
    ///
    /// Retourne `true` si le trade est autorisé, `false` s'il est bloqué
    pub fn intercept_request(&mut self, _amount: f64, risk_percent: f64) -> bool {
        if self.is_halted {
            self.trades_blocked += 1;
            warn!(
                "🛑 KILL-SWITCH: Trade rejeté (#{}) — système en arrêt d'urgence",
                self.trades_blocked
            );
            return false;
        }

        // Vérification Loi 2: risque par trade ≤ 1%
        if risk_percent > 1.0 {
            warn!(
                "⚠️ KERNEL: Risque {:.2}% > 1% — trade bloqué (Loi 2)",
                risk_percent
            );
            return false;
        }

        // Vérification drawdown journalier ≤ seuil (défaut 4%)
        if self.current_drawdown >= self.max_daily_drawdown {
            self.activate("Drawdown journalier maximal atteint");
            return false;
        }

        true
    }

    /// Met à jour le drawdown courant (appelé après chaque tick P&L)
    pub fn update_drawdown(&mut self, drawdown_percent: f64) {
        self.current_drawdown = drawdown_percent;

        if drawdown_percent >= self.max_daily_drawdown && !self.is_halted {
            self.activate(&format!(
                "Drawdown {:.2}% >= seuil {:.1}%",
                drawdown_percent, self.max_daily_drawdown
            ));
        }
    }

    /// Active le Kill-Switch manuellement ou automatiquement
    pub fn activate(&mut self, reason: &str) {
        self.is_halted = true;
        self.activated_at = Some(Utc::now());
        self.reason = Some(reason.to_string());
        error!(
            "🚨 KILL-SWITCH ACTIVÉ: {}. Tout trading est BLOQUÉ.",
            reason
        );
    }

    /// Alias pour activation d'urgence manuelle
    pub fn force_shutdown(&mut self) {
        self.activate("Arrêt d'urgence manuel (force_shutdown)");
    }

    /// Reset du Kill-Switch (nécessite autorisation admin)
    pub fn reset(&mut self) {
        self.is_halted = false;
        self.activated_at = None;
        self.reason = None;
        self.current_drawdown = 0.0;
        self.trades_blocked = 0;
        info!("✅ KILL-SWITCH DÉSACTIVÉ — Trading rétabli");
    }

    /// Reset automatique après 24h (cycle circadien)
    pub fn check_auto_reset(&mut self) {
        if let Some(activated) = self.activated_at {
            let elapsed = Utc::now() - activated;
            if elapsed.num_hours() >= 24 {
                info!("🔄 Kill-Switch auto-reset après 24h");
                self.reset();
            }
        }
    }

    /// Retourne le statut complet du Kill-Switch
    pub fn get_status(&self) -> KillSwitchStatus {
        KillSwitchStatus {
            is_active: self.is_halted,
            activated_at: self.activated_at,
            reason: self.reason.clone(),
            current_drawdown: self.current_drawdown,
            max_daily_drawdown: self.max_daily_drawdown,
            trades_blocked: self.trades_blocked,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_kill_switch_is_inactive() {
        let ks = KillSwitch::new(4.0);
        assert!(!ks.is_active());
    }

    #[test]
    fn test_force_shutdown_activates() {
        let mut ks = KillSwitch::new(4.0);
        ks.force_shutdown();
        assert!(ks.is_active());
    }

    #[test]
    fn test_reset_deactivates() {
        let mut ks = KillSwitch::new(4.0);
        ks.force_shutdown();
        assert!(ks.is_active());
        ks.reset();
        assert!(!ks.is_active());
    }

    #[test]
    fn test_drawdown_triggers_halt() {
        let mut ks = KillSwitch::new(4.0);
        ks.update_drawdown(4.5);
        assert!(ks.is_active());
    }

    #[test]
    fn test_intercept_blocks_when_halted() {
        let mut ks = KillSwitch::new(4.0);
        ks.force_shutdown();
        assert!(!ks.intercept_request(1000.0, 0.5));
        assert_eq!(ks.get_status().trades_blocked, 1);
    }

    #[test]
    fn test_intercept_blocks_high_risk() {
        let mut ks = KillSwitch::new(4.0);
        assert!(!ks.intercept_request(1000.0, 2.5)); // > 1%
    }

    #[test]
    fn test_intercept_allows_valid_trade() {
        let mut ks = KillSwitch::new(4.0);
        assert!(ks.intercept_request(1000.0, 0.5));
    }
}
