//! Module Validator - Validation des trades selon la Constitution
//!
//! Implémente la validation Loi 2 en Rust pour performance critique.

use crate::laws::Constitution;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Résultat de validation d'un trade
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub allowed: bool,
    pub reason: Option<String>,
    pub law_reference: Option<String>,
    pub risk_percent: f64,
    pub checks: Vec<ValidationCheck>,
}

/// Check individuel de validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationCheck {
    pub name: String,
    pub passed: bool,
    pub message: String,
}

/// Requête de validation de trade
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeValidationRequest {
    pub id: Uuid,
    pub symbol: String,
    pub action: String,       // "BUY" ou "SELL"
    pub volume: f64,
    pub stop_loss: Option<f64>,
    pub take_profit: Option<f64>,
    pub current_price: f64,
    pub account_balance: f64,
    pub open_positions_count: u32,
    pub daily_drawdown_percent: f64,
}

/// Validateur de trades
pub struct TradeValidator {
    constitution: Constitution,
}

impl TradeValidator {
    /// Crée un nouveau validateur avec la Constitution
    pub fn new(constitution: Constitution) -> Self {
        Self { constitution }
    }

    /// Valide un trade selon la Constitution Loi 2
    pub fn validate(&self, request: &TradeValidationRequest) -> ValidationResult {
        let mut checks = Vec::new();
        let mut allowed = true;
        let mut reason: Option<String> = None;

        // 1. Vérification Stop Loss obligatoire
        let sl_check = if request.stop_loss.is_some() {
            ValidationCheck {
                name: "stop_loss".to_string(),
                passed: true,
                message: "Stop Loss présent".to_string(),
            }
        } else {
            allowed = false;
            reason = Some("Stop Loss obligatoire (ROE Trading)".to_string());
            ValidationCheck {
                name: "stop_loss".to_string(),
                passed: false,
                message: "Stop Loss manquant".to_string(),
            }
        };
        checks.push(sl_check);

        // Calcul du risque
        let risk_percent = if let Some(sl) = request.stop_loss {
            self.calculate_risk_percent(request, sl)
        } else {
            100.0 // Risque infini sans SL
        };

        // 2. Vérification risque par trade
        let risk_check = if self.constitution.is_trade_risk_allowed(risk_percent) {
            ValidationCheck {
                name: "risk_per_trade".to_string(),
                passed: true,
                message: format!("Risque {:.2}% <= max {}%", 
                    risk_percent, 
                    self.constitution.trading.max_risk_per_trade_percent),
            }
        } else {
            allowed = false;
            reason = Some(format!(
                "Risque {:.2}% > max {}% (Loi 2)",
                risk_percent,
                self.constitution.trading.max_risk_per_trade_percent
            ));
            ValidationCheck {
                name: "risk_per_trade".to_string(),
                passed: false,
                message: format!("Risque {:.2}% trop élevé", risk_percent),
            }
        };
        checks.push(risk_check);

        // 3. Vérification drawdown journalier
        let dd_check = if self.constitution.is_daily_drawdown_ok(request.daily_drawdown_percent) {
            ValidationCheck {
                name: "daily_drawdown".to_string(),
                passed: true,
                message: format!("DD journalier {:.2}% OK", request.daily_drawdown_percent),
            }
        } else {
            allowed = false;
            reason = Some(format!(
                "Drawdown journalier {:.2}% >= limite {}%",
                request.daily_drawdown_percent,
                self.constitution.trading.max_daily_drawdown_percent
            ));
            ValidationCheck {
                name: "daily_drawdown".to_string(),
                passed: false,
                message: format!("DD journalier limite atteinte"),
            }
        };
        checks.push(dd_check);

        // 4. Vérification nombre de positions
        let pos_check = if self.constitution.can_open_position(request.open_positions_count) {
            ValidationCheck {
                name: "max_positions".to_string(),
                passed: true,
                message: format!(
                    "Positions {}/{}", 
                    request.open_positions_count,
                    self.constitution.trading.max_concurrent_positions
                ),
            }
        } else {
            allowed = false;
            reason = Some(format!(
                "Max positions atteint ({})",
                self.constitution.trading.max_concurrent_positions
            ));
            ValidationCheck {
                name: "max_positions".to_string(),
                passed: false,
                message: "Limite de positions atteinte".to_string(),
            }
        };
        checks.push(pos_check);

        ValidationResult {
            allowed,
            reason,
            law_reference: if !allowed {
                Some("Loi 2 - Protection du Capital".to_string())
            } else {
                None
            },
            risk_percent,
            checks,
        }
    }

    /// Calcule le pourcentage de risque d'un trade
    fn calculate_risk_percent(&self, request: &TradeValidationRequest, stop_loss: f64) -> f64 {
        if request.account_balance <= 0.0 {
            return 100.0;
        }

        // Distance au SL
        let sl_distance = (request.current_price - stop_loss).abs();

        // Valeur par point (approximation)
        let point_value = if request.symbol.contains("XAU") {
            100.0 // Gold: $100 par lot par point
        } else {
            10.0 // Forex: $10 par lot par pip
        };

        // Perte potentielle
        let potential_loss = sl_distance * request.volume * point_value;

        // Pourcentage
        (potential_loss / request.account_balance) * 100.0
    }

    /// Retourne le risque max par trade configuré
    pub fn get_max_risk_per_trade(&self) -> f64 {
        self.constitution.trading.max_risk_per_trade_percent
    }

    /// Retourne le drawdown journalier max configuré
    pub fn get_max_daily_drawdown(&self) -> f64 {
        self.constitution.trading.max_daily_drawdown_percent
    }

    /// Retourne le nombre max de positions configuré
    pub fn get_max_positions(&self) -> u32 {
        self.constitution.trading.max_concurrent_positions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_request() -> TradeValidationRequest {
        TradeValidationRequest {
            id: Uuid::new_v4(),
            symbol: "XAUUSD".to_string(),
            action: "BUY".to_string(),
            volume: 0.5,
            stop_loss: Some(2050.0),
            take_profit: Some(2120.0),
            current_price: 2080.0,
            account_balance: 100000.0,
            open_positions_count: 0,
            daily_drawdown_percent: 0.0,
        }
    }

    #[test]
    fn test_valid_trade() {
        let constitution = Constitution::default();
        let validator = TradeValidator::new(constitution);
        let request = create_test_request();

        let result = validator.validate(&request);
        assert!(result.allowed);
        assert!(result.reason.is_none());
    }

    #[test]
    fn test_missing_stop_loss() {
        let constitution = Constitution::default();
        let validator = TradeValidator::new(constitution);
        let mut request = create_test_request();
        request.stop_loss = None;

        let result = validator.validate(&request);
        assert!(!result.allowed);
        assert!(result.reason.unwrap().contains("Stop Loss"));
    }

    #[test]
    fn test_max_positions_reached() {
        let constitution = Constitution::default();
        let validator = TradeValidator::new(constitution);
        let mut request = create_test_request();
        request.open_positions_count = 3;

        let result = validator.validate(&request);
        assert!(!result.allowed);
        assert!(result.reason.unwrap().contains("positions"));
    }
}
