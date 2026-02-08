//! Module Laws - Parsing et validation de la Constitution
//!
//! Charge le fichier TOML de la Constitution depuis The Tablet
//! et expose les règles pour validation.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use thiserror::Error;

/// Erreurs liées au chargement de la Constitution
#[derive(Error, Debug)]
pub enum ConstitutionError {
    #[error("Fichier non trouvé: {0}")]
    FileNotFound(String),

    #[error("Erreur de parsing TOML: {0}")]
    ParseError(#[from] toml::de::Error),

    #[error("Erreur IO: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Hash invalide - Constitution potentiellement modifiée")]
    InvalidHash,
}

/// Loi fondamentale de la Constitution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Law {
    pub id: u8,
    pub name: String,
    pub description: String,
    pub priority: u8,
    pub enforcement: String,
}

/// Règles d'Engagement (ROE) pour le Trading
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TradingRoe {
    /// Risque maximum par trade en pourcentage
    pub max_risk_per_trade_percent: f64,
    /// Drawdown journalier maximum
    pub max_daily_drawdown_percent: f64,
    /// Drawdown total maximum
    pub max_total_drawdown_percent: f64,
    /// Nombre maximum de positions ouvertes
    pub max_concurrent_positions: u32,
    /// Stop Loss obligatoire
    pub require_stop_loss: bool,
    /// Nombre de pertes consécutives avant Anti-Tilt
    pub anti_tilt_consecutive_losses: u32,
    /// Durée Anti-Tilt en heures
    pub anti_tilt_duration_hours: u32,
    /// Minutes avant/après news à éviter
    pub news_filter_minutes: u32,
}

/// Règles d'Engagement pour la Sécurité
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SecurityRoe {
    /// Tentatives de login max avant blocage
    pub max_login_attempts: u32,
    /// Durée du blocage en minutes
    pub lockout_duration_minutes: u32,
    /// Température GPU critique
    pub gpu_temp_critical_celsius: f64,
}

/// Constitution complète
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Constitution {
    pub version: String,
    pub last_updated: String,
    pub laws: Vec<Law>,
    pub roe: HashMap<String, toml::Value>,
    #[serde(default)]
    pub trading: TradingRoe,
    #[serde(default)]
    pub security: SecurityRoe,
}

impl Default for Constitution {
    fn default() -> Self {
        Self {
            version: "1.0.0".to_string(),
            last_updated: "2026-02-05".to_string(),
            laws: vec![
                Law {
                    id: 0,
                    name: "Intégrité Systémique".to_string(),
                    description: "Préserver l'intégrité de THE HIVE".to_string(),
                    priority: 100,
                    enforcement: "hardware".to_string(),
                },
                Law {
                    id: 1,
                    name: "Épanouissement Humain".to_string(),
                    description: "Bien-être du Maître prioritaire".to_string(),
                    priority: 95,
                    enforcement: "software".to_string(),
                },
                Law {
                    id: 2,
                    name: "Protection du Capital".to_string(),
                    description: "Ne jamais risquer plus que les limites".to_string(),
                    priority: 90,
                    enforcement: "kernel".to_string(),
                },
            ],
            roe: HashMap::new(),
            trading: TradingRoe {
                max_risk_per_trade_percent: 1.0,
                max_daily_drawdown_percent: 4.0,
                max_total_drawdown_percent: 8.0,
                max_concurrent_positions: 3,
                require_stop_loss: true,
                anti_tilt_consecutive_losses: 2,
                anti_tilt_duration_hours: 24,
                news_filter_minutes: 30,
            },
            security: SecurityRoe {
                max_login_attempts: 3,
                lockout_duration_minutes: 30,
                gpu_temp_critical_celsius: 90.0,
            },
        }
    }
}

impl Constitution {
    /// Charge la Constitution depuis un fichier TOML
    pub fn load(path: &Path) -> Result<Self, ConstitutionError> {
        if !path.exists() {
            return Err(ConstitutionError::FileNotFound(
                path.to_string_lossy().to_string(),
            ));
        }

        let content = fs::read_to_string(path)?;
        let constitution: Constitution = toml::from_str(&content)?;

        Ok(constitution)
    }

    /// Retourne le temps de modification du fichier
    pub fn get_modification_time(path: &Path) -> Option<std::time::SystemTime> {
        fs::metadata(path).ok()?.modified().ok()
    }

    /// Vérifie si un risque de trade est autorisé selon Loi 2
    pub fn is_trade_risk_allowed(&self, risk_percent: f64) -> bool {
        risk_percent <= self.trading.max_risk_per_trade_percent
    }

    /// Vérifie si le drawdown journalier permet de trader
    pub fn is_daily_drawdown_ok(&self, dd_percent: f64) -> bool {
        dd_percent < self.trading.max_daily_drawdown_percent
    }

    /// Vérifie si le nombre de positions permet d'en ouvrir une nouvelle
    pub fn can_open_position(&self, current_count: u32) -> bool {
        current_count < self.trading.max_concurrent_positions
    }

    /// Retourne la Loi par son ID
    pub fn get_law(&self, id: u8) -> Option<&Law> {
        self.laws.iter().find(|l| l.id == id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_constitution() {
        let c = Constitution::default();
        assert_eq!(c.laws.len(), 3);
        assert_eq!(c.trading.max_risk_per_trade_percent, 1.0);
    }

    #[test]
    fn test_trade_risk_validation() {
        let c = Constitution::default();
        assert!(c.is_trade_risk_allowed(0.5));
        assert!(c.is_trade_risk_allowed(1.0));
        assert!(!c.is_trade_risk_allowed(1.5));
    }

    #[test]
    fn test_position_limit() {
        let c = Constitution::default();
        assert!(c.can_open_position(0));
        assert!(c.can_open_position(2));
        assert!(!c.can_open_position(3));
    }
}
