//! THE HIVE - Rust Kernel Types
//! ═══════════════════════════════════════════════════════════════════════════════
//! Types et structures pour le Kernel de sécurité (eva-kernel crate).
//! Ces structures sont compilées en binaire et vérifiées par The Tablet.
//! 
//! Enforcement: Constitution Laws 0, 2 (Intégrité Système, Protection Capital)

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::net::IpAddr;

// ═══════════════════════════════════════════════════════════════════════════════
// CONSTITUTION TYPES
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration des Lois chargée depuis The Tablet (Lois.toml).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstitutionConfig {
    pub law_zero: LawZeroConfig,
    pub law_two: LawTwoConfig,
    pub roe_trading: RoeTradingConfig,
}

/// Loi 0: Intégrité Systémique (Survival)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LawZeroConfig {
    /// Température GPU max avant shutdown (défaut: 90°C)
    pub max_gpu_temp_celsius: f32,
    /// Durée max au-dessus du seuil avant action (défaut: 5s)
    pub temp_critical_duration_secs: u32,
    /// Hash SHA-512 du kernel validé
    pub kernel_hash: String,
}

impl Default for LawZeroConfig {
    fn default() -> Self {
        Self {
            max_gpu_temp_celsius: 90.0,
            temp_critical_duration_secs: 5,
            kernel_hash: String::new(),
        }
    }
}

/// Loi 2: Protection du Capital (Risk Management)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LawTwoConfig {
    /// Max Daily Drawdown en pourcentage (défaut: 4.0%)
    pub max_daily_drawdown_percent: f64,
    /// Max Total Drawdown en pourcentage (défaut: 8.0%)
    pub max_total_drawdown_percent: f64,
    /// Risque max par trade en pourcentage (défaut: 1.0%)
    pub max_single_trade_risk_percent: f64,
    /// Nombre max de positions ouvertes simultanément (défaut: 3)
    pub max_open_positions: u32,
}

impl Default for LawTwoConfig {
    fn default() -> Self {
        Self {
            max_daily_drawdown_percent: 4.0,
            max_total_drawdown_percent: 8.0,
            max_single_trade_risk_percent: 1.0,
            max_open_positions: 3,
        }
    }
}

/// Rules of Engagement pour le Trading
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoeTradingConfig {
    /// Minutes avant/après news high impact pour interdire trading
    pub news_filter_minutes: u32,
    /// Nombre de pertes consécutives déclenchant anti-tilt
    pub anti_tilt_loss_count: u32,
    /// Durée de suspension anti-tilt en heures
    pub anti_tilt_duration_hours: u32,
    /// Slippage max toléré en points
    pub max_slippage_points: i32,
}

impl Default for RoeTradingConfig {
    fn default() -> Self {
        Self {
            news_filter_minutes: 30,
            anti_tilt_loss_count: 2,
            anti_tilt_duration_hours: 24,
            max_slippage_points: 5,
        }
    }
}


// ═══════════════════════════════════════════════════════════════════════════════
// TRADING TYPES (Financial Watchdog)
// ═══════════════════════════════════════════════════════════════════════════════

/// Action de trading
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradeAction {
    Buy,
    Sell,
}

/// Demande de validation d'un ordre par le Kernel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeValidationRequest {
    pub request_id: String,
    pub account_id: String,
    pub symbol: String,
    pub action: TradeAction,
    pub volume: f64,
    pub stop_loss_price: f64,
    pub take_profit_price: Option<f64>,
    pub risk_percent: f64,
    pub timestamp: DateTime<Utc>,
}

/// Résultat de validation d'un ordre
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeValidationResult {
    pub request_id: String,
    pub approved: bool,
    pub rejection_reason: Option<RejectionReason>,
    pub risk_check: RiskCheckResult,
    pub validated_at: DateTime<Utc>,
}

/// Raisons de rejet d'un ordre
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RejectionReason {
    RiskTooHigh { requested: f64, max_allowed: f64 },
    DailyDrawdownLimit { current: f64, limit: f64 },
    TotalDrawdownLimit { current: f64, limit: f64 },
    MaxPositionsReached { current: u32, limit: u32 },
    NewsFilterActive { next_window: DateTime<Utc> },
    AntiTiltActive { expires_at: DateTime<Utc> },
    BiometricAlert { reason: String },
    MissingStopLoss,
    SymbolNotWhitelisted,
    MarketClosed,
}

/// Résultat du risk check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskCheckResult {
    pub daily_drawdown_percent: f64,
    pub total_drawdown_percent: f64,
    pub current_risk_percent: f64,
    pub open_positions: u32,
    pub consecutive_losses: u32,
    pub trading_allowed: bool,
}

/// État du compte MT5 lu par le Watchdog
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountState {
    pub login: i64,
    pub server: String,
    pub balance: f64,
    pub equity: f64,
    pub margin: f64,
    pub free_margin: f64,
    pub profit: f64,
    pub positions_count: u32,
    pub last_update: DateTime<Utc>,
}

/// Événement Kill-Switch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KillSwitchEvent {
    pub triggered_at: DateTime<Utc>,
    pub reason: KillSwitchReason,
    pub account_state: AccountState,
    pub action_taken: KillSwitchAction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KillSwitchReason {
    DailyLossLimit { loss_percent: f64 },
    TotalLossLimit { loss_percent: f64 },
    ThermalEmergency { temp_celsius: f32 },
    ManualTrigger,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KillSwitchAction {
    CloseAllPositions,
    KillMt5Process,
    ShutdownTrading,
    EmergencyNotification,
}


// ═══════════════════════════════════════════════════════════════════════════════
// HARDWARE MONITORING TYPES (The Keeper Integration)
// ═══════════════════════════════════════════════════════════════════════════════

/// Métriques GPU temps réel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuMetrics {
    pub temperature_celsius: f32,
    pub power_watts: f32,
    pub memory_used_mb: u64,
    pub memory_total_mb: u64,
    pub utilization_percent: f32,
    pub fan_speed_percent: Option<f32>,
    pub timestamp: DateTime<Utc>,
}

impl GpuMetrics {
    /// Vérifie si la température dépasse le seuil critique (Loi 0)
    pub fn is_temperature_critical(&self, threshold: f32) -> bool {
        self.temperature_celsius > threshold
    }
}

/// Métriques système globales
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub gpu: GpuMetrics,
    pub cpu_load_percent: f32,
    pub ram_used_mb: u64,
    pub ram_total_mb: u64,
    pub disk_used_gb: f64,
    pub disk_total_gb: f64,
    pub timestamp: DateTime<Utc>,
}

/// Alerte thermique
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalAlert {
    pub level: ThermalLevel,
    pub current_temp: f32,
    pub threshold: f32,
    pub duration_above_threshold_secs: u32,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThermalLevel {
    Normal,
    Warning,  // > 80°C
    Critical, // > 90°C
    Emergency, // > 95°C - Shutdown immédiat
}


// ═══════════════════════════════════════════════════════════════════════════════
// SECURITY TYPES (The Sentinel Integration)
// ═══════════════════════════════════════════════════════════════════════════════

/// Événement de sécurité détecté
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityEvent {
    pub event_id: String,
    pub event_type: SecurityEventType,
    pub severity: Severity,
    pub source_ip: Option<IpAddr>,
    pub target_service: Option<String>,
    pub description: String,
    pub action_taken: SecurityAction,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityEventType {
    IntrusionAttempt,
    AuthenticationFailure,
    SuspiciousConnection,
    MalwareDetected,
    BruteForceAttempt,
    PortScan,
    UnauthorizedAccess,
    DataExfiltration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Severity {
    Info,
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityAction {
    Logged,
    Blocked { duration_secs: u64 },
    Quarantined,
    Alerted,
    ProcessKilled { pid: u32 },
}


// ═══════════════════════════════════════════════════════════════════════════════
// AUDIT TRAIL TYPES (Black Box)
// ═══════════════════════════════════════════════════════════════════════════════

/// Enregistrement d'audit immutable
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditRecord {
    pub sequence_id: u64,
    pub event_type: String,
    pub actor: String,
    pub action: String,
    pub target: Option<String>,
    pub context: serde_json::Value,
    pub prev_hash: String,
    pub record_hash: String,
    pub timestamp: DateTime<Utc>,
}

impl AuditRecord {
    /// Calcule le hash SHA-512 de l'enregistrement
    pub fn compute_hash(&self) -> String {
        use sha2::{Sha512, Digest};
        
        let input = format!(
            "{}|{}|{}|{}|{}|{}",
            self.timestamp.to_rfc3339(),
            self.event_type,
            self.actor,
            self.action,
            self.target.as_deref().unwrap_or(""),
            self.prev_hash
        );
        
        let mut hasher = Sha512::new();
        hasher.update(input.as_bytes());
        hex::encode(hasher.finalize())
    }
    
    /// Vérifie l'intégrité de la chaîne d'audit
    pub fn verify_chain_integrity(&self) -> bool {
        self.record_hash == self.compute_hash()
    }
}


// ═══════════════════════════════════════════════════════════════════════════════
// INTER-PROCESS COMMUNICATION TYPES
// ═══════════════════════════════════════════════════════════════════════════════

/// Message entrant vers le Kernel (depuis Python via FFI/Channel)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KernelRequest {
    ValidateTrade(TradeValidationRequest),
    GetRiskStatus { account_id: String },
    TriggerKillSwitch { reason: String },
    GetSystemHealth,
    VerifyKernelIntegrity,
    RecordAudit(AuditRecord),
}

/// Réponse du Kernel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KernelResponse {
    TradeValidation(TradeValidationResult),
    RiskStatus(RiskCheckResult),
    KillSwitchTriggered(KillSwitchEvent),
    SystemHealth { healthy: bool, details: String },
    IntegrityCheck { valid: bool, expected_hash: String, actual_hash: String },
    AuditRecorded { sequence_id: u64 },
    Error { code: String, message: String },
}


// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_law_two_defaults() {
        let law = LawTwoConfig::default();
        assert_eq!(law.max_daily_drawdown_percent, 4.0);
        assert_eq!(law.max_total_drawdown_percent, 8.0);
        assert_eq!(law.max_single_trade_risk_percent, 1.0);
    }

    #[test]
    fn test_gpu_critical_temp() {
        let gpu = GpuMetrics {
            temperature_celsius: 92.0,
            power_watts: 350.0,
            memory_used_mb: 20000,
            memory_total_mb: 24576,
            utilization_percent: 98.0,
            fan_speed_percent: Some(100.0),
            timestamp: Utc::now(),
        };
        
        assert!(gpu.is_temperature_critical(90.0));
        assert!(!gpu.is_temperature_critical(95.0));
    }

    #[test]
    fn test_audit_hash_consistency() {
        let record = AuditRecord {
            sequence_id: 1,
            event_type: "TEST".to_string(),
            actor: "unit_test".to_string(),
            action: "create".to_string(),
            target: Some("test_target".to_string()),
            context: serde_json::json!({}),
            prev_hash: "GENESIS".to_string(),
            record_hash: String::new(),
            timestamp: Utc::now(),
        };
        
        let hash1 = record.compute_hash();
        let hash2 = record.compute_hash();
        
        assert_eq!(hash1, hash2);
        assert_eq!(hash1.len(), 128); // SHA-512 = 64 bytes = 128 hex chars
    }
}
