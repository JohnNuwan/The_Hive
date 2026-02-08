//! Serveur HTTP Axum — Interface REST du Kernel
//!
//! Expose des endpoints critiques pour :
//! - Health check (monitoring)
//! - Validation de trades (Loi 2)
//! - Gestion du Kill-Switch (activation/reset)
//! - Consultation de la Constitution
//! - Audit Trail (Black Box)

use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{error, info};

use crate::audit::AuditTrail;
use crate::kill_switch::{KillSwitch, KillSwitchStatus};
use crate::laws::Constitution;
use crate::validator::{TradeValidationRequest, TradeValidator, ValidationResult};

// ═══════════════════════════════════════════════════════════════════════════════
// STATE PARTAGÉ
// ═══════════════════════════════════════════════════════════════════════════════

/// État partagé entre tous les handlers Axum (Arc pour concurrence)
#[derive(Clone)]
pub struct AppState {
    pub validator: Arc<Mutex<TradeValidator>>,
    pub kill_switch: Arc<Mutex<KillSwitch>>,
    pub constitution: Arc<Mutex<Constitution>>,
    pub audit_trail: Arc<Mutex<AuditTrail>>,
}

// ═══════════════════════════════════════════════════════════════════════════════
// MODÈLES API
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub message: String,
    pub kill_switch_active: bool,
    pub constitution_version: String,
    pub audit_records: usize,
}

#[derive(Debug, Deserialize)]
pub struct KillSwitchRequest {
    pub action: String, // "activate" ou "reset"
    pub reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct KillSwitchResponse {
    pub success: bool,
    pub status: KillSwitchStatus,
    pub message: String,
}

// ═══════════════════════════════════════════════════════════════════════════════
// HANDLERS
// ═══════════════════════════════════════════════════════════════════════════════

/// GET /health — Vérifie l'état opérationnel du Kernel
pub async fn health_check(State(state): State<AppState>) -> Json<HealthResponse> {
    let ks = state.kill_switch.lock().await;
    let audit = state.audit_trail.lock().await;
    let constitution = state.constitution.lock().await;

    Json(HealthResponse {
        status: "operational".to_string(),
        message: "EVA Kernel is running.".to_string(),
        kill_switch_active: ks.is_active(),
        constitution_version: constitution.version.clone(),
        audit_records: audit.len(),
    })
}

/// POST /validate — Valide un trade selon la Constitution (Loi 2)
pub async fn validate_trade(
    State(state): State<AppState>,
    Json(request): Json<TradeValidationRequest>,
) -> impl IntoResponse {
    // Vérifier le Kill-Switch en premier
    let ks = state.kill_switch.lock().await;
    if ks.is_active() {
        error!("Trade rejeté: Kill-Switch actif.");
        return (
            StatusCode::FORBIDDEN,
            Json(ValidationResult {
                allowed: false,
                reason: Some("KILL_SWITCH_ACTIVE — Tout trading est bloqué".to_string()),
                law_reference: Some("Loi 2 — Kill-Switch".to_string()),
                risk_percent: 0.0,
                checks: vec![],
            }),
        );
    }
    drop(ks); // Libérer le lock avant la validation

    // Valider le trade
    let validator = state.validator.lock().await;
    let result = validator.validate(&request);
    drop(validator);

    // Enregistrer dans l'Audit Trail (Black Box)
    let mut audit = state.audit_trail.lock().await;
    audit.record(
        "kernel",
        if result.allowed {
            "TRADE_VALIDATED"
        } else {
            "TRADE_REJECTED"
        },
        serde_json::json!({
            "trade_id": request.id.to_string(),
            "symbol": request.symbol,
            "action": request.action,
            "volume": request.volume,
            "allowed": result.allowed,
            "risk_percent": result.risk_percent,
            "reason": result.reason,
        }),
    );

    if result.allowed {
        info!("✅ Trade validé: {} {} {}", request.symbol, request.action, request.volume);
        (StatusCode::OK, Json(result))
    } else {
        info!("❌ Trade rejeté: {:?}", result.reason);
        (StatusCode::BAD_REQUEST, Json(result))
    }
}

/// POST /kill-switch — Active ou désactive le Kill-Switch
pub async fn manage_kill_switch(
    State(state): State<AppState>,
    Json(request): Json<KillSwitchRequest>,
) -> impl IntoResponse {
    let mut ks = state.kill_switch.lock().await;
    let mut audit = state.audit_trail.lock().await;

    match request.action.as_str() {
        "activate" => {
            let reason = request
                .reason
                .unwrap_or_else(|| "Activation manuelle via HTTP".to_string());
            ks.activate(&reason);

            audit.record(
                "kernel",
                "KILL_SWITCH_ACTIVATED",
                serde_json::json!({ "reason": reason }),
            );

            (
                StatusCode::OK,
                Json(KillSwitchResponse {
                    success: true,
                    status: ks.get_status(),
                    message: "Kill-Switch activé.".to_string(),
                }),
            )
        }
        "reset" => {
            ks.reset();

            audit.record(
                "kernel",
                "KILL_SWITCH_RESET",
                serde_json::json!({ "admin": true }),
            );

            (
                StatusCode::OK,
                Json(KillSwitchResponse {
                    success: true,
                    status: ks.get_status(),
                    message: "Kill-Switch désactivé.".to_string(),
                }),
            )
        }
        _ => (
            StatusCode::BAD_REQUEST,
            Json(KillSwitchResponse {
                success: false,
                status: ks.get_status(),
                message: format!("Action '{}' inconnue. Utilisez 'activate' ou 'reset'.", request.action),
            }),
        ),
    }
}

/// GET /kill-switch — État actuel du Kill-Switch
pub async fn get_kill_switch_status(State(state): State<AppState>) -> Json<KillSwitchStatus> {
    let ks = state.kill_switch.lock().await;
    Json(ks.get_status())
}

/// GET /constitution — Retourne la Constitution complète
pub async fn get_constitution(State(state): State<AppState>) -> Json<Constitution> {
    let constitution = state.constitution.lock().await;
    Json((*constitution).clone())
}

/// GET /audit — Retourne les derniers enregistrements de l'Audit Trail
pub async fn get_audit_trail(State(state): State<AppState>) -> impl IntoResponse {
    let audit = state.audit_trail.lock().await;
    let records = audit.get_recent(50);
    let serialized: Vec<_> = records.into_iter().cloned().collect();
    (StatusCode::OK, Json(serialized))
}

// ═══════════════════════════════════════════════════════════════════════════════
// LANCEMENT DU SERVEUR
// ═══════════════════════════════════════════════════════════════════════════════

/// Démarre le serveur Axum du Kernel sur le port 8080
pub async fn start_kernel_server(
    validator: Arc<Mutex<TradeValidator>>,
    kill_switch: Arc<Mutex<KillSwitch>>,
    constitution: Arc<Mutex<Constitution>>,
    audit_trail: Arc<Mutex<AuditTrail>>,
) {
    let state = AppState {
        validator,
        kill_switch,
        constitution,
        audit_trail,
    };

    let app = Router::new()
        .route("/health", get(health_check))
        .route("/validate", post(validate_trade))
        .route("/kill-switch", get(get_kill_switch_status).post(manage_kill_switch))
        .route("/constitution", get(get_constitution))
        .route("/audit", get(get_audit_trail))
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], 8080));
    info!("🛡️ EVA Kernel HTTP (Axum) listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("Failed to bind Kernel HTTP port 8080");

    axum::serve(listener, app)
        .await
        .expect("Kernel HTTP server crashed");
}
