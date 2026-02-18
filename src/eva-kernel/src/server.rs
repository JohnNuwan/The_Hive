//! Serveur HTTP Axum — Interface REST du Kernel
//!
//! Expose des endpoints critiques pour :
//! - Health check (monitoring)
//! - Validation de trades (Loi 2)
//! - Gestion du Kill-Switch (activation/reset)
//! - Consultation de la Constitution
//! - Audit Trail (Black Box)

use axum::{
    extract::{Json, Request, State},
    http::{HeaderMap, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
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
    pub kernel_secret_key: String,
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

/// Middleware d'authentification par API Key
async fn auth_middleware(
    State(state): State<AppState>,
    headers: HeaderMap,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    if let Some(key) = headers.get("X-EVA-KERNEL-KEY") {
        if key == state.kernel_secret_key.as_str() {
            return Ok(next.run(request).await);
        }
    }

    Err(StatusCode::UNAUTHORIZED)
}

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

/// Crée le routeur Axum avec la configuration de sécurité
pub fn create_router(state: AppState) -> Router {
    let middleware = middleware::from_fn_with_state(state.clone(), auth_middleware);

    Router::new()
        // Routes protégées par authentification
        .route("/validate", post(validate_trade))
        .route("/kill-switch", get(get_kill_switch_status).post(manage_kill_switch))
        .route("/constitution", get(get_constitution))
        .route("/audit", get(get_audit_trail))
        .route_layer(middleware)
        // Routes publiques (Doivent être ajoutées APRES le layer pour ne pas être affectées)
        .route("/health", get(health_check))
        .with_state(state)
}

/// Démarre le serveur Axum du Kernel sur le port 8080
pub async fn start_kernel_server(
    validator: Arc<Mutex<TradeValidator>>,
    kill_switch: Arc<Mutex<KillSwitch>>,
    constitution: Arc<Mutex<Constitution>>,
    audit_trail: Arc<Mutex<AuditTrail>>,
) {
    let kernel_secret_key = std::env::var("EVA_KERNEL_SECRET_KEY")
        .expect("CRITICAL: EVA_KERNEL_SECRET_KEY must be set");

    let state = AppState {
        validator,
        kill_switch,
        constitution,
        audit_trail,
        kernel_secret_key,
    };

    let app = create_router(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], 8080));
    info!("🛡️ EVA Kernel HTTP (Axum) listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("Failed to bind Kernel HTTP port 8080");

    axum::serve(listener, app)
        .await
        .expect("Kernel HTTP server crashed");
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use tower::ServiceExt; // for oneshot

    #[tokio::test]
    async fn test_health_check_public() {
        let state = AppState {
            validator: Arc::new(Mutex::new(TradeValidator::new(Constitution::default()))),
            kill_switch: Arc::new(Mutex::new(KillSwitch::new(5.0))),
            constitution: Arc::new(Mutex::new(Constitution::default())),
            audit_trail: Arc::new(Mutex::new(AuditTrail::new(100))),
            kernel_secret_key: "test_key".to_string(),
        };

        let app = create_router(state);

        let response = app
            .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_validate_trade_unauthorized() {
        let state = AppState {
            validator: Arc::new(Mutex::new(TradeValidator::new(Constitution::default()))),
            kill_switch: Arc::new(Mutex::new(KillSwitch::new(5.0))),
            constitution: Arc::new(Mutex::new(Constitution::default())),
            audit_trail: Arc::new(Mutex::new(AuditTrail::new(100))),
            kernel_secret_key: "test_key".to_string(),
        };

        let app = create_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/validate")
                    .header("content-type", "application/json")
                    .body(Body::from("{}")) // Body doesn't matter for auth check
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_validate_trade_authorized() {
        let state = AppState {
            validator: Arc::new(Mutex::new(TradeValidator::new(Constitution::default()))),
            kill_switch: Arc::new(Mutex::new(KillSwitch::new(5.0))),
            constitution: Arc::new(Mutex::new(Constitution::default())),
            audit_trail: Arc::new(Mutex::new(AuditTrail::new(100))),
            kernel_secret_key: "test_secret_123".to_string(),
        };

        let app = create_router(state);

        let request_body = serde_json::to_string(&TradeValidationRequest {
            id: uuid::Uuid::new_v4(),
            symbol: "BTCUSD".to_string(),
            action: "BUY".to_string(),
            volume: 0.1,
            stop_loss: Some(49000.0),
            take_profit: Some(51000.0),
            current_price: 50000.0,
            account_balance: 100000.0,
            open_positions_count: 0,
            daily_drawdown_percent: 0.0,
        }).unwrap();

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/validate")
                    .header("content-type", "application/json")
                    .header("X-EVA-KERNEL-KEY", "test_secret_123")
                    .body(Body::from(request_body))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_ne!(response.status(), StatusCode::UNAUTHORIZED);
    }
}
