//! EVA Kernel — Point d'entrée principal
//!
//! Le Kernel est le composant de sécurité critique de THE HIVE.
//! Il valide les actions selon la Constitution, maintient l'audit trail,
//! et intercepte les signaux via Redis + MQTT en parallèle du serveur Axum.

mod audit;
mod kill_switch;
mod laws;
mod protocols;
mod server;
mod validator;

use std::path::PathBuf;
use chrono::{TimeZone, Utc};
use serde_json::Value;
use tracing::{error, info, warn, Level};
use tracing_subscriber::FmtSubscriber;
use uuid::Uuid;

use crate::kill_switch::KillSwitch;
use crate::laws::Constitution;
use crate::server::start_kernel_server;
use futures::StreamExt;
use crate::validator::TradeValidator;

fn extract_string_field(value: &Value, keys: &[&str]) -> Option<String> {
    for key in keys {
        if let Some(field) = value.get(*key) {
            let extracted = match field {
                Value::String(text) => Some(text.trim().to_string()),
                Value::Number(number) => Some(number.to_string()),
                Value::Bool(flag) => Some(flag.to_string()),
                _ => None,
            };

            if let Some(text) = extracted {
                if !text.is_empty() {
                    return Some(text);
                }
            }
        }
    }

    for container in ["data", "payload", "metadata", "context", "details"] {
        if let Some(nested) = value.get(container) {
            if let Some(text) = extract_string_field(nested, keys) {
                return Some(text);
            }
        }
    }

    None
}

fn title_case(raw: &str) -> String {
    raw.split_whitespace()
        .map(|segment| {
            let mut chars = segment.chars();
            match chars.next() {
                Some(first) => format!("{}{}", first.to_uppercase(), chars.as_str().to_lowercase()),
                None => String::new(),
            }
        })
        .filter(|segment| !segment.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

fn normalize_agent_name(raw: &str) -> String {
    let normalized = raw
        .trim()
        .replace(['.', '_', '-', '/'], " ")
        .to_lowercase();

    if normalized.contains("core") {
        "EVA Core".to_string()
    } else if normalized.contains("banker") {
        "Banker".to_string()
    } else if normalized.contains("sentinel") {
        "Sentinel".to_string()
    } else if normalized.contains("compliance") {
        "Compliance".to_string()
    } else if normalized.contains("accountant") {
        "Accountant".to_string()
    } else if normalized.contains("researcher") {
        "Researcher".to_string()
    } else if normalized.contains("wraith") {
        "Wraith".to_string()
    } else if normalized.contains("muse") {
        "Muse".to_string()
    } else if normalized.contains("shadow") {
        "Shadow".to_string()
    } else if normalized.contains("sage") {
        "Sage".to_string()
    } else if normalized.contains("lab") {
        "Lab".to_string()
    } else if normalized.contains("rwa") {
        "RWA".to_string()
    } else if normalized.contains("kernel") {
        "Kernel".to_string()
    } else {
        title_case(&normalized)
    }
}

fn infer_agent_from_channel(channel_name: &str) -> String {
    let parts: Vec<&str> = channel_name
        .split(|c| matches!(c, '.' | ':' | '/' | '-'))
        .filter(|part| !part.trim().is_empty())
        .collect();

    let candidate = if parts.first().is_some_and(|part| part.eq_ignore_ascii_case("eva")) && parts.len() > 1 {
        parts[1]
    } else {
        parts.first().copied().unwrap_or("system")
    };

    normalize_agent_name(candidate)
}

fn normalize_feed_type(raw_type: Option<&str>) -> &'static str {
    match raw_type.unwrap_or("").trim().to_ascii_lowercase().as_str() {
        "alert" | "error" | "critical" | "fatal" => "error",
        "event" | "action" | "trade" | "order" => "action",
        "request" | "thought" | "analysis" | "reasoning" => "thought",
        "response" | "result" | "success" | "done" => "result",
        "" => "message",
        _ => "message",
    }
}

fn normalize_timestamp(raw_timestamp: Option<String>) -> String {
    if let Some(timestamp) = raw_timestamp {
        if chrono::DateTime::parse_from_rfc3339(&timestamp).is_ok() {
            return timestamp;
        }

        if let Ok(epoch) = timestamp.parse::<i64>() {
            let parsed = if epoch > 1_000_000_000_000 {
                Utc.timestamp_millis_opt(epoch).single()
            } else {
                Utc.timestamp_opt(epoch, 0).single()
            };

            if let Some(date) = parsed {
                return date.to_rfc3339();
            }
        }
    }

    Utc::now().to_rfc3339()
}

fn normalize_feed_message(channel_name: &str, payload_str: &str) -> String {
    if let Ok(value) = serde_json::from_str::<Value>(payload_str) {
        let raw_agent = extract_string_field(
            &value,
            &["source_agent", "agent", "source", "sender", "service", "origin", "role"],
        );
        let agent = raw_agent
            .as_deref()
            .map(normalize_agent_name)
            .unwrap_or_else(|| infer_agent_from_channel(channel_name));

        let raw_target = extract_string_field(
            &value,
            &["target_agent", "target", "recipient", "destination"],
        );
        let target = raw_target
            .as_deref()
            .map(normalize_agent_name)
            .filter(|name| !name.is_empty());

        let company = extract_string_field(&value, &["company", "team", "swarm", "domain"])
            .unwrap_or_else(|| "Hive Swarm".to_string());

        let content = extract_string_field(
            &value,
            &["action", "content", "message", "text", "summary", "result", "reason", "description"],
        )
        .unwrap_or_else(|| payload_str.to_string());

        let message_type = normalize_feed_type(
            extract_string_field(
                &value,
                &["type", "message_type", "event_type", "level", "kind"],
            )
            .as_deref(),
        );

        let timestamp = normalize_timestamp(extract_string_field(
            &value,
            &["timestamp", "created_at", "time", "datetime", "occurred_at", "date"],
        ));

        let id = extract_string_field(
            &value,
            &["id", "event_id", "request_id", "uuid", "trace_id"],
        )
        .unwrap_or_else(|| Uuid::new_v4().to_string());

        return serde_json::json!({
            "id": id,
            "agent": agent,
            "company": company,
            "type": message_type,
            "content": content,
            "timestamp": timestamp,
            "target": target,
        })
        .to_string();
    }

    serde_json::json!({
        "id": Uuid::new_v4().to_string(),
        "agent": infer_agent_from_channel(channel_name),
        "company": "System",
        "type": "message",
        "content": payload_str,
        "timestamp": Utc::now().to_rfc3339(),
    })
    .to_string()
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configuration du logging
    let _subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_target(false)
        .pretty()
        .init();

    info!("🔒 EVA Kernel démarrage...");

    // ═══════════════════════════════════════════════════════════════════
    // CHARGER LA CONSTITUTION
    // ═══════════════════════════════════════════════════════════════════
    let constitution_path = std::env::var("CONSTITUTION_PATH")
        .unwrap_or_else(|_| "/mnt/tablet/constitution.toml".to_string());

    info!("📜 Chargement Constitution: {}", constitution_path);

    let constitution = match Constitution::load(&PathBuf::from(&constitution_path)) {
        Ok(c) => {
            info!(
                "✅ Constitution chargée: {} lois, {} ROE",
                c.laws.len(),
                c.roe.len()
            );
            c
        }
        Err(e) => {
            warn!(
                "⚠️ Constitution non trouvée, utilisation des valeurs par défaut: {}",
                e
            );
            Constitution::default()
        }
    };

    // ═══════════════════════════════════════════════════════════════════
    // CRÉER LES COMPOSANTS CRITIQUES (PARTAGÉS)
    // ═══════════════════════════════════════════════════════════════════
    use std::sync::Arc;
    use tokio::sync::{broadcast, Mutex};
    use crate::audit::AuditTrail;

    // Créer un canal de broadcast pour les messages d'agents (Redis -> WebSocket)
    let (tx, _rx) = broadcast::channel::<String>(1024);
    let tx_clone = tx.clone();

    let audit_path = std::path::PathBuf::from("/mnt/black_box/audit.json");
    let mut audit_trail = AuditTrail::load_from_disk(&audit_path, 10_000)
        .unwrap_or_else(|_| AuditTrail::new(10_000));
    audit_trail.set_persistence_path(audit_path.clone());

    let validator = TradeValidator::new(constitution.clone());
    let kill_switch = KillSwitch::new(constitution.trading.max_daily_drawdown_percent);

    let constitution_arc = Arc::new(Mutex::new(constitution.clone()));
    let validator_arc = Arc::new(Mutex::new(validator));
    let kill_switch_arc = Arc::new(Mutex::new(kill_switch));
    let audit_trail_arc = Arc::new(Mutex::new(audit_trail));

    info!("✅ EVA Kernel prêt — Lancement des systèmes parallèles");

    // ═══════════════════════════════════════════════════════════════════
    // LANCER LE SERVEUR AXUM EN PARALLÈLE
    // ═══════════════════════════════════════════════════════════════════
    tokio::spawn(start_kernel_server(
        validator_arc.clone(),
        kill_switch_arc.clone(),
        constitution_arc.clone(),
        audit_trail_arc.clone(),
        tx_clone,
    ));

    // ═══════════════════════════════════════════════════════════════════
    // HOT-RELOAD CONSTITUTION (The Tablet Watchdog)
    // ═══════════════════════════════════════════════════════════════════
    let path_clone = PathBuf::from(&constitution_path);
    let const_clone = constitution_arc.clone();
    let valid_clone = validator_arc.clone();

    tokio::spawn(async move {
        let mut last_mod = Constitution::get_modification_time(&path_clone);
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            let current_mod = Constitution::get_modification_time(&path_clone);
            
            if current_mod != last_mod {
                info!("📜 Modification Constitution détectée. Hot-reloading...");
                if let Ok(new_const) = Constitution::load(&path_clone) {
                    let mut c = const_clone.lock().await;
                    *c = new_const.clone();
                    
                    let mut v = valid_clone.lock().await;
                    *v = TradeValidator::new(new_const);
                    
                    last_mod = current_mod;
                    info!("✅ Constitution rechargée à chaud avec succès.");
                } else {
                    error!("❌ Échec du rechargement de la Constitution (Erreur de parsing).");
                }
            }
        }
    });

    // ═══════════════════════════════════════════════════════════════════
    // BOUCLE D'INTERCEPTION Redis + MQTT
    // ═══════════════════════════════════════════════════════════════════
    let redis_url = std::env::var("REDIS_URL")
        .unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());

    // Connexion Redis (avec retry gracieux)
    let redis_result = redis::Client::open(redis_url.as_str());
    let redis_ok = match &redis_result {
        Ok(client) => {
            match client.get_multiplexed_async_connection().await {
                Ok(_con) => {
                    info!("✅ Redis connecté pour interception");
                    true
                }
                Err(e) => {
                    warn!("⚠️ Redis non disponible: {}. Mode dégradé.", e);
                    false
                }
            }
        }
        Err(e) => {
            warn!("⚠️ Redis URL invalide: {}. Mode dégradé.", e);
            false
        }
    };

    // Connexion MQTT (Neural Link Secondaire)
    let mqtt_host = std::env::var("MQTT_HOST").unwrap_or_else(|_| "localhost".to_string());
    let mqtt_port: u16 = std::env::var("MQTT_PORT")
        .unwrap_or_else(|_| "1883".to_string())
        .parse()
        .unwrap_or(1883);

    let mqtt_ok = {
        use rumqttc::{AsyncClient, MqttOptions, QoS};
        let mut mqttoptions = MqttOptions::new("eva_kernel", &mqtt_host, mqtt_port);
        mqttoptions.set_keep_alive(std::time::Duration::from_secs(5));

        match AsyncClient::new(mqttoptions, 10) {
            (client, mut eventloop) => {
                if let Err(e) = client
                    .subscribe("eva/banker/requests/critical", QoS::AtLeastOnce)
                    .await
                {
                    warn!("⚠️ MQTT subscribe échoué: {}", e);
                    false
                } else {
                    info!("✅ MQTT connecté — interception signaux critiques");

                    // Spawn MQTT listener
                    tokio::spawn(async move {
                        loop {
                            match eventloop.poll().await {
                                Ok(rumqttc::Event::Incoming(rumqttc::Packet::Publish(p))) => {
                                    let payload = String::from_utf8_lossy(&p.payload);
                                    info!("🛡️ MQTT CRITICAL INTERCEPTION: {}", payload);
                                }
                                Ok(_) => {}
                                Err(e) => {
                                    error!("⚠️ MQTT error: {}. Reconnecting...", e);
                                    tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                                }
                            }
                        }
                    });
                    true
                }
            }
        }
    };

    // Redis PubSub listener (si disponible)
    if redis_ok {
        if let Ok(client) = redis_result {
            if let Ok(con) = client.get_async_connection().await {
                let mut pubsub = con.into_pubsub();
                
                // On s'abonne à tous les canaux eva.*
                if let Err(e) = pubsub.psubscribe("eva.*").await {
                    error!("❌ Échec psubscribe Redis: {}", e);
                }

                info!("🛡️ Kernel Monitoring: Interception Redis (Pattern eva.*) + Watchdog actifs");

                let mut last_heartbeat = std::time::Instant::now();
                let tx_redis = tx.clone();

                // On lance l'écouteur Redis dans sa propre tâche pour isoler les types
                tokio::spawn(async move {
                    let mut msg_stream = pubsub.on_message();
                    while let Some(msg) = msg_stream.next().await {
                        let channel_name = String::from(msg.get_channel_name());
                        if let Ok(payload_str) = msg.get_payload::<String>() {
                            let normalized = normalize_feed_message(&channel_name, &payload_str);
                            let _ = tx_redis.send(normalized);
                        }
                    }
                });

                // La boucle principale ne gère plus que le watchdog
                loop {
                    tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                    // Note: Le heartbeat est maintenant géré différemment ou simplement via logs
                    info!("🛡️ Kernel Watchdog active...");
                }
            }
        }
    }

    // Mode dégradé : boucle keep-alive si ni Redis ni MQTT
    if !redis_ok && !mqtt_ok {
        warn!("⚠️ Kernel en mode dégradé — ni Redis ni MQTT disponibles");
    }

    // Keep-alive minimal
    loop {
        tokio::time::sleep(std::time::Duration::from_secs(30)).await;
        info!("💓 Kernel heartbeat — Axum server actif sur :8080");
    }
}
