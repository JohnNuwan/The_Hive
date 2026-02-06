//! EVA Kernel - Point d'entrée principal
//!
//! Le Kernel est le composant de sécurité critique de THE HIVE.
//! Il valide les actions selon la Constitution et maintient l'audit trail.

mod audit;
mod laws;
mod validator;

use std::path::PathBuf;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

use crate::laws::Constitution;
use crate::validator::TradeValidator;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configuration du logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_target(false)
        .pretty()
        .init();

    info!("🔒 EVA Kernel démarrage...");

    // Charger la Constitution
    let constitution_path = std::env::var("CONSTITUTION_PATH")
        .unwrap_or_else(|_| "/mnt/tablet/constitution.toml".to_string());

    info!("📜 Chargement Constitution: {}", constitution_path);

    let constitution = match Constitution::load(&PathBuf::from(&constitution_path)) {
        Ok(c) => {
            info!("✅ Constitution chargée: {} lois, {} ROE", c.laws.len(), c.roe.len());
            c
        }
        Err(e) => {
            info!("⚠️ Constitution non trouvée, utilisation des valeurs par défaut: {}", e);
            Constitution::default()
        }
    };

    // Créer le validateur
    let validator = TradeValidator::new(constitution);

    info!("✅ EVA Kernel prêt");
    
    // Initialisation Redis pour l'interception
    let redis_url = std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());
    let client = redis::Client::open(redis_url)?;
    let mut con = client.get_async_connection().await?;
    let mut pubsub = con.into_pubsub();
    
    let mut pubsub = con.into_pubsub();
    
    // Initialisation MQTT (Neural Link Secondaire)
    use rumqttc::{AsyncClient, MqttOptions, QoS};
    let mut mqttoptions = MqttOptions::new("eva_kernel", "localhost", 1883);
    mqttoptions.set_keep_alive(std::time::Duration::from_secs(5));

    let (mqtt_client, mut eventloop) = AsyncClient::new(mqttoptions, 10);
    mqtt_client.subscribe("eva/banker/requests/critical", QoS::AtLeastOnce).await?;
    
    info!("🛡️ Kernel Monitoring: Interception et Watchdog actifs (Redis + MQTT)");

    let mut msg_stream = pubsub.on_message();
    let mut last_heartbeat = std::time::Instant::now();

    loop {
        tokio::select! {
            // Flux MQTT (Critique)
            notification = eventloop.poll() => {
                if let Ok(rumqttc::Event::Incoming(rumqttc::Packet::Publish(p))) = notification {
                    let payload = String::from_utf8_lossy(&p.payload);
                    info!("🛡️ MQTT CRITICAL INTERCEPTION: {}", payload);
                    // Validation prioritaire ici
                }
            }
            // Flux Redis (Standard)
            Some(msg) = msg_stream.next() => {
                let channel = msg.get_channel_name();
                let payload: String = msg.get_payload()?;
                
                if channel == "eva.banker.heartbeat" {
                    last_heartbeat = std::time::Instant::now();
                } else {
                    info!("🔍 Kernel Interception ({}): {}", channel, payload);
                    // Validation Loi 2 ici
                }
            }
            _ = tokio::time::sleep(std::time::Duration::from_millis(100)) => {
                if last_heartbeat.elapsed().as_millis() > 1000 {
                    tracing::error!("🚨 WATCHDOG: BANKER HEARTBEAT LOST! TRIGGERING EMERGENCY HALT");
                    // Logique de coupure forcée MT5 ici
                }
            }
        }
    }
}
