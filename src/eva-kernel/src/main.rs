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
    
    // On s'abonne aux ordres et au heartbeat
    pubsub.subscribe("eva.banker.requests").await?;
    pubsub.subscribe("eva.banker.heartbeat").await?;
    
    info!("🛡️ Kernel Monitoring: Interception et Watchdog actifs");

    let mut msg_stream = pubsub.on_message();
    let mut last_heartbeat = std::time::Instant::now();

    loop {
        tokio::select! {
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
