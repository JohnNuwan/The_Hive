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
    info!("📊 Limites Trading:");
    info!("   - Risque max par trade: {}%", validator.get_max_risk_per_trade());
    info!("   - Drawdown journalier max: {}%", validator.get_max_daily_drawdown());
    info!("   - Positions max: {}", validator.get_max_positions());

    // TODO: Démarrer le serveur gRPC ou le listener Redis
    // Pour l'instant, on attend indéfiniment
    info!("🔄 En attente de requêtes...");

    tokio::signal::ctrl_c().await?;
    info!("🛑 Arrêt du Kernel");

    Ok(())
}
