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
use tracing::{error, info, warn, Level};
use tracing_subscriber::FmtSubscriber;

use crate::kill_switch::KillSwitch;
use crate::laws::Constitution;
use crate::server::start_kernel_server;
use crate::validator::TradeValidator;

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
    // CRÉER LES COMPOSANTS CRITIQUES
    // ═══════════════════════════════════════════════════════════════════
    let validator = TradeValidator::new(constitution.clone());
    let kill_switch = KillSwitch::new(constitution.trading.max_daily_drawdown_percent);

    info!("✅ EVA Kernel prêt — Lancement des systèmes parallèles");

    // ═══════════════════════════════════════════════════════════════════
    // LANCER LE SERVEUR AXUM EN PARALLÈLE
    // ═══════════════════════════════════════════════════════════════════
    tokio::spawn(start_kernel_server(
        validator,
        kill_switch,
        constitution,
    ));

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
                let _ = pubsub.subscribe("eva.banker.heartbeat").await;
                let _ = pubsub.subscribe("eva.banker.requests.critical").await;

                info!("🛡️ Kernel Monitoring: Interception Redis + Watchdog actifs");

                let mut msg_stream = pubsub.on_message();
                let mut last_heartbeat = std::time::Instant::now();

                loop {
                    tokio::select! {
                        Some(msg) = msg_stream.next() => {
                            let channel = msg.get_channel_name();
                            if let Ok(payload) = msg.get_payload::<String>() {
                                if channel == "eva.banker.heartbeat" {
                                    last_heartbeat = std::time::Instant::now();
                                } else {
                                    info!("🔍 Kernel Interception ({}): {}", channel, payload);
                                }
                            }
                        }
                        _ = tokio::time::sleep(std::time::Duration::from_millis(500)) => {
                            if last_heartbeat.elapsed().as_secs() > 10 {
                                error!("🚨 WATCHDOG: BANKER HEARTBEAT LOST >10s! Alert triggered.");
                                // En prod: déclencher kill-switch via channel Redis
                                last_heartbeat = std::time::Instant::now(); // Reset pour éviter spam
                            }
                        }
                    }
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
