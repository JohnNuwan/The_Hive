use bollard::container::StopContainerOptions;
use bollard::Docker;
use redis::AsyncCommands;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::time::sleep;
use tracing::{error, info, warn};

#[derive(Debug, Deserialize, Serialize)]
struct BankerHeartbeat {
    status: String,
    ts: f64,
    expert: String,
    equity: f64,
    balance: f64,
    currency: String,
}

#[derive(Debug, Deserialize)]
struct Loi2 {
    max_daily_drawdown_percent: f64,
    monitored_containers: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct Constitution {
    loi_2_risque: Loi2,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    info!("⚡ SENTINEL CORE (Rust) Starting...");

    // 1. Load Constitution (Loi 1.3 - Hardware Key check)
    let constitution_path = "/mnt/tablet/Lois.toml";
    let constitution_content = std::fs::read_to_string(constitution_path)
        .map_err(|e| {
            error!("🚨 CONSTITUTION ERROR: 'The Tablet' (USB) is not mounted at {}! System panic initiated.", constitution_path);
            e
        })?;
    
    let constitution: Constitution = toml::from_str(&constitution_content)?;
    let max_drawdown = constitution.loi_2_risque.max_daily_drawdown_percent;
    let monitored_containers = constitution.loi_2_risque.monitored_containers;

    info!("✅ Constitution loaded from 'The Tablet'. Loi 2: {}% DD limit.", max_drawdown);

    // 2. Connection to Docker
    let docker = Docker::connect_with_unix_defaults()?;

    // 3. Connect to Redis
    let redis_host = std::env::var("REDIS_HOST").unwrap_or_else(|_| "localhost".to_string());
    let redis_password = std::env::var("REDIS_PASSWORD").unwrap_or_default();
    let redis_url = if redis_password.is_empty() {
        format!("redis://{}:6379", redis_host)
    } else {
        format!("redis://:{}@{}:6379", redis_password, redis_host)
    };

    let client = redis::Client::open(redis_url)?;
    let mut connection = client.get_async_connection().await?;
    let mut pubsub = client.get_async_pubsub().await?;

    info!("✅ Connected to Redis, subscribing to heartbeats...");
    pubsub.subscribe("eva.banker.heartbeat").await?;

    let mut pubsub_stream = pubsub.on_message();

    // 4. Monitoring Loop
    loop {
        tokio::select! {
            msg = pubsub_stream.next() => {
                if let Some(msg) = msg {
                    let payload: String = msg.get_payload()?;
                    match serde_json::from_str::<BankerHeartbeat>(&payload) {
                        Ok(hb) => {
                            let drawdown = (1.0 - (hb.equity / hb.balance)) * 100.0;
                            
                            if drawdown >= max_drawdown {
                                warn!("🚨 CRITICAL DRAWDOWN DETECTED: {:.2}%! TRIGGERING KILL-SWITCH...", drawdown);
                                
                                for container in &monitored_containers {
                                    info!("🛑 Killing container: {}", container);
                                    let options = Some(StopContainerOptions { t: 0 });
                                    if let Err(e) = docker.stop_container(container, options).await {
                                        error!("❌ Failed to kill container {}: {}", container, e);
                                    }
                                }
                                
                                // Send emergency broadcast
                                let _ : () = connection.publish("eva.kernel.emergency", 
                                    format!("{{\"action\":\"KILL_SWITCH_TRIGGERED\", \"drawdown\":{}}}", drawdown)).await?;
                            }
                        }
                        Err(e) => error!("Failed to parse heartbeat: {}", e),
                    }
                }
            }
            _ = sleep(Duration::from_secs(30)) => {
                info!("💓 Sentinel Check: All systems nominal");
            }
        }
    }
}
