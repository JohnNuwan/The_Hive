use bollard::container::StopContainerOptions;
use bollard::Docker;
use redis::AsyncCommands;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::time::sleep;
use tracing::{error, info, warn};
use libp2p::{
    gossipsub, mdns, noise, swarm::{NetworkBehaviour, SwarmEvent}, tcp, yamux, Multiaddr, PeerId,
};
use std::error::Error;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use hmac::{Hmac, Mac};
use sha2::Sha256;
use hex;
use serde_json::value::RawValue;

type HmacSha256 = Hmac<Sha256>;

#[derive(Debug, Serialize, Deserialize)]
struct SwarmMessage {
    source: String,
    target: String,
    payload: Box<RawValue>,
    auth_hash: String,
    ts: i64,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
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

#[derive(NetworkBehaviour)]
struct MyBehaviour {
    gossipsub: gossipsub::Behaviour,
    mdns: mdns::tokio::Behaviour,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    info!("⚡ SENTINEL CORE (Rust) Starting with Polyglot Hardening (libp2p)...");

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

    info!("✅ Constitution loaded. Loi 2: {}% DD limit.", max_drawdown);

    // 2. Setup libp2p
    let mut swarm = libp2p::SwarmBuilder::with_new_identity()
        .with_tokio()
        .with_tcp(
            tcp::Config::default(),
            noise::Config::new,
            yamux::Config::default,
        )?
        .with_behaviour(|key| {
            let message_id_fn = |message: &gossipsub::Message| {
                let mut s = DefaultHasher::new();
                message.data.hash(&mut s);
                gossipsub::MessageId::from(s.finish().to_string())
            };

            let gossipsub_config = gossipsub::ConfigBuilder::default()
                .heartbeat_interval(Duration::from_secs(10))
                .validation_mode(gossipsub::ValidationMode::Strict)
                .message_id_fn(message_id_fn)
                .build()
                .map_err(|msg| std::io::Error::new(std::io::ErrorKind::Other, msg))?;

            let gossipsub = gossipsub::Behaviour::new(
                gossipsub::MessageAuthenticity::Signed(key.clone()),
                gossipsub_config,
            )?;

            let mdns = mdns::tokio::Behaviour::new(mdns::Config::default(), key.public().to_peer_id())?;
            Ok(MyBehaviour { gossipsub, mdns })
        })?
        .with_swarm_config(|c| c.with_idle_connection_timeout(Duration::from_secs(60)))
        .build();

    let topic = gossipsub::IdentTopic::new("eva.swarm.heartbeats");
    swarm.behaviour_mut().gossipsub.subscribe(&topic)?;

    swarm.listen_on("/ip4/0.0.0.0/tcp/0".parse()?)?;

    // 3. Connection to Docker
    let docker = Docker::connect_with_unix_defaults()?;

    // 4. Connect to Redis (Fallback)
    let redis_host = std::env::var("REDIS_HOST").unwrap_or_else(|_| "localhost".to_string());
    let redis_password = std::env::var("REDIS_PASSWORD").unwrap_or_default();
    let redis_url = if redis_password.is_empty() {
        format!("redis://{}:6379", redis_host)
    } else {
        format!("redis://:{}@{}:6379", redis_password, redis_host)
    };

    let client = redis::Client::open(redis_url)?;
    // Use multiplexed async connection to fix deprecation and get_db error
    let redis_conn = client.get_multiplexed_async_connection().await.ok();

    let mut redis_pubsub = if redis_conn.is_some() {
        if let Ok(mut ps) = client.get_async_pubsub().await {
            let _ = ps.subscribe("eva.banker.heartbeat").await;
            Some(ps)
        } else { None }
    } else { None };
    
    // We need a separate connection for publishing if we want to publish from the loop
    let mut redis_publish_conn = client.get_multiplexed_async_connection().await.ok();

    info!("✅ P2P Node Started. PeerID: {}", swarm.local_peer_id());

    let secret_key = get_secret_key();

    // 5. Monitoring Loop
    loop {
        tokio::select! {
             event = swarm.select_next_some() => match event {
                SwarmEvent::Behaviour(MyBehaviourEvent::Mdns(mdns::Event::Discovered(list))) => {
                    for (peer_id, _multiaddr) in list {
                        info!("🌐 P2P: Discovered peer {}", peer_id);
                        swarm.behaviour_mut().gossipsub.add_explicit_peer(&peer_id);
                    }
                },
                SwarmEvent::Behaviour(MyBehaviourEvent::Gossipsub(gossipsub::Event::Message {
                    propagation_source: _,
                    message_id: _,
                    message,
                })) => {
                    if let Ok(swarm_msg) = serde_json::from_slice::<SwarmMessage>(&message.data) {
                        if verify_signature(&swarm_msg, &secret_key) {
                            if let Ok(hb) = serde_json::from_str::<BankerHeartbeat>(swarm_msg.payload.get()) {
                                let drawdown = (1.0 - (hb.equity / hb.balance)) * 100.0;
                                if drawdown >= max_drawdown {
                                    warn!("🚨 P2P ALERT: Critical Drawdown {:.2}% from {}", drawdown, hb.expert);
                                    // Logic to kill containers (shared with Redis logic)
                                    for container in &monitored_containers {
                                        let _ = docker.stop_container(container, Some(StopContainerOptions { t: 0 })).await;
                                    }
                                }
                            }
                        } else {
                            warn!("🚨 P2P WARNING: Invalid signature from peer {}", message.source.map(|p| p.to_string()).unwrap_or_default());
                        }
                    } else {
                        warn!("🚨 P2P WARNING: Received malformed or unsigned message");
                    }
                },
                SwarmEvent::NewListenAddr { address, .. } => {
                    info!("📍 Sentinel listening on {}", address);
                },
                _ => {}
            },
            msg = async {
                if let Some(ref mut ps) = redis_pubsub {
                    ps.on_message().next().await
                } else {
                    futures_util::future::pending().await
                }
            } => {
                if let Some(msg) = msg {
                    let payload: String = msg.get_payload().unwrap_or_default();
                    if let Ok(hb) = serde_json::from_str::<BankerHeartbeat>(&payload) {
                        let drawdown = (1.0 - (hb.equity / hb.balance)) * 100.0;
                        if drawdown >= max_drawdown {
                            warn!("🚨 REDIS ALERT: Critical Drawdown {:.2}%", drawdown);
                            for container in &monitored_containers {
                                let _ = docker.stop_container(container, Some(StopContainerOptions { t: 0 })).await;
                            }
                            // Broadcast to P2P as well!
                            let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as i64;
                            let target = "Broadcast";
                            let source = "sentinel_bridge";
                            let signature = generate_hmac(source, target, &payload, ts, &secret_key);

                            if let Ok(raw_payload) = serde_json::value::RawValue::from_string(payload.clone()) {
                                let swarm_msg = SwarmMessage {
                                    source: source.to_string(),
                                    target: target.to_string(),
                                    payload: raw_payload,
                                    auth_hash: signature,
                                    ts,
                                };
                                if let Ok(data) = serde_json::to_vec(&swarm_msg) {
                                    let _ = swarm.behaviour_mut().gossipsub.publish(topic.clone(), data);
                                }
                            }
                            
                            // Publish to Kernel via Redis if connected
                             if let Some(ref mut conn) = redis_publish_conn {
                                let _ : () = conn.publish("eva.kernel.emergency", 
                                    format!("{{\"action\":\"KILL_SWITCH_TRIGGERED\", \"drawdown\":{}}}", drawdown)).await.unwrap_or(());
                            }
                        }
                    }
                }
            },
            _ = sleep(Duration::from_secs(30)) => {
                info!("💓 Sentinel Check: P2P Peers: {}", swarm.connected_peers().count());
                // Publish local heartbeat to P2P
                let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64();
                let hb = BankerHeartbeat {
                    status: "online".to_string(),
                    ts: now,
                    expert: "sentinel_rust".to_string(),
                    equity: 0.0,
                    balance: 0.0,
                    currency: "N/A".to_string(),
                };
                if let Ok(hb_json) = serde_json::to_string(&hb) {
                    let ts = now as i64;
                    let target = "Broadcast";
                    let source = "sentinel_rust";
                    let signature = generate_hmac(source, target, &hb_json, ts, &secret_key);

                    if let Ok(raw_payload) = serde_json::value::RawValue::from_string(hb_json) {
                        let swarm_msg = SwarmMessage {
                            source: source.to_string(),
                            target: target.to_string(),
                            payload: raw_payload,
                            auth_hash: signature,
                            ts,
                        };
                        if let Ok(data) = serde_json::to_vec(&swarm_msg) {
                            let _ = swarm.behaviour_mut().gossipsub.publish(topic.clone(), data);
                        }
                    }
                }
            }
        }
    }
}

fn get_secret_key() -> String {
    std::env::var("NERVOUS_SECRET_KEY").expect("🚨 CRITICAL: NERVOUS_SECRET_KEY must be set!")
}

fn generate_hmac(source: &str, target: &str, payload: &str, ts: i64, secret: &str) -> String {
    let auth_input = format!("{}|{}|{}|{}", source, target, payload, ts);
    let mut mac = HmacSha256::new_from_slice(secret.as_bytes())
        .expect("HMAC can take key of any size");
    mac.update(auth_input.as_bytes());
    hex::encode(mac.finalize().into_bytes())
}

fn verify_signature(msg: &SwarmMessage, secret: &str) -> bool {
    let computed = generate_hmac(&msg.source, &msg.target, msg.payload.get(), msg.ts, secret);
    if computed != msg.auth_hash {
        return false;
    }

    // Check timestamp (prevent replay) - allow 60s window
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as i64;
    if (now - msg.ts).abs() > 60 {
        warn!("🚨 P2P WARNING: Message timestamp stale/future: {} (now: {})", msg.ts, now);
        return false;
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hmac_signing() {
        let secret = "test-secret";
        let source = "test-source";
        let target = "test-target";
        let payload = "{\"foo\":\"bar\"}";
        // Use current time
        let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as i64;

        let signature = generate_hmac(source, target, payload, ts, secret);

        let msg_json = format!(r#"{{
            "source": "{}",
            "target": "{}",
            "payload": {},
            "auth_hash": "{}",
            "ts": {}
        }}"#, source, target, payload, signature, ts);

        let msg: SwarmMessage = serde_json::from_str(&msg_json).expect("Failed to parse SwarmMessage");
        assert!(verify_signature(&msg, secret));

        // Test invalid signature
        let bad_msg_json = format!(r#"{{
            "source": "{}",
            "target": "{}",
            "payload": {},
            "auth_hash": "badhash",
            "ts": {}
        }}"#, source, target, payload, ts);
        let bad_msg: SwarmMessage = serde_json::from_str(&bad_msg_json).expect("Failed to parse bad SwarmMessage");
        assert!(!verify_signature(&bad_msg, secret));

        // Test stale timestamp
        let old_ts = ts - 100;
        let old_sig = generate_hmac(source, target, payload, old_ts, secret);
        let old_msg_json = format!(r#"{{
            "source": "{}",
            "target": "{}",
            "payload": {},
            "auth_hash": "{}",
            "ts": {}
        }}"#, source, target, payload, old_sig, old_ts);
        let old_msg: SwarmMessage = serde_json::from_str(&old_msg_json).expect("Failed to parse old SwarmMessage");
        assert!(!verify_signature(&old_msg, secret));
    }
}
