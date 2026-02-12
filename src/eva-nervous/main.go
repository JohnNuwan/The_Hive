package main

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"sync"
	"sync/atomic"
	"time"

	"github.com/go-redis/redis/v8"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// ═══════════════════════════════════════════════════════════════════════════════
// THE NERVOUS SYSTEM (Golang)
// Router Haute Fréquence pour la Ruche.
//
// Fonctions :
// 1. Routage prioritaire des signaux (danger > trade > info)
// 2. Watchdog : surveille le heartbeat de chaque agent
// 3. Health endpoint HTTP :9090/health pour monitoring Docker
// 4. Reconnect automatique Redis
// 5. Métriques temps réel (messages routés, latence)
// ═══════════════════════════════════════════════════════════════════════════════

var ctx = context.Background()

// Métriques globales
type Metrics struct {
	DangerSignals    atomic.Int64
	TradeSignals     atomic.Int64
	SwarmEvents      atomic.Int64
	HeartbeatsRecv   atomic.Int64
	MessagesRouted   atomic.Int64
	ErrorsTotal      atomic.Int64
	LastHeartbeatAt  sync.Map // agent_name -> time.Time
	UptimeStart      time.Time
}

var metrics = Metrics{
	UptimeStart: time.Now(),
}

// Prometheus Metrics
var (
	promMessagesRouted = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "nervous_messages_routed_total",
		Help: "The total number of routed messages",
	}, []string{"channel", "priority"})

	promRoutingLatency = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "nervous_routing_latency_seconds",
		Help:    "Latency of message routing in seconds",
		Buckets: prometheus.DefBuckets,
	}, []string{"channel"})

	promErrors = promauto.NewCounter(prometheus.CounterOpts{
		Name: "nervous_errors_total",
		Help: "The total number of errors encountered",
	})
)

// ═══════════════════════════════════════════════════════════════════════════════
// REDIS CONNECTION (avec retry + auth)
// ═══════════════════════════════════════════════════════════════════════════════

func createRedisClient() *redis.Client {
	host := getEnv("REDIS_HOST", "localhost")
	port := getEnv("REDIS_PORT", "6379")
	password := getEnv("REDIS_PASSWORD", "")

	rdb := redis.NewClient(&redis.Options{
		Addr:         fmt.Sprintf("%s:%s", host, port),
		Password:     password,
		DB:           0,
		DialTimeout:  5 * time.Second,
		ReadTimeout:  3 * time.Second,
		WriteTimeout: 3 * time.Second,
		PoolSize:     10,
	})

	return rdb
}

func waitForRedis(rdb *redis.Client) {
	for {
		_, err := rdb.Ping(ctx).Result()
		if err == nil {
			log.Println("✅ Redis connecté")
			return
		}
		log.Printf("⏳ En attente de Redis... (%v)", err)
		time.Sleep(2 * time.Second)
	}
}

// ═══════════════════════════════════════════════════════════════════════════════
// STANDARD SWARM PROTOCOL
// ═══════════════════════════════════════════════════════════════════════════════

type SwarmMessage struct {
	Source   string          `json:"source"`
	Target   string          `json:"target"`
	Payload  json.RawMessage `json:"payload"`
	AuthHash string          `json:"auth_hash"`
	Ts       int64           `json:"ts"`
}

// computeHMAC generates a SHA256 HMAC for the message contents
func computeHMAC(source, target string, payload []byte, ts int64, secret string) string {
	h := hmac.New(sha256.New, []byte(secret))
	h.Write([]byte(source))
	h.Write([]byte(":"))
	h.Write([]byte(target))
	h.Write([]byte(":"))
	h.Write(payload)
	h.Write([]byte(":"))
	h.Write([]byte(fmt.Sprintf("%d", ts)))
	return hex.EncodeToString(h.Sum(nil))
}

// Canal 1 : DANGER — Priorité Absolue (latence <1ms visée)
func listenDangerSignals(rdb *redis.Client) {
	secret := getEnv("SWARM_SECRET_KEY", "default-insecure-secret")
	for {
		pubsub := rdb.Subscribe(ctx, "danger_signal", "eva.sentinel.alert", "eva.kernel.emergency")
		ch := pubsub.Channel()

		for msg := range ch {
			start := time.Now()
			metrics.DangerSignals.Add(1)
			metrics.MessagesRouted.Add(1)

			log.Printf("🚨 DANGER [%s]: %s — Routing to KILL_SWITCH", msg.Channel, msg.Payload)

			ts := time.Now().Unix()
			payload := json.RawMessage(msg.Payload)
			target := "Kernel"

			// Standardisation du message
			swarmMsg := SwarmMessage{
				Source:   msg.Channel,
				Target:   target,
				Payload:  payload,
				AuthHash: computeHMAC(msg.Channel, target, payload, ts, secret),
				Ts:       ts,
			}
			jsonMsg, _ := json.Marshal(swarmMsg)

			// Priorité Absolue : Forward direct au Kernel
			rdb.Publish(ctx, "kernel_action", jsonMsg)

			elapsed := time.Since(start)
			promMessagesRouted.WithLabelValues(msg.Channel, "P0").Inc()
			promRoutingLatency.WithLabelValues(msg.Channel).Observe(elapsed.Seconds())
			log.Printf("   ⚡ Routed in %v", elapsed)
		}

		log.Println("⚠️ Danger channel fermé, reconnecting...")
		pubsub.Close()
		time.Sleep(1 * time.Second)
	}
}

// Canal 2 : TRADING — Haute priorité (Core → Risk Check → Banker)
func listenTradeSignals(rdb *redis.Client) {
	secret := getEnv("SWARM_SECRET_KEY", "default-insecure-secret")
	for {
		pubsub := rdb.Subscribe(ctx, "trade_opportunity", "eva.core.trade_request")
		ch := pubsub.Channel()

		for msg := range ch {
			metrics.TradeSignals.Add(1)
			metrics.MessagesRouted.Add(1)

			log.Printf("💰 TRADE [%s]: %s — Routing to Banker via Kernel validation", msg.Channel, msg.Payload)

			ts := time.Now().Unix()
			payload := json.RawMessage(msg.Payload)
			target := "Banker/Kernel"

			// Standardisation
			swarmMsg := SwarmMessage{
				Source:   msg.Channel,
				Target:   target,
				Payload:  payload,
				AuthHash: computeHMAC(msg.Channel, target, payload, ts, secret),
				Ts:       ts,
			}
			jsonMsg, _ := json.Marshal(swarmMsg)

			// Étape 1 : Envoyer au Kernel pour validation Loi 2
			rdb.Publish(ctx, "eva.banker.requests.critical", jsonMsg)

			// Étape 2 : Forward au Banker (le Kernel intercepte en parallèle)
			rdb.Publish(ctx, "banker_orders", jsonMsg)

			promMessagesRouted.WithLabelValues(msg.Channel, "P1").Inc()
		}

		log.Println("⚠️ Trade channel fermé, reconnecting...")
		pubsub.Close()
		time.Sleep(1 * time.Second)
	}
}

// Canal 3 : SWARM — Coordination inter-agents
func listenSwarmEvents(rdb *redis.Client) {
	secret := getEnv("SWARM_SECRET_KEY", "default-insecure-secret")
	for {
		pubsub := rdb.Subscribe(ctx, "eva.swarm.events", "eva.swarm.broadcast", "eva.swarm.drone_scale")
		ch := pubsub.Channel()

		for msg := range ch {
			metrics.SwarmEvents.Add(1)
			metrics.MessagesRouted.Add(1)

			log.Printf("🐝 SWARM [%s]: %s", msg.Channel, msg.Payload)

			ts := time.Now().Unix()
			payload := json.RawMessage(msg.Payload)
			target := "Broadcast"

			// Standardisation
			swarmMsg := SwarmMessage{
				Source:   msg.Channel,
				Target:   target,
				Payload:  payload,
				AuthHash: computeHMAC(msg.Channel, target, payload, ts, secret),
				Ts:       ts,
			}
			jsonMsg, _ := json.Marshal(swarmMsg)

			// Broadcast à tous les agents intéressés
			rdb.Publish(ctx, "eva.core.swarm_event", jsonMsg)
			rdb.Publish(ctx, "eva.banker.swarm_event", jsonMsg)
		}

		log.Println("⚠️ Swarm channel fermé, reconnecting...")
		pubsub.Close()
		time.Sleep(1 * time.Second)
	}
}

// ═══════════════════════════════════════════════════════════════════════════════
// WATCHDOG — Surveille le heartbeat de chaque agent
// ═══════════════════════════════════════════════════════════════════════════════

func listenHeartbeats(rdb *redis.Client) {
	for {
		pubsub := rdb.Subscribe(ctx,
			"eva.banker.heartbeat",
			"eva.core.heartbeat",
			"eva.sentinel.heartbeat",
			"eva.lab.heartbeat",
		)
		ch := pubsub.Channel()

		for msg := range ch {
			metrics.HeartbeatsRecv.Add(1)
			metrics.LastHeartbeatAt.Store(msg.Channel, time.Now())
		}

		pubsub.Close()
		time.Sleep(1 * time.Second)
	}
}

func watchdogLoop(rdb *redis.Client) {
	criticalAgents := []string{
		"eva.banker.heartbeat",
	}
	warningAgents := []string{
		"eva.core.heartbeat",
		"eva.sentinel.heartbeat",
	}

	for {
		time.Sleep(5 * time.Second)
		now := time.Now()

		// Agents critiques : alerte si >10s sans heartbeat
		for _, agent := range criticalAgents {
			val, ok := metrics.LastHeartbeatAt.Load(agent)
			if !ok {
				continue // Pas encore reçu de heartbeat
			}
			lastBeat := val.(time.Time)
			if now.Sub(lastBeat) > 10*time.Second {
				metrics.ErrorsTotal.Add(1)
				promErrors.Inc()
				log.Printf("🚨 WATCHDOG: %s HEARTBEAT LOST (>10s)! Alerte Kernel.", agent)
				rdb.Publish(ctx, "kernel_action", fmt.Sprintf(`{"action":"WATCHDOG_ALERT","agent":"%s","last_seen":"%s"}`, agent, lastBeat.Format(time.RFC3339)))
			}
		}

		// Agents non-critiques : warning si >30s
		for _, agent := range warningAgents {
			val, ok := metrics.LastHeartbeatAt.Load(agent)
			if !ok {
				continue
			}
			lastBeat := val.(time.Time)
			if now.Sub(lastBeat) > 30*time.Second {
				log.Printf("⚠️ WATCHDOG: %s silencieux depuis %v", agent, now.Sub(lastBeat).Round(time.Second))
			}
		}
	}
}

// ═══════════════════════════════════════════════════════════════════════════════
// HEALTH HTTP ENDPOINT (:9090)
// ═══════════════════════════════════════════════════════════════════════════════

type HealthResponse struct {
	Status         string            `json:"status"`
	Uptime         string            `json:"uptime"`
	DangerSignals  int64             `json:"danger_signals"`
	TradeSignals   int64             `json:"trade_signals"`
	SwarmEvents    int64             `json:"swarm_events"`
	Heartbeats     int64             `json:"heartbeats_received"`
	TotalRouted    int64             `json:"total_messages_routed"`
	Errors         int64             `json:"errors_total"`
	AgentStatus    map[string]string `json:"agent_status"`
}

func healthHandler(w http.ResponseWriter, r *http.Request) {
	agentStatus := make(map[string]string)
	now := time.Now()

	metrics.LastHeartbeatAt.Range(func(key, value any) bool {
		agent := key.(string)
		lastBeat := value.(time.Time)
		delta := now.Sub(lastBeat)

		if delta < 10*time.Second {
			agentStatus[agent] = "ONLINE"
		} else if delta < 30*time.Second {
			agentStatus[agent] = "DEGRADED"
		} else {
			agentStatus[agent] = "OFFLINE"
		}
		return true
	})

	resp := HealthResponse{
		Status:        "operational",
		Uptime:        time.Since(metrics.UptimeStart).Round(time.Second).String(),
		DangerSignals: metrics.DangerSignals.Load(),
		TradeSignals:  metrics.TradeSignals.Load(),
		SwarmEvents:   metrics.SwarmEvents.Load(),
		Heartbeats:    metrics.HeartbeatsRecv.Load(),
		TotalRouted:   metrics.MessagesRouted.Load(),
		Errors:        metrics.ErrorsTotal.Load(),
		AgentStatus:   agentStatus,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func startHealthServer() {
	mux := http.NewServeMux()
	mux.HandleFunc("/health", healthHandler)
	mux.Handle("/metrics", promhttp.Handler())

	port := getEnv("HEALTH_PORT", "9090")
	log.Printf("📊 Health endpoint: http://0.0.0.0:%s/health", port)
	if err := http.ListenAndServe(":"+port, mux); err != nil {
		log.Fatalf("Health server failed: %v", err)
	}
}

// ═══════════════════════════════════════════════════════════════════════════════
// UTILS
// ═══════════════════════════════════════════════════════════════════════════════

func getEnv(key, fallback string) string {
	if val, ok := os.LookupEnv(key); ok {
		return val
	}
	return fallback
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════════

func main() {
	log.SetFlags(log.Ltime | log.Lmicroseconds)

	fmt.Println("╔══════════════════════════════════════════╗")
	fmt.Println("║  ⚡ THE NERVOUS SYSTEM (Go Runtime)      ║")
	fmt.Println("║  Router Haute Fréquence — THE HIVE       ║")
	fmt.Println("╚══════════════════════════════════════════╝")

	rdb := createRedisClient()
	waitForRedis(rdb)

	// Health endpoint HTTP (monitoring Docker + Dashboard)
	go startHealthServer()

	// Canaux prioritaires (du plus critique au moins critique)
	go listenDangerSignals(rdb)     // P0: Signaux de danger → Kill-Switch
	go listenTradeSignals(rdb)      // P1: Opportunités trading → Banker
	go listenSwarmEvents(rdb)       // P2: Coordination Swarm
	go listenHeartbeats(rdb)        // Heartbeat monitoring

	// Watchdog (détection d'agents morts)
	go watchdogLoop(rdb)

	// Heartbeat principal
	for {
		time.Sleep(10 * time.Second)
		log.Printf("💓 Nervous System alive — %d messages routés, %d erreurs",
			metrics.MessagesRouted.Load(), metrics.ErrorsTotal.Load())
	}
}
