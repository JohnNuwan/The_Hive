package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/go-redis/redis/v8"
)

// THE NERVOUS SYSTEM (Golang)
// Router Haute Fréquence pour la Ruche.
// Remplace les API REST lentes par du Pub/Sub Redis ultra-rapide.

var ctx = context.Background()

func main() {
	redisHost := os.Getenv("REDIS_HOST")
	if redisHost == "" {
		redisHost = "localhost"
	}
	redisPort := os.Getenv("REDIS_PORT")
	if redisPort == "" {
		redisPort = "6379"
	}

	rdb := redis.NewClient(&redis.Options{
		Addr: fmt.Sprintf("%s:%s", redisHost, redisPort),
	})

	fmt.Println("⚡ NERVOUS SYSTEM ONLINE (Go Runtime)")
	fmt.Println("Listening for critical signals...")

	// Goroutine 1: Écoute des signaux de Danger (Sentinel)
	go func() {
		pubsub := rdb.Subscribe(ctx, "danger_signal")
		defer pubsub.Close()

		for msg := range pubsub.Channel() {
			fmt.Printf("🚨 DANGER RECEIVED: %s. Routing to KILL_SWITCH in <1ms.\n", msg.Payload)
			// Priorité Absolue : On forward direct au Kernel
			rdb.Publish(ctx, "kernel_action", "ACTIVATE_KILL_SWITCH")
		}
	}()

	// Goroutine 2: Écoute des Opportunités de Trading (Core -> Banker)
	go func() {
		pubsub := rdb.Subscribe(ctx, "trade_opportunity")
		defer pubsub.Close()

		for msg := range pubsub.Channel() {
			// Routing intelligent : Core -> (Risk Check) -> Banker
			// Ici on simule un check rapide
			fmt.Printf("💰 TRADE SIGNAL: %s. Routing to Banker.\n", msg.Payload)
			rdb.Publish(ctx, "banker_orders", msg.Payload)
		}
	}()

	// Main Loop (Heartbeat)
	for {
		time.Sleep(5 * time.Second)
		fmt.Println("💓 Nervous System Pulse: Active & Listening.")
	}
}
