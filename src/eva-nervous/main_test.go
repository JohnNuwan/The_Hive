package main

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"testing"
)

func TestGenerateHMAC(t *testing.T) {
	secret := "my-secret-key"
	message := "source|target|payload|1234567890"

	// Expected HMAC calculation
	h := hmac.New(sha256.New, []byte(secret))
	h.Write([]byte(message))
	expected := hex.EncodeToString(h.Sum(nil))

	result := generateHMAC(message, secret)

	if result != expected {
		t.Errorf("generateHMAC() = %v, want %v", result, expected)
	}
}

func TestGenerateHMAC_EmptySecret(t *testing.T) {
	secret := ""
	message := "test-message"

	// Expected HMAC with empty secret
	h := hmac.New(sha256.New, []byte(secret))
	h.Write([]byte(message))
	expected := hex.EncodeToString(h.Sum(nil))

	result := generateHMAC(message, secret)

	if result != expected {
		t.Errorf("generateHMAC() with empty secret = %v, want %v", result, expected)
	}
}

func TestGenerateHMAC_DifferentMessages(t *testing.T) {
	secret := "secret"
	msg1 := "message1"
	msg2 := "message2"

	hash1 := generateHMAC(msg1, secret)
	hash2 := generateHMAC(msg2, secret)

	if hash1 == hash2 {
		t.Error("generateHMAC() should return different hashes for different messages")
	}
}

func TestGenerateHMAC_Consistency(t *testing.T) {
	// Known test vector (HMAC-SHA256)
	// key: "key"
	// data: "The quick brown fox jumps over the lazy dog"
	// expected: f7bc83f430538424b13298e6aa6fb143ef4d59a14946175997479dbc2d1a3cd8

	secret := "key"
	message := "The quick brown fox jumps over the lazy dog"
	expected := "f7bc83f430538424b13298e6aa6fb143ef4d59a14946175997479dbc2d1a3cd8"

	result := generateHMAC(message, secret)

	if result != expected {
		t.Errorf("generateHMAC() check against test vector failed. Got %v, want %v", result, expected)
	}
}

func TestGetSecretKey_Default(t *testing.T) {
	// Ensure no env var set
	t.Setenv("NERVOUS_SECRET_KEY", "")

	key := getSecretKey()
	expected := "insecure-dev-secret-change-me"

	if key != expected {
		t.Errorf("getSecretKey() default = %v, want %v", key, expected)
	}
}

func TestGetSecretKey_EnvSet(t *testing.T) {
	expected := "env-secret-key"
	t.Setenv("NERVOUS_SECRET_KEY", expected)

	key := getSecretKey()

	if key != expected {
		t.Errorf("getSecretKey() with env set = %v, want %v", key, expected)
	}
}
