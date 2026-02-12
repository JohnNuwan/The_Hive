package main

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"testing"
)

func TestComputeHMAC(t *testing.T) {
	secret := "my-secret-key"
	source := "test-source"
	target := "test-target"
	payloadStr := `{"foo":"bar"}`
	payload := json.RawMessage(payloadStr)
	ts := int64(1678886400)

	// Expected HMAC
	h := hmac.New(sha256.New, []byte(secret))
	h.Write([]byte(source))
	h.Write([]byte(":"))
	h.Write([]byte(target))
	h.Write([]byte(":"))
	h.Write(payload)
	h.Write([]byte(":"))
	h.Write([]byte(fmt.Sprintf("%d", ts)))
	expectedHash := hex.EncodeToString(h.Sum(nil))

	hash := computeHMAC(source, target, payload, ts, secret)

	if hash == "" {
		t.Errorf("Expected non-empty hash, got empty string")
	}

	if hash != expectedHash {
		t.Errorf("Hash mismatch: got %s, expected %s", hash, expectedHash)
	}

	// Verify deterministic
	hash2 := computeHMAC(source, target, payload, ts, secret)
	if hash != hash2 {
		t.Errorf("Hash mismatch on second call: %s != %s", hash, hash2)
	}

	// Verify sensitivity to payload
	payload2 := json.RawMessage(`{"foo":"baz"}`)
	hash3 := computeHMAC(source, target, payload2, ts, secret)
	if hash == hash3 {
		t.Errorf("Hash collision for different payload")
	}

	// Verify sensitivity to timestamp
	ts2 := ts + 1
	hash4 := computeHMAC(source, target, payload, ts2, secret)
	if hash == hash4 {
		t.Errorf("Hash collision for different timestamp")
	}

	// Verify sensitivity to secret
	secret2 := "other-secret"
	hash5 := computeHMAC(source, target, payload, ts, secret2)
	if hash == hash5 {
		t.Errorf("Hash collision for different secret")
	}
}
