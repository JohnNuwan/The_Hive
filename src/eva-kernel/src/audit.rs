//! Module Audit - Black Box et Audit Trail immuable
//!
//! Implémente l'audit trail blockchain-like selon Loi 3 (Transparence Absolue).

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::VecDeque;
use uuid::Uuid;

/// Enregistrement d'audit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditRecord {
    pub id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub agent: String,
    pub action: String,
    pub details: serde_json::Value,
    pub user_id: Option<String>,
    pub session_id: Option<Uuid>,
    pub previous_hash: String,
    pub record_hash: String,
}

impl AuditRecord {
    /// Crée un nouvel enregistrement d'audit
    pub fn new(
        agent: String,
        action: String,
        details: serde_json::Value,
        previous_hash: String,
    ) -> Self {
        let mut record = Self {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            agent,
            action,
            details,
            user_id: None,
            session_id: None,
            previous_hash,
            record_hash: String::new(),
        };
        record.record_hash = record.compute_hash();
        record
    }

    /// Calcule le hash SHA-256 de l'enregistrement
    pub fn compute_hash(&self) -> String {
        let data = serde_json::json!({
            "id": self.id.to_string(),
            "timestamp": self.timestamp.to_rfc3339(),
            "agent": self.agent,
            "action": self.action,
            "details": self.details,
            "previous_hash": self.previous_hash,
        });

        let content = serde_json::to_string(&data).unwrap_or_default();
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        let result = hasher.finalize();

        hex::encode(result)
    }

    /// Vérifie l'intégrité de l'enregistrement
    pub fn verify(&self) -> bool {
        self.record_hash == self.compute_hash()
    }
}

/// Trait pour encoder en hex (simple implémentation)
mod hex {
    pub fn encode(bytes: impl AsRef<[u8]>) -> String {
        bytes
            .as_ref()
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect()
    }
}

/// Audit Trail - Chaîne d'enregistrements
pub struct AuditTrail {
    records: VecDeque<AuditRecord>,
    max_records: usize,
    last_hash: String,
}

impl AuditTrail {
    /// Crée un nouvel Audit Trail
    pub fn new(max_records: usize) -> Self {
        Self {
            records: VecDeque::new(),
            max_records,
            last_hash: "genesis".to_string(),
        }
    }

    /// Ajoute un enregistrement à la chaîne
    pub fn record(
        &mut self,
        agent: &str,
        action: &str,
        details: serde_json::Value,
    ) -> AuditRecord {
        let record = AuditRecord::new(
            agent.to_string(),
            action.to_string(),
            details,
            self.last_hash.clone(),
        );

        self.last_hash = record.record_hash.clone();
        self.records.push_back(record.clone());

        // Garder seulement les N derniers enregistrements en mémoire
        while self.records.len() > self.max_records {
            self.records.pop_front();
        }

        record
    }

    /// Vérifie l'intégrité de la chaîne
    pub fn verify_chain(&self) -> bool {
        let mut expected_previous = "genesis".to_string();

        for record in &self.records {
            // Vérifier que le hash précédent correspond
            if record.previous_hash != expected_previous {
                return false;
            }

            // Vérifier l'intégrité de l'enregistrement
            if !record.verify() {
                return false;
            }

            expected_previous = record.record_hash.clone();
        }

        true
    }

    /// Retourne les derniers enregistrements
    pub fn get_recent(&self, count: usize) -> Vec<&AuditRecord> {
        self.records
            .iter()
            .rev()
            .take(count)
            .collect()
    }

    /// Retourne le nombre d'enregistrements
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Vérifie si l'audit trail est vide
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Retourne le dernier hash
    pub fn get_last_hash(&self) -> &str {
        &self.last_hash
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audit_record_hash() {
        let record = AuditRecord::new(
            "banker".to_string(),
            "TRADE_EXECUTED".to_string(),
            serde_json::json!({"symbol": "XAUUSD", "volume": 0.5}),
            "genesis".to_string(),
        );

        assert!(!record.record_hash.is_empty());
        assert!(record.verify());
    }

    #[test]
    fn test_audit_trail_chain() {
        let mut trail = AuditTrail::new(100);

        trail.record(
            "core",
            "INTENT_CLASSIFIED",
            serde_json::json!({"intent": "TRADING_ORDER"}),
        );

        trail.record(
            "banker",
            "TRADE_EXECUTED",
            serde_json::json!({"ticket": 12345}),
        );

        trail.record(
            "kernel",
            "RISK_VALIDATED",
            serde_json::json!({"allowed": true}),
        );

        assert_eq!(trail.len(), 3);
        assert!(trail.verify_chain());
    }

    #[test]
    fn test_audit_trail_integrity() {
        let mut trail = AuditTrail::new(10);

        trail.record("test", "ACTION", serde_json::json!({"data": 1}));
        trail.record("test", "ACTION", serde_json::json!({"data": 2}));

        // La chaîne doit être valide
        assert!(trail.verify_chain());
    }
}
