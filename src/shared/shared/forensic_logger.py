import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ForensicLogger:
    """
    Enregistreur de Vol Immuable (Black-Box).
    Chaîne chaque décision par un hash pour garantir l'intégrité de l'audit post-morten.
    """
    def __init__(self, log_dir: str = "Forensics"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / "hive_blackbox.jsonl"
        self.last_hash = "00000000000000000000000000000000"

    def log_critical_event(self, event_type: str, data: Dict[str, Any]):
        """
        Enregistre un événement et le lie au précédent via un hash.
        """
        payload = {
            "timestamp": datetime.now().isoformat(),
            "event": event_type,
            "data": data,
            "previous_hash": self.last_hash
        }
        
        # Calcul du nouveau hash
        payload_str = json.dumps(payload, sort_keys=True)
        current_hash = hashlib.sha256(payload_str.encode()).hexdigest()
        payload["hash"] = current_hash
        
        # Stockage immuable (append-only)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
            
        self.last_hash = current_hash
        logger.info(f"Forensic: Event {event_type} hashed and chained.")

    def verify_integrity(self) -> bool:
        """Vérifie que la chaîne de logs n'a pas été altérée."""
        if not self.log_file.exists(): return True
        
        expected_prev_hash = "00000000000000000000000000000000"
        
        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line)
                    if entry["previous_hash"] != expected_prev_hash:
                        return False
                    
                    # Re-calcul du hash pour vérification
                    temp_entry = entry.copy()
                    stored_hash = temp_entry.pop("hash")
                    payload_str = json.dumps(temp_entry, sort_keys=True)
                    computed_hash = hashlib.sha256(payload_str.encode()).hexdigest()
                    
                    if stored_hash != computed_hash:
                        return False
                    
                    expected_prev_hash = stored_hash
            return True
        except Exception:
            return False
