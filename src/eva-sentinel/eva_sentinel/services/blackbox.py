"""
Black Box Recorder - THE SENTINEL
Enregistreur immuable de toutes les décisions et logs critiques du système.
"""

import os
import json
import logging
from datetime import datetime
from hashlib import sha256

class BlackBox:
    """
    Système de monitoring froid. 
    Chaque log est haché et chaîné pour garantir son immuabilité (Merkle Proof).
    """
    
    def __init__(self, log_dir: str = "/var/log/hive/blackbox"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.last_hash = "0" * 64 # Genesis hash

    def record_decision(self, agent_id: str, decision: str, data: dict):
        """Enregistre une décision critique et génère un hash proof"""
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "agent_id": agent_id,
            "decision": decision,
            "data": data,
            "previous_hash": self.last_hash
        }
        
        # Génération du hash d'immuabilité
        entry_json = json.dumps(entry, sort_keys=True)
        current_hash = sha256(entry_json.encode()).hexdigest()
        
        entry["current_hash"] = current_hash
        self.last_hash = current_hash
        
        # Sauvegarde
        filename = f"box_{datetime.now().strftime('%Y-%m-%d')}.jsonl"
        with open(os.path.join(self.log_dir, filename), "a") as f:
            f.write(json.dumps(entry) + "\n")
            
        return current_hash
