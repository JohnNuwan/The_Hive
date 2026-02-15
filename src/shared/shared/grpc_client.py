import logging
import time
import json
import grpc
from pathlib import Path
from typing import Any, Optional

# On tente d'importer les protos générés.
# S'ils n'existent pas encore, on fournit une interface pour les charger dynamiquement
# ou on attend le prochain build.
try:
    from shared.proto import swarm_pb2, swarm_pb2_grpc
    HAS_PROTOS = True
except ImportError:
    HAS_PROTOS = False

logger = logging.getLogger(__name__)

class SwarmGRPCClient:
    """
    Client gRPC pour injecter des signaux dans le Nervous System (Go).
    Préfère le gRPC pour la latence, mais peut être utilisé en complément de Redis.
    """

    def __init__(self, host: str = "nervous", port: int = 9091):
        self.target = f"{host}:{port}"
        self.channel: Optional[grpc.Channel] = None
        self.stub: Optional[Any] = None
        self._connected = False

    def connect(self):
        """Initialise la connexion gRPC."""
        if not HAS_PROTOS:
            logger.warning("⚠️ Protos gRPC non trouvés. Le client gRPC est désactivé.")
            return False
            
        try:
            self.channel = grpc.insecure_channel(self.target)
            self.stub = swarm_pb2_grpc.SwarmRouterStub(self.channel)
            self._connected = True
            logger.info(f"🚀 Connecté au Nervous gRPC sur {self.target}")
            return True
        except Exception as e:
            logger.error(f"❌ Échec connexion gRPC: {e}")
            self._connected = False
            return False

    def send_signal(
        self,
        source: str,
        target: str,
        action: str,
        payload: dict,
        priority: int = 2,  # P2_NORMAL par défaut
        auth_hash: str = ""
    ) -> bool:
        """Envoie un signal vers le système nerveux via gRPC."""
        if not self._connected and not self.connect():
            return False

        try:
            # Construction du message proto
            message = swarm_pb2.SwarmMessage(
                source=source,
                target=target,
                action=action,
                payload=json.dumps(payload).encode('utf-8'),
                auth_hash=auth_hash,
                ts=int(time.time()),
                priority=priority
            )
            
            # Appel RPC (timeout court pour ne pas bloquer l'expert)
            response = self.stub.SendSignal(message, timeout=0.1)
            
            if not response.accepted:
                logger.warning(f"⚠️ Signal gRPC rejeté par Nervous: {response.message}")
            
            return response.accepted
            
        except grpc.RpcError as e:
            logger.error(f"❌ Erreur RPC lors de l'envoi du signal: {e.code()} - {e.details()}")
            self._connected = False # Forcer reconnexion au prochain appel
            return False
        except Exception as e:
            logger.error(f"❌ Erreur inattendue gRPC: {e}")
            return False

    def close(self):
        """Ferme la connexion."""
        if self.channel:
            self.channel.close()
            self._connected = False
