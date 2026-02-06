import socket
import asyncio
import json
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class P2PDiscovery:
    """
    Lien de Découverte Maillé (mDNS-style).
    Permet à la ruche de se propager sur plusieurs machines sans coordinateur.
    """
    def __init__(self, agent_name: str, port: int = 42421):
        self.agent_name = agent_name
        self.port = port
        self.known_nodes: Dict[str, str] = {}
        self.active = False

    async def start_broadcasting(self):
        """Diffuse la présence de l'agent sur le réseau local via UDP."""
        self.active = True
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        
        logger.info(f"P2P: Lancement du signal de découverte pour {self.agent_name}...")
        
        while self.active:
            message = json.dumps({
                "agent": self.agent_name,
                "status": "online",
                "timestamp": asyncio.get_event_loop().time()
            }).encode()
            
            try:
                sock.sendto(message, ('<broadcast>', self.port))
            except Exception as e:
                logger.debug(f"P2P Broadcast error: {e}")
                
            await asyncio.sleep(10) # Signal toutes les 10 secondes

    async def start_listening(self):
        """Écoute les signaux des autres agents sur le réseau."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(('', self.port))
        sock.setblocking(False)
        
        loop = asyncio.get_event_loop()
        logger.info("P2P: Scan de la ruche locale actif.")
        
        while self.active:
            try:
                data, addr = await loop.sock_recvfrom(sock, 1024)
                info = json.loads(data.decode())
                
                agent = info.get("agent")
                if agent and agent != self.agent_name:
                    if agent not in self.known_nodes:
                        logger.info(f"✨ P2P: Nouvel expert découvert sur le réseau : {agent} ({addr[0]})")
                    self.known_nodes[agent] = addr[0]
            except Exception:
                pass
            await asyncio.sleep(0.1)

    def get_swarm_map(self) -> Dict[str, str]:
        return self.known_nodes
