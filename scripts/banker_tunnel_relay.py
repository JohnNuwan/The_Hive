#!/usr/bin/env python3
"""Expose un port TCP public vers un tunnel SSH local au serveur.

Ce relay est lance sur le serveur Proxmox. Il ecoute sur une interface
accessible aux conteneurs Docker, puis transfere les connexions vers le port
localhost ouvert par le reverse tunnel SSH du Banker Windows.
"""

from __future__ import annotations

import argparse
import select
import socket
import socketserver
from dataclasses import dataclass


BUFFER_SIZE = 64 * 1024


@dataclass(frozen=True)
class RelayConfig:
    """Decrit la destination du relay TCP.

    Args:
        target_host: Hote cible joignable depuis le serveur Linux.
        target_port: Port cible expose par le tunnel SSH.
    """

    target_host: str
    target_port: int


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """Serveur TCP multi-clients pour le relay du Banker."""

    allow_reuse_address = True
    daemon_threads = True


class RelayHandler(socketserver.BaseRequestHandler):
    """Transfere une connexion cliente vers la cible du tunnel.

    La classe lit les donnees dans les deux sens jusqu'a fermeture de l'une
    des sockets. Le relay reste volontairement simple pour eviter toute
    logique applicative dans cette couche reseau.
    """

    server: ThreadedTCPServer

    def handle(self) -> None:
        """Pompe les octets entre le client entrant et la cible."""

        config: RelayConfig = self.server.relay_config  # type: ignore[attr-defined]
        with socket.create_connection((config.target_host, config.target_port), timeout=5) as upstream:
            self.request.setblocking(False)
            upstream.setblocking(False)
            sockets = [self.request, upstream]

            while True:
                readable, _, exceptional = select.select(sockets, [], sockets, 30.0)
                if exceptional:
                    return
                if not readable:
                    continue

                for current in readable:
                    try:
                        payload = current.recv(BUFFER_SIZE)
                    except OSError:
                        return

                    if not payload:
                        return

                    destination = upstream if current is self.request else self.request
                    try:
                        destination.sendall(payload)
                    except OSError:
                        return


def parse_args() -> argparse.Namespace:
    """Construit les arguments CLI du relay.

    Returns:
        argparse.Namespace: Configuration demandee au lancement.
    """

    parser = argparse.ArgumentParser(description="Relay TCP pour le tunnel Banker.")
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, default=18101)
    parser.add_argument("--target-host", default="127.0.0.1")
    parser.add_argument("--target-port", type=int, default=18100)
    return parser.parse_args()


def main() -> None:
    """Demarre le relay TCP bloquant."""

    args = parse_args()
    relay_config = RelayConfig(target_host=args.target_host, target_port=args.target_port)

    with ThreadedTCPServer((args.listen_host, args.listen_port), RelayHandler) as server:
        server.relay_config = relay_config  # type: ignore[attr-defined]
        server.serve_forever()


if __name__ == "__main__":
    main()
