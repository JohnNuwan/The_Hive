import datetime
import json
import os
import uuid
from dataclasses import asdict, dataclass
from decimal import Decimal

import requests

from .hmac_helper import generate_hmac_signature


def json_default(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


@dataclass(kw_only=True)
class SourceStrategy:
    name: str = "eva"
    broker_name: str
    account_size: int
    strategy_name: str = "orion_v1"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(kw_only=True)
class OrderPayload:
    source_ticket_id: str
    comment: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(kw_only=True)
class OpenPayload(OrderPayload):
    symbol: str
    volume: float
    type: str
    magic_number: str | None = None


@dataclass(kw_only=True)
class ClosePayload(OrderPayload):
    full_close: bool
    reason: str
    master_profit: float | None = None


class BridgeHeader:
    def __init__(self, payload: str):
        self.api_key = os.getenv("BRIDGE_API_KEY")
        if not self.api_key:
            raise ValueError("BRIDGE_API_KEY must be set.")

        self.timestamp, self.signature = generate_hmac_signature(payload)
        self.content_type = "application/json"

    def to_dict(self):
        return {
            "Content-Type": self.content_type,
            "X-API-Key": self.api_key,
            "X-Signature": self.signature,
            "X-Timestamp": self.timestamp,
        }


class BridgeConnector:
    command: dict[type[OrderPayload], str] = {OpenPayload: "open", ClosePayload: "close"}

    def __init__(self):
        self._url = os.getenv("BRIDGE_URL")

    def _generate_command_id(self, command_type: str):
        uuid_cmd = str(uuid.uuid1())
        return f"{command_type}-{uuid_cmd}"

    def _generate_iso_datetime(self) -> str:
        created_at = datetime.datetime.now(datetime.timezone.utc)  # ISO 8601
        return created_at.isoformat()

    def _generate_full_payload(
        self,
        command_id: str,
        command_type: str,
        created_at: str,
        source: SourceStrategy,
        order_payload: OpenPayload | ClosePayload,
    ):
        return {
            "command_id": command_id,
            "command_type": command_type,
            "createdAt": created_at,
            "source": source.to_dict(),
            "payload": order_payload.to_dict(),
        }

    def send_order(
        self, source_strategy: SourceStrategy, payload: OpenPayload | ClosePayload
    ) -> requests.Response | None:
        created_at = self._generate_iso_datetime()
        command_type = self.command[type(payload)]
        command_id = self._generate_command_id(command_type)

        full_payload = self._generate_full_payload(
            command_id, command_type, created_at, source_strategy, payload
        )
        json_payload = json.dumps(full_payload, default=json_default)

        bridge_url = os.getenv("BRIDGE_URL")
        if not bridge_url:
            raise ValueError("BRIDGE_URL Must be set")

        headers = BridgeHeader(json_payload).to_dict()

        bridge_route = "api/v1/events"

        try:
            response = requests.post(
                url=f"{bridge_url}/{bridge_route}", headers=headers, data=json_payload
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as err:
            print(f"BridgeError: {err}")
        else:
            return response
