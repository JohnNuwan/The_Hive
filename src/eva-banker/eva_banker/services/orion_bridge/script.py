import os
import sys

from orion_bridge.bridge_helper import (
    SourceStrategy,
    OpenPayload,
    ClosePayload,
    BridgeConnector,
)
from dotenv import load_dotenv


def _build_source():
    return SourceStrategy(
        name="eva", broker_name="FTMO", account_size=10000, strategy_name="orion_v1"
    )


def open_position():
    source_strategy = _build_source()
    payload = OpenPayload(source_ticket_id="123456789", symbol="EURUSD", volume=0.10, type="buy")
    try:
        response = BridgeConnector().send_order(source_strategy=source_strategy, payload=payload)
    except Exception as e:
        print(e)
    finally:
        if response:
            print("Response: ", response.text)
        else:
            print("No response")


def close_position():
    source_strategy = _build_source()
    payload = ClosePayload(source_ticket_id="123456789", full_close=True, reason="reason")

    try:
        response = BridgeConnector().send_order(source_strategy=source_strategy, payload=payload)
    except Exception as e:
        print(e)
    finally:
        if response:
            print("Response: ", response.text)
        else:
            print("No response")


def close_partial_position():
    source_strategy = _build_source()
    payload = ClosePayload(source_ticket_id="123456789", full_close=False, reason="reason")

    try:
        response = BridgeConnector().send_order(source_strategy=source_strategy, payload=payload)
    except Exception as e:
        print(e)
    finally:
        if response:
            print("Response: ", response.text)
        else:
            print("No response")


if __name__ == "__main__":
    load_dotenv()
    cmd = sys.argv[1]
    if cmd == "open":
        open_position()
        # print("lets open")
    elif cmd == "close":
        close_position()
    elif cmd == "partial":
        close_partial_position()
    else:
        print("Erreur: arg must be one of 'open', 'close', 'partial'")
