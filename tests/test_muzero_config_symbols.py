import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "eva-lab"))
sys.path.insert(0, str(ROOT / "src" / "shared"))

from eva_lab.muzero.config import MuZeroConfigV3


def test_explicit_symbols_override_family_hint(monkeypatch):
    monkeypatch.setenv("MUZERO_HORIZON", "scalp")
    monkeypatch.setenv("MUZERO_MODEL_FAMILY", "fx")
    monkeypatch.setenv("MUZERO_MAX_SYMBOLS", "7")
    monkeypatch.setenv("MUZERO_DATASET_SOURCE", "timescaledb")
    monkeypatch.setenv("TRAINING_FOCUS_SYMBOLS", "EURUSD,XAUUSD,GBPUSD")
    monkeypatch.delenv("MUZERO_SYMBOLS_SCALP", raising=False)
    monkeypatch.delenv("MUZERO_SYMBOLS", raising=False)

    config = MuZeroConfigV3()

    assert config.symbols == ["EURUSD", "XAUUSD", "GBPUSD"]
    assert config.model_family == "mixed"
