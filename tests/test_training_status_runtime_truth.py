import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "eva-lab"))
sys.path.insert(0, str(ROOT / "src" / "shared"))

from eva_lab.training_status import normalize_runtime_training_status


def test_normalize_runtime_training_status_rehydrates_running_container_state():
    status = {
        "active": False,
        "status": "running",
        "current_step": {"name": "muzero_scalp", "status": "running"},
        "launcher": {
            "phase": "trainer_running",
            "trainer_container": "the_hive-eva-trainer-run-test",
            "remote_pid": None,
        },
    }

    normalized = normalize_runtime_training_status(status, sequence_state={"state": "running"})

    assert normalized["active"] is True
    assert normalized["status"] == "running"
    assert normalized["current_step"]["status"] == "running"
