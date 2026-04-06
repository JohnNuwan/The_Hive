import sys
import threading
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "eva-lab"))
sys.path.insert(0, str(ROOT / "src" / "shared"))

import eva_lab.training_status as training_status


def test_atomic_write_json_supports_parallel_writers(tmp_path):
    target = tmp_path / "training_status.json"
    errors: list[Exception] = []

    def worker(index: int) -> None:
        try:
            for step in range(25):
                training_status._atomic_write_json(
                    target,
                    {"worker": index, "step": step},
                )
        except Exception as exc:  # pragma: no cover - la collecte d'erreur est le sujet du test.
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    payload = target.read_text(encoding="utf-8")
    assert '"worker"' in payload
    assert '"step"' in payload
