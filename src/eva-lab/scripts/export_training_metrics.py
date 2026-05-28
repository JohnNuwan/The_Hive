"""Export des metriques d'entrainement en JSON structure.

Lit les logs de la sequence nightly et produit un fichier JSON unique
`data/checkpoints/training_metrics_latest.json` consommable par le
Nexus Dashboard et les outils de supervision.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

# Configuration du PYTHONPATH pour EVA Lab
package_root = Path(__file__).resolve().parents[1]
shared_root = package_root.parent / "shared"
for candidate in (package_root, shared_root):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eva_lab.export_training_metrics")

WORKDIR = Path(__file__).resolve().parents[1]

def _resolve_data_dir() -> Path:
    """Résout le répertoire data en testant plusieurs chemins possibles.

    Priorité :
      1. Variable d'environnement HIVE_DATA_DIR (injection Docker)
      2. data/ relatif au package root (container : /app/eva-lab/data)
      3. ../../data relatif au package root (host : /home/aza/The_Hive/data)
    """
    # 1. Env var explicite
    env_dir = os.getenv("HIVE_DATA_DIR")
    if env_dir:
        p = Path(env_dir)
        if p.exists():
            return p

    # 2. Chemin container standard (/app/eva-lab/data)
    container_path = WORKDIR / "data"
    if (container_path / "checkpoints" / "training_run.log").exists():
        return container_path

    # 3. Chemin host (/home/aza/The_Hive/data)
    host_path = WORKDIR.parent.parent / "data"
    if (host_path / "checkpoints" / "training_run.log").exists():
        return host_path

    # 4. Fallback : on retourne le chemin container même s'il n'existe pas
    return container_path

DATA_DIR = _resolve_data_dir()
TRAINING_RUN_LOG = DATA_DIR / "checkpoints" / "training_run.log"
TRAINING_STATUS_JSON = DATA_DIR / "checkpoints" / "training_status.json"
NIGHTLY_SUMMARY_JSON = DATA_DIR / "checkpoints" / "nightly_training_summary.json"
GNN_METRICS_JSON = DATA_DIR / "models" / "gnn_master_metrics.json"
ARENA_RESULT_JSON = DATA_DIR / "checkpoints" / "arena_result_latest.json"
OUTPUT_PATH = DATA_DIR / "checkpoints" / "training_metrics_latest.json"


def _load_json_safe(path: Path) -> dict:
    """Charge un fichier JSON sans planter si absent ou corrompu."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _parse_gnn_metrics_from_log(log_path: Path) -> dict:
    """Extrait les dernières métriques GNN depuis training_run.log.

    Cherche les lignes du format :
      [gnn] GNN: epoch {n}/{total} | loss={loss} | scalp={pct}%
    """
    metrics: dict = {}
    if not log_path.exists():
        return metrics

    # Lire les 3000 dernières lignes pour ne pas tout charger
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()[-3000:]
    except OSError:
        return metrics

    pattern = re.compile(
        r"\[gnn\] GNN: epoch (\d+)/(\d+)"
        r" \| loss=([\d.]+)"
        r"(?: \| scalp=([\d.]+)%)?"
    )

    last_epoch = 0
    for line in reversed(lines):
        m = pattern.search(line)
        if m:
            epoch = int(m.group(1))
            if epoch > last_epoch:
                last_epoch = epoch
                metrics = {
                    "epoch": epoch,
                    "epoch_total": int(m.group(2)),
                    "loss": float(m.group(3)),
                    "scalp_accuracy_pct": float(m.group(4)) if m.group(4) else None,
                    "progress_pct": round(100.0 * epoch / int(m.group(2)), 2),
                }
                break

    return metrics


def _parse_muzero_metrics_from_log(log_path: Path) -> dict:
    """Extrait les dernières métriques MuZero depuis training_run.log."""
    metrics: dict = {}
    if not log_path.exists():
        return metrics

    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()[-3000:]
    except OSError:
        return metrics

    # Collecte : [muzero] MuZero scalp: collecte sur BTCUSD (3/12)
    collect_pattern = re.compile(
        r"\[muzero\] MuZero (\w+): collecte sur (\S+) \((\d+)/(\d+)\)"
    )
    # Optimisation : [muzero] MuZero scalp: step 1234/32000 | loss=0.345
    opt_pattern = re.compile(
        r"\[muzero\] MuZero (\w+): step (\d+)/(\d+)"
        r"(?: \| loss=([\d.]+))?"
    )

    for line in reversed(lines):
        m_opt = opt_pattern.search(line)
        if m_opt and "optimize_step" not in metrics:
            metrics["optimize_step"] = {
                "horizon": m_opt.group(1),
                "step": int(m_opt.group(2)),
                "step_total": int(m_opt.group(3)),
                "loss": float(m_opt.group(4)) if m_opt.group(4) else None,
            }

        m_col = collect_pattern.search(line)
        if m_col and "collect_step" not in metrics:
            metrics["collect_step"] = {
                "horizon": m_col.group(1),
                "symbol": m_col.group(2),
                "symbol_index": int(m_col.group(3)),
                "symbol_total": int(m_col.group(4)),
            }

        if len(metrics) >= 2:
            break

    return metrics


def _parse_dreamer_metrics_from_log(log_path: Path) -> dict:
    """Extrait les dernières métriques DreamerV3 depuis training_run.log."""
    metrics: dict = {}
    if not log_path.exists():
        return metrics

    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()[-2000:]
    except OSError:
        return metrics

    # [dreamer] Dreamer: 2500 episodes charges pour 2399488 pas de temps.
    load_pattern = re.compile(
        r"\[dreamer\] Dreamer: (\d+) episodes charges pour ([\d]+) pas de temps"
    )
    # [dreamer] Dreamer: epoch 42/1500 | loss=0.123
    epoch_pattern = re.compile(
        r"\[dreamer\] Dreamer: epoch (\d+)/(\d+)"
        r"(?: \| loss=([\d.]+))?"
    )

    for line in reversed(lines):
        m_ep = epoch_pattern.search(line)
        if m_ep and "epoch" not in metrics:
            metrics["epoch"] = int(m_ep.group(1))
            metrics["epoch_total"] = int(m_ep.group(2))
            metrics["loss"] = float(m_ep.group(3)) if m_ep.group(3) else None

        m_load = load_pattern.search(line)
        if m_load and "episodes_loaded" not in metrics:
            metrics["episodes_loaded"] = int(m_load.group(1))
            metrics["timesteps_loaded"] = int(m_load.group(2))

        if len(metrics) >= 4:
            break

    return metrics


def _extract_champion_info() -> dict:
    """Résume l'état des champions actuels (fichiers présents)."""
    weights_dir = DATA_DIR / "muzero" / "weights"
    champions: dict = {}

    for horizon in ("scalp", "intraday", "swing"):
        muzero_champ = weights_dir / f"muzero_champion_{horizon}.pkl"
        dreamer_champ = weights_dir / f"dreamer_champion_{horizon}.pkl"
        champions[horizon] = {
            "muzero_champion": {
                "exists": muzero_champ.exists(),
                "path": str(muzero_champ),
                "modified_at": (
                    datetime.fromtimestamp(muzero_champ.stat().st_mtime).isoformat()
                    if muzero_champ.exists() else None
                ),
            },
            "dreamer_champion": {
                "exists": dreamer_champ.exists(),
                "path": str(dreamer_champ),
                "modified_at": (
                    datetime.fromtimestamp(dreamer_champ.stat().st_mtime).isoformat()
                    if dreamer_champ.exists() else None
                ),
            },
        }

    # Fallback : champion générique
    for name in ("muzero_champion.pkl", "dreamer_champion.pkl", "gnn_champion.pkl"):
        p = weights_dir / name
        if p.exists():
            champions[name.replace(".pkl", "")] = {
                "exists": True,
                "path": str(p),
                "modified_at": datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
            }

    return champions


def ingest_all_parsed_metrics_to_db(log_path: Path, run_id: str | None = None) -> None:
    """Parcourt l'intégralité de training_run.log, extrait toutes les métriques de convergence et les insère dans TimescaleDB."""
    if not log_path.exists():
        logger.warning("Fichier de log introuvable : %s", log_path)
        return

    logger.info("Ingestion des métriques depuis %s dans TimescaleDB...", log_path)

    try:
        from eva_lab.timescale_store import insert_training_metric
    except ImportError as exc:
        logger.warning("timescale_store indisponible : %s. Ingestion annulée.", exc)
        return

    # Compilation des motifs regex
    # GNN : [gnn] GNN: epoch 42/500 | loss=0.1234 | scalp=78.5%
    gnn_pattern = re.compile(
        r"\[gnn\] GNN: epoch (\d+)/(\d+) \| loss=([\d.]+) \| scalp=([\d.]+)%"
    )
    # MuZero : [muzero] MuZero scalp: step 1234/32000 | loss=0.345
    muzero_pattern = re.compile(
        r"\[muzero\] MuZero (\w+): step (\d+)/(\d+)(?: \| loss=([\d.]+))?"
    )
    # Dreamer : [dreamer] Dreamer: epoch 42/1500 | loss=0.123
    dreamer_pattern = re.compile(
        r"\[dreamer\] Dreamer: epoch (\d+)/(\d+)(?: \| loss=([\d.]+))?"
    )
    # VICReg (JEPA) : Étape 200/3000 | Perte Totale: 0.1234 | Inv: 0.111 | Var: 0.222 | Cov: 0.033
    jepa_pattern = re.compile(
        r"Étape (\d+)/(\d+) \| Perte Totale: ([\d.-]+) \| Inv: ([\d.-]+) \| Var: ([\d.-]+) \| Cov: ([\d.-]+)"
    )

    count = 0
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                # 1. Tester GNN
                m_gnn = gnn_pattern.search(line)
                if m_gnn:
                    epoch = int(m_gnn.group(1))
                    loss = float(m_gnn.group(3))
                    scalp = float(m_gnn.group(4))
                    insert_training_metric("gnn", "loss", loss, epoch, epoch, run_id)
                    insert_training_metric("gnn", "scalp_accuracy", scalp, epoch, epoch, run_id)
                    count += 2
                    continue

                # 2. Tester MuZero
                m_mu = muzero_pattern.search(line)
                if m_mu:
                    horizon = m_mu.group(1)
                    step = int(m_mu.group(2))
                    loss_val = m_mu.group(4)
                    if loss_val:
                        insert_training_metric(f"muzero_{horizon}", "loss", float(loss_val), step, None, run_id)
                        count += 1
                    continue

                # 3. Tester Dreamer
                m_dr = dreamer_pattern.search(line)
                if m_dr:
                    epoch = int(m_dr.group(1))
                    loss_val = m_dr.group(3)
                    if loss_val:
                        insert_training_metric("dreamer", "loss", float(loss_val), epoch, epoch, run_id)
                        count += 1
                    continue

                # 4. Tester JEPA
                m_je = jepa_pattern.search(line)
                if m_je:
                    step = int(m_je.group(1))
                    loss = float(m_je.group(3))
                    inv = float(m_je.group(4))
                    var = float(m_je.group(5))
                    cov = float(m_je.group(6))
                    insert_training_metric("jepa", "loss_total", loss, step, None, run_id)
                    insert_training_metric("jepa", "loss_invariance", inv, step, None, run_id)
                    insert_training_metric("jepa", "loss_variance", var, step, None, run_id)
                    insert_training_metric("jepa", "loss_covariance", cov, step, None, run_id)
                    count += 4
                    continue
        logger.info("Ingestion terminée : %d lignes de métriques enregistrées dans TimescaleDB.", count)
    except Exception as exc:
        logger.warning("Erreur lors de l'ingestion des métriques : %s", exc)


def main() -> dict:
    """Consolide et exporte toutes les métriques d'entraînement en JSON."""
    logger.info("Export des métriques d'entraînement vers %s", OUTPUT_PATH)

    # 1. Métriques live depuis les logs
    gnn_metrics = _parse_gnn_metrics_from_log(TRAINING_RUN_LOG)
    muzero_metrics = _parse_muzero_metrics_from_log(TRAINING_RUN_LOG)
    dreamer_metrics = _parse_dreamer_metrics_from_log(TRAINING_RUN_LOG)

    # 2. Fichiers JSON existants
    training_status = _load_json_safe(TRAINING_STATUS_JSON)
    nightly_summary = _load_json_safe(NIGHTLY_SUMMARY_JSON)
    gnn_master_metrics = _load_json_safe(GNN_METRICS_JSON)
    arena_result = _load_json_safe(ARENA_RESULT_JSON)

    # 3. État des champions sur disque
    champion_info = _extract_champion_info()

    # 4. Bilan consolidé
    payload: dict = {
        "exported_at": datetime.now().isoformat(),
        "run_id": training_status.get("run_id"),
        "strategy": nightly_summary.get("strategy"),
        "nightly_status": nightly_summary.get("status"),
        "nightly_steps": nightly_summary.get("steps", []),
        "gnn": {
            "live": gnn_metrics,
            "master_metrics": gnn_master_metrics,
        },
        "muzero": {
            "live": muzero_metrics,
            "training_status_step": training_status.get("current_step"),
        },
        "dreamer": {
            "live": dreamer_metrics,
        },
        "champions": champion_info,
        "arena_last_result": arena_result,
    }

    # 5. Écriture atomique
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = OUTPUT_PATH.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(OUTPUT_PATH)

    logger.info(
        "Métriques exportées : GNN epoch=%s loss=%s | champions_scalp_muzero=%s | champions_scalp_dreamer=%s",
        gnn_metrics.get("epoch"),
        gnn_metrics.get("loss"),
        champion_info.get("scalp", {}).get("muzero_champion", {}).get("exists"),
        champion_info.get("scalp", {}).get("dreamer_champion", {}).get("exists"),
    )

    # 6. Ingestion asynchrone dans TimescaleDB
    try:
        ingest_all_parsed_metrics_to_db(TRAINING_RUN_LOG, run_id=training_status.get("run_id"))
    except Exception as db_exc:
        logger.warning("Échec de l'ingestion TimescaleDB : %s", db_exc)

    return payload


if __name__ == "__main__":
    result = main()
    print(json.dumps(result, indent=2, ensure_ascii=False))

