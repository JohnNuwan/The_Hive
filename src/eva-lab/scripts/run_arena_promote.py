"""Arena + Promotion automatique des champions post-entraînement.

Orchestre pour chaque horizon (scalp, intraday, swing) :
  1. Résolution du dernier checkpoint challenger
  2. Duel Arena : challenger vs champion courant
  3. Promotion automatique si VICTORY + gate validée
  4. Rapport JSON écrit dans data/checkpoints/arena_result_latest.json

Contrôlé par :
  MUZERO_HORIZONS        : horizons à évaluer (défaut : scalp)
  ARENA_GAMES_PER_SYMBOL : parties par symbole (défaut : 8)
  ARENA_MIN_GAMES        : minimum de parties pour valider (défaut : 24)
  ARENA_MIN_SYMBOLS      : minimum de symboles couverts (défaut : 6)
  ARENA_MAX_SYMBOLS      : symboles max à évaluer (0 = tous, défaut : 0)
  RUN_DREAMER_ARENA      : activer l'Arena Dreamer (défaut : 1)
"""

from __future__ import annotations

import json
import logging
import os
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

from eva_lab.arena import Arena
from eva_lab.champion_promoter import ChampionPromoter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eva_lab.run_arena_promote")

WORKDIR = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = WORKDIR / "data" / "muzero" / "weights"
ARENA_RESULT_PATH = WORKDIR / "data" / "checkpoints" / "arena_result_latest.json"
ARENA_REPORTS_DIR = WORKDIR / "data" / "checkpoints" / "arena_reports"


def _env_flag(name: str, default: bool) -> bool:
    """Lit un flag booléen depuis les variables d'environnement."""
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() not in {"0", "false", "no", "off"}


def _resolve_horizons() -> list[str]:
    """Résout les horizons à évaluer selon MUZERO_HORIZONS."""
    raw = os.getenv("MUZERO_HORIZONS", "scalp")
    return [h.strip().lower() for h in raw.split(",") if h.strip()]


def _find_best_challenger(horizon: str, engine: str = "muzero") -> tuple[Path | None, str | None]:
    """Retourne le meilleur checkpoint challenger disponible pour un horizon.

    Priorité :
      1. *_latest.pkl  (dernier modèle entraîné)
      2. ckpt_*.pkl le plus récent

    Returns:
        tuple[Path | None, str | None]: (chemin, identifiant_generique)
    """
    prefix = "muzero" if engine == "muzero" else "dreamer"
    # Chercher *_latest.pkl pour cet horizon
    candidates = sorted(
        WEIGHTS_DIR.glob(f"{prefix}_{horizon}_latest.pkl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        path = candidates[0]
        ts = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y%m%d_%H%M%S")
        return path, f"{prefix}_{horizon}_{ts}_challenger"

    # Fallback : dernier ckpt
    ckpt_candidates = sorted(
        list(WEIGHTS_DIR.glob(f"{prefix}_{horizon}_ckpt_*.pkl"))
        + list(WEIGHTS_DIR.glob(f"{prefix}_{horizon}_ckpt*.pkl")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if ckpt_candidates:
        path = ckpt_candidates[0]
        ts = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y%m%d_%H%M%S")
        return path, f"{prefix}_{horizon}_{ts}_ckpt_challenger"

    return None, None


def _run_arena_for_horizon(
    horizon: str,
    engine: str,
    arena: Arena,
    promoter: ChampionPromoter,
) -> dict:
    """Lance l'Arena et tente une promotion pour un horizon et un moteur.

    Args:
        horizon (str): Horizon à évaluer.
        engine (str): Moteur : ``muzero`` ou ``dreamer``.
        arena (Arena): Instance Arena partagée.
        promoter (ChampionPromoter): Promoteur pour la promotion live.

    Returns:
        dict: Résultat du duel et de la tentative de promotion.
    """
    challenger_path, challenger_id = _find_best_challenger(horizon, engine)
    if challenger_path is None:
        logger.warning(
            "Arena %s/%s : aucun checkpoint challenger trouvé. Étape ignorée.",
            engine,
            horizon,
        )
        return {
            "horizon": horizon,
            "engine": engine,
            "outcome": "SKIPPED",
            "reason": "no_challenger_found",
        }

    logger.info(
        "Arena %s/%s : challenger=%s (%s)",
        engine,
        horizon,
        challenger_id,
        challenger_path.name,
    )

    # Résolution du champion courant
    current_champion_path = promoter.get_champion_path(horizon, engine=engine)
    champion_id = (
        "muzero_champion" if not current_champion_path.exists()
        else f"champion_{horizon}_{engine}"
    )

    try:
        battle_report = arena.battle(
            challenger_id=challenger_id,
            champion_id=champion_id if current_champion_path.exists() else "gen_000_baseline",
            horizon=horizon,
            engine=engine,
        )
    except Exception as battle_exc:
        logger.exception("Duel Arena %s/%s échoué : %s", engine, horizon, battle_exc)
        return {
            "horizon": horizon,
            "engine": engine,
            "outcome": "ERROR",
            "reason": str(battle_exc),
        }

    outcome = battle_report.get("outcome", "DEFEAT")
    action_required = battle_report.get("action_required", "KEEP_CURRENT")
    challenger_score = battle_report.get("challenger", {}).get("score", 0.0)
    champion_score = battle_report.get("champion", {}).get("score", 0.0)

    logger.info(
        "Arena %s/%s → %s (%s) | challenger=%.4f | champion=%.4f",
        engine,
        horizon,
        outcome,
        action_required,
        challenger_score,
        champion_score,
    )

    # Sauvegarde du rapport Arena par horizon
    ARENA_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = ARENA_REPORTS_DIR / f"arena_{engine}_{horizon}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(
        json.dumps(battle_report, indent=2, default=float, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Rapport Arena sauvegardé : %s", report_path)

    # Promotion si VICTORY
    promotion_result: dict = {}
    if outcome == "VICTORY":
        logger.info(
            "VICTORY %s/%s — tentative de promotion en champion live...",
            engine,
            horizon,
        )
        try:
            promote_fn = (
                promoter.promote_muzero_challenger
                if engine == "muzero"
                else promoter.promote_dreamer_challenger
            )
            promotion_result = promote_fn(
                challenger_path=challenger_path,
                horizon=horizon,
                battle_report=battle_report,
                challenger_id=challenger_id,
                gate_profile="standard",
            )
            promo_status = promotion_result.get("status", "unknown")
            promo_reason = promotion_result.get("reason", "")
            logger.info(
                "Promotion %s/%s : status=%s | reason=%s | paths=%s",
                engine,
                horizon,
                promo_status,
                promo_reason,
                promotion_result.get("champion_paths", []),
            )
        except Exception as promo_exc:
            logger.exception("Promotion %s/%s échouée : %s", engine, horizon, promo_exc)
            promotion_result = {"status": "error", "reason": str(promo_exc)}
    else:
        logger.info(
            "Pas de promotion %s/%s : %s (%s)",
            engine,
            horizon,
            outcome,
            action_required,
        )

    return {
        "horizon": horizon,
        "engine": engine,
        "challenger_id": challenger_id,
        "challenger_path": str(challenger_path),
        "outcome": outcome,
        "action_required": action_required,
        "challenger_score": challenger_score,
        "champion_score": champion_score,
        "validation": battle_report.get("validation", {}),
        "promotion": promotion_result,
        "report_path": str(report_path),
    }


def main() -> dict:
    """Orchestre l'Arena et la promotion pour tous les horizons demandés.

    Returns:
        dict: Résumé global de l'Arena (tous horizons, tous moteurs).
    """
    horizons = _resolve_horizons()
    run_dreamer_arena = _env_flag("RUN_DREAMER_ARENA", True)

    logger.info(
        "Arena + Promotion : horizons=%s | dreamer=%s",
        horizons,
        run_dreamer_arena,
    )

    arena = Arena(
        weights_dir=str(WEIGHTS_DIR),
        data_dir=str(WORKDIR / "data" / "history"),
    )
    promoter = ChampionPromoter(
        weights_dir=str(WEIGHTS_DIR),
        results_dir=str(WORKDIR / "data" / "muzero" / "results"),
    )

    results: list[dict] = []
    engines = ["muzero"]
    if run_dreamer_arena:
        engines.append("dreamer")

    for horizon in horizons:
        for engine in engines:
            try:
                result = _run_arena_for_horizon(horizon, engine, arena, promoter)
                results.append(result)
            except Exception as exc:
                logger.exception(
                    "Erreur inattendue Arena %s/%s : %s", engine, horizon, exc
                )
                results.append({
                    "horizon": horizon,
                    "engine": engine,
                    "outcome": "ERROR",
                    "reason": str(exc),
                })

    # Résumé global
    victories = [r for r in results if r.get("outcome") == "VICTORY"]
    promoted = [
        r for r in victories
        if (r.get("promotion") or {}).get("status") == "promoted"
    ]

    summary = {
        "exported_at": datetime.now().isoformat(),
        "horizons": horizons,
        "engines": engines,
        "total_duels": len(results),
        "victories": len(victories),
        "promotions": len(promoted),
        "results": results,
    }

    # Écriture atomique du résultat
    ARENA_RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = ARENA_RESULT_PATH.with_suffix(".tmp")
    tmp_path.write_text(
        json.dumps(summary, indent=2, default=float, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp_path.replace(ARENA_RESULT_PATH)

    logger.info(
        "Arena terminée : %d duel(s) | %d victoire(s) | %d promotion(s) → %s",
        len(results),
        len(victories),
        len(promoted),
        ARENA_RESULT_PATH,
    )

    if promoted:
        for r in promoted:
            champ_paths = (r.get("promotion") or {}).get("champion_paths", [])
            logger.info("🏆 CHAMPION promu : %s/%s → %s", r["engine"], r["horizon"], champ_paths)

    return summary


if __name__ == "__main__":
    result = main()
    print(json.dumps(result, indent=2, default=float, ensure_ascii=False))
