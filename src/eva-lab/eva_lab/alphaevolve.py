"""Scaffold offline d'optimisation de variantes inspire d'AlphaEvolve."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from eva_lab.timescale_store import load_ga_trials


@dataclass(frozen=True, slots=True)
class ParametreMutation:
    """Decrit une mutation sure appliquee a un parametre de recherche."""

    nom: str
    minimum: float
    maximum: float
    ratio_mutation: float
    precision: int = 4


@dataclass(slots=True)
class VarianteAlphaEvolve:
    """Represente une variante candidate generee offline."""

    variant_id: str
    campaign_id: str
    created_at: str
    parent_label: str
    params: dict[str, float]
    notes: list[str] = field(default_factory=list)
    score_proxy: float | None = None
    score_arena: float | None = None
    score_nemesis: float | None = None
    rejection_reason: str | None = None


@dataclass(slots=True)
class CampagneAlphaEvolve:
    """Regroupe les variantes et leur contexte d'evaluation offline."""

    campaign_id: str
    created_at: str
    mode: str
    parent_label: str
    base_profile: dict[str, float]
    variants: list[VarianteAlphaEvolve]


MUTATIONS_SURES: tuple[ParametreMutation, ...] = (
    ParametreMutation("split_window_activation_bonus", 0.08, 0.24, 0.18),
    ParametreMutation("runner_window_hold_bonus", 0.06, 0.18, 0.16),
    ParametreMutation("pyramid_window_activation_bonus", 0.08, 0.20, 0.16),
    ParametreMutation("missed_window_penalty", 0.02, 0.08, 0.20),
    ParametreMutation("giveback_soft_penalty", 0.06, 0.16, 0.18),
    ParametreMutation("giveback_hard_penalty", 0.14, 0.30, 0.18),
    ParametreMutation("muzero_collection_num_simulations_xauusd", 96.0, 192.0, 0.20, precision=0),
    ParametreMutation("muzero_collection_max_moves_xauusd", 56.0, 96.0, 0.18, precision=0),
    ParametreMutation("muzero_collection_max_episode_seconds_xauusd", 120.0, 210.0, 0.18, precision=0),
)

PROFIL_DE_BASE = {
    "split_window_activation_bonus": 0.14,
    "runner_window_hold_bonus": 0.10,
    "pyramid_window_activation_bonus": 0.12,
    "missed_window_penalty": 0.05,
    "giveback_soft_penalty": 0.10,
    "giveback_hard_penalty": 0.22,
    "muzero_collection_num_simulations_xauusd": 128.0,
    "muzero_collection_max_moves_xauusd": 72.0,
    "muzero_collection_max_episode_seconds_xauusd": 150.0,
}


def _horodatage() -> str:
    """Retourne un horodatage compact compatible nom de fichier."""

    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _arrondir(valeur: float, precision: int) -> float:
    """Arrondit une valeur numerique pour garder des variantes lisibles."""

    if precision <= 0:
        return float(int(round(valeur)))
    return round(float(valeur), precision)


def _muter_valeur(
    valeur_initiale: float,
    mutation: ParametreMutation,
    generateur: random.Random,
) -> float:
    """Applique une mutation bornee autour d'une valeur de reference."""

    amplitude = max(abs(float(valeur_initiale)) * mutation.ratio_mutation, 0.01)
    valeur = float(valeur_initiale) + generateur.uniform(-amplitude, amplitude)
    valeur = min(max(valeur, mutation.minimum), mutation.maximum)
    return _arrondir(valeur, mutation.precision)


def charger_profil_de_base(chemin: Path | None = None) -> dict[str, float]:
    """Charge un profil de base ou retourne le profil integre.

    Args:
        chemin (Path | None): Fichier JSON optionnel contenant le profil source.

    Returns:
        dict[str, float]: Profil de base normalise.

    Raises:
        ValueError: Si le fichier fourni n'est pas un objet JSON valide.
    """

    if chemin is None:
        return dict(PROFIL_DE_BASE)
    payload = json.loads(chemin.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Le profil AlphaEvolve doit etre un objet JSON.")
    profil = dict(PROFIL_DE_BASE)
    for key, value in payload.items():
        try:
            profil[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return profil


def generer_variantes(
    *,
    campaign_id: str,
    parent_label: str,
    base_profile: dict[str, float],
    variant_count: int = 8,
    seed: int = 42,
) -> list[VarianteAlphaEvolve]:
    """Genere des variantes offline sans effet sur le live.

    Args:
        campaign_id (str): Identifiant de campagne offline.
        parent_label (str): Label de la reference parente.
        base_profile (dict[str, float]): Parametres de depart.
        variant_count (int): Nombre de variantes a construire.
        seed (int): Graine pseudo-aleatoire stable.

    Returns:
        list[VarianteAlphaEvolve]: Variantes generees et pretes a evaluer.
    """

    generateur = random.Random(seed)
    variantes: list[VarianteAlphaEvolve] = []
    mutations = list(MUTATIONS_SURES)
    for index in range(max(1, int(variant_count))):
        params = dict(base_profile)
        notes: list[str] = []
        nombre_mutations = 2 if len(mutations) >= 2 else 1
        for mutation in generateur.sample(mutations, k=nombre_mutations):
            valeur_initiale = float(params.get(mutation.nom, PROFIL_DE_BASE.get(mutation.nom, mutation.minimum)))
            nouvelle_valeur = _muter_valeur(valeur_initiale, mutation, generateur)
            params[mutation.nom] = nouvelle_valeur
            notes.append(
                f"{mutation.nom}: {valeur_initiale} -> {nouvelle_valeur}"
            )
        variantes.append(
            VarianteAlphaEvolve(
                variant_id=f"{campaign_id}_variant_{index + 1:03d}",
                campaign_id=campaign_id,
                created_at=datetime.utcnow().isoformat(),
                parent_label=parent_label,
                params=params,
                notes=notes,
            )
        )
    return variantes


def scorer_variantes_depuis_ga(
    variantes: list[VarianteAlphaEvolve],
    *,
    campaign_id: str | None = None,
) -> list[VarianteAlphaEvolve]:
    """Associe un score proxy aux variantes a partir des essais GA existants.

    Args:
        variantes (list[VarianteAlphaEvolve]): Variantes a enrichir.
        campaign_id (str | None): Campagne GA optionnelle pour filtrer les essais.

    Returns:
        list[VarianteAlphaEvolve]: Variantes enrichies avec un score proxy si possible.
    """

    essais = load_ga_trials(campaign_id=campaign_id, limit=512)
    index_scores: dict[str, dict[str, Any]] = {}
    for essai in essais:
        payload = essai.get("payload") or {}
        params = payload.get("params") or {}
        variant_id = str(params.get("alphaevolve_variant_id") or "").strip()
        if not variant_id:
            continue
        index_scores[variant_id] = {
            "fitness_score": essai.get("fitness_score"),
            "failure_mode": essai.get("failure_mode"),
        }

    for variante in variantes:
        score = index_scores.get(variante.variant_id)
        if score is None:
            continue
        try:
            variante.score_proxy = float(score.get("fitness_score"))
        except (TypeError, ValueError):
            variante.score_proxy = None
        failure_mode = str(score.get("failure_mode") or "").strip()
        if failure_mode:
            variante.rejection_reason = failure_mode
    return variantes


def construire_campagne(
    *,
    parent_label: str,
    base_profile: dict[str, float],
    variant_count: int = 8,
    seed: int = 42,
) -> CampagneAlphaEvolve:
    """Construit une campagne offline complete.

    Args:
        parent_label (str): Label de la reference parente.
        base_profile (dict[str, float]): Parametres de depart.
        variant_count (int): Nombre de variantes a produire.
        seed (int): Graine pseudo-aleatoire.

    Returns:
        CampagneAlphaEvolve: Campagne offline prete a serialiser.
    """

    campaign_id = f"alphaevolve_{_horodatage()}"
    variantes = generer_variantes(
        campaign_id=campaign_id,
        parent_label=parent_label,
        base_profile=base_profile,
        variant_count=variant_count,
        seed=seed,
    )
    return CampagneAlphaEvolve(
        campaign_id=campaign_id,
        created_at=datetime.utcnow().isoformat(),
        mode="offline_safe",
        parent_label=parent_label,
        base_profile=dict(base_profile),
        variants=variantes,
    )


def sauvegarder_campagne(campagne: CampagneAlphaEvolve, dossier_sortie: Path) -> Path:
    """Ecrit une campagne AlphaEvolve sur disque.

    Args:
        campagne (CampagneAlphaEvolve): Campagne a serialiser.
        dossier_sortie (Path): Dossier cible.

    Returns:
        Path: Chemin du fichier JSON ecrit.
    """

    dossier_sortie.mkdir(parents=True, exist_ok=True)
    chemin = dossier_sortie / f"{campagne.campaign_id}.json"
    payload = asdict(campagne)
    chemin.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return chemin


def charger_campagne(chemin: Path) -> CampagneAlphaEvolve:
    """Recharge une campagne serialisee.

    Args:
        chemin (Path): Fichier JSON d'une campagne.

    Returns:
        CampagneAlphaEvolve: Objet metier reconstruit.
    """

    payload = json.loads(chemin.read_text(encoding="utf-8"))
    variantes = [VarianteAlphaEvolve(**item) for item in payload.get("variants", [])]
    return CampagneAlphaEvolve(
        campaign_id=str(payload["campaign_id"]),
        created_at=str(payload["created_at"]),
        mode=str(payload.get("mode") or "offline_safe"),
        parent_label=str(payload.get("parent_label") or "unknown"),
        base_profile=dict(payload.get("base_profile") or {}),
        variants=variantes,
    )

