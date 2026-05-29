"""Utilitaires MuZero pour serialiser, verifier et archiver les checkpoints."""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

CHECKPOINT_SCHEMA_VERSION = 2
DEFAULT_PARAM_DTYPE = "float32"


class MuZeroCheckpointCompatibilityError(RuntimeError):
    """Signale qu'un checkpoint MuZero ne peut pas etre charge."""


def _now_iso() -> str:
    """Retourne l'horodatage courant au format ISO."""

    return datetime.now().isoformat()


def _sanitize_token(value: Any, default: str = "artifact") -> str:
    """Convertit une valeur libre en nom de fichier stable."""

    raw_value = str(value or "").strip()
    if not raw_value:
        return default
    sanitized = [
        char if char.isalnum() or char in {"-", "_", "."} else "_"
        for char in raw_value
    ]
    cleaned = "".join(sanitized).strip("._")
    return cleaned or default


def _coerce_shape(value: Any) -> list[int]:
    """Retourne une forme numerique serialisable pour une feuille de pytree."""

    try:
        return [int(item) for item in tuple(np.shape(value))]
    except Exception:
        return []


def _coerce_dtype(value: Any) -> str:
    """Retourne le dtype textuel le plus robuste possible."""

    try:
        return str(np.asarray(value).dtype)
    except Exception:
        return type(value).__name__


def _flatten_tree_signature(tree: Any, prefix: str = "") -> list[dict[str, Any]]:
    """Aplati un pytree en signature stable basee sur les chemins et formes."""

    if isinstance(tree, dict) or (hasattr(tree, "keys") and hasattr(tree, "__getitem__")):
        flattened: list[dict[str, Any]] = []
        for key in sorted(list(tree.keys())):
            child_prefix = f"{prefix}/{key}" if prefix else str(key)
            flattened.extend(_flatten_tree_signature(tree[key], prefix=child_prefix))
        return flattened

    if isinstance(tree, (list, tuple)):
        flattened = []
        for index, item in enumerate(tree):
            child_prefix = f"{prefix}/{index}" if prefix else str(index)
            flattened.extend(_flatten_tree_signature(item, prefix=child_prefix))
        return flattened

    return [
        {
            "path": prefix or "<root>",
            "shape": _coerce_shape(tree),
            "dtype": _coerce_dtype(tree),
        }
    ]


def _compare_tree_signatures(
    *,
    expected_signature: list[dict[str, Any]],
    artifact_signature: list[dict[str, Any]],
) -> tuple[bool, str]:
    """Compare deux signatures de pytree et retourne un message stable."""

    expected_by_path = {
        str(item.get("path") or ""): item
        for item in expected_signature
    }
    artifact_by_path = {
        str(item.get("path") or ""): item
        for item in artifact_signature
    }

    expected_paths = set(expected_by_path)
    artifact_paths = set(artifact_by_path)

    missing_paths = sorted(expected_paths - artifact_paths)
    if missing_paths:
        return False, (
            "Arbre de poids incompatible: parametre manquant "
            f"{missing_paths[0]}."
        )

    extra_paths = sorted(artifact_paths - expected_paths)
    if extra_paths:
        return False, (
            "Arbre de poids incompatible: parametre inattendu "
            f"{extra_paths[0]}."
        )

    for path in sorted(expected_paths):
        expected_item = expected_by_path[path]
        artifact_item = artifact_by_path[path]
        expected_shape = list(expected_item.get("shape") or [])
        artifact_shape = list(artifact_item.get("shape") or [])
        if expected_shape != artifact_shape:
            return False, (
                "Arbre de poids incompatible: forme differente pour "
                f"{path} (attendu={expected_shape}, obtenu={artifact_shape})."
            )
        expected_dtype = str(expected_item.get("dtype") or "")
        artifact_dtype = str(artifact_item.get("dtype") or "")
        if expected_dtype != artifact_dtype:
            return False, (
                "Arbre de poids incompatible: dtype different pour "
                f"{path} (attendu={expected_dtype}, obtenu={artifact_dtype})."
            )

    return True, "Arbre de poids compatible avec l'architecture courante."


def build_muzero_config_snapshot(config: Any) -> dict[str, Any]:
    """Construit le sous-ensemble bloquant du contrat de checkpoint MuZero."""

    return {
        "observation_shape": [
            int(value)
            for value in tuple(getattr(config, "observation_shape", ()) or ())
        ],
        "action_space_size": int(getattr(config, "action_space_size", 0) or 0),
        "hidden_state_size": int(getattr(config, "hidden_state_size", 0) or 0),
        "network_hidden_dims": [
            int(value)
            for value in list(getattr(config, "network_hidden_dims", []) or [])
        ],
        "support_size": int(getattr(config, "support_size", 0) or 0),
        "use_jepa_encoder": bool(getattr(config, "use_jepa_encoder", False)),
        "jepa_latent_size": int(getattr(config, "jepa_latent_size", 128)),
    }


def build_muzero_config_fingerprint(config_snapshot: dict[str, Any]) -> dict[str, Any]:
    """Construit l'empreinte stable de configuration d'un checkpoint."""

    normalized = {
        "observation_shape": [int(value) for value in list(config_snapshot.get("observation_shape") or [])],
        "action_space_size": int(config_snapshot.get("action_space_size") or 0),
        "hidden_state_size": int(config_snapshot.get("hidden_state_size") or 0),
        "network_hidden_dims": [int(value) for value in list(config_snapshot.get("network_hidden_dims") or [])],
        "support_size": int(config_snapshot.get("support_size") or 0),
        "use_jepa_encoder": bool(config_snapshot.get("use_jepa_encoder", False)),
        "jepa_latent_size": int(config_snapshot.get("jepa_latent_size", 128)),
    }
    serialized = json.dumps(
        normalized,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        **normalized,
        "sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
    }


def _build_signature_leaf(path: str, shape: list[int], dtype: str = DEFAULT_PARAM_DTYPE) -> dict[str, Any]:
    """Construit une feuille de signature de parametre stable.

    Args:
        path (str): Chemin complet du parametre.
        shape (list[int]): Forme numerique attendue.
        dtype (str): Type de donnees attendu.

    Returns:
        dict[str, Any]: Feuille de signature serialisable.
    """

    return {
        "path": str(path),
        "shape": [int(value) for value in list(shape)],
        "dtype": str(dtype or DEFAULT_PARAM_DTYPE),
    }


def _build_linear_signature(
    module_prefix: str,
    input_dim: int,
    output_dim: int,
    *,
    dtype: str = DEFAULT_PARAM_DTYPE,
) -> list[dict[str, Any]]:
    """Construit la signature attendue d'une couche lineaire Haiku.

    Args:
        module_prefix (str): Prefixe de module Haiku.
        input_dim (int): Dimension d'entree.
        output_dim (int): Dimension de sortie.
        dtype (str): Type des poids et biais attendus.

    Returns:
        list[dict[str, Any]]: Signature stable des poids et biais.
    """

    return [
        _build_signature_leaf(f"{module_prefix}/w", [int(input_dim), int(output_dim)], dtype=dtype),
        _build_signature_leaf(f"{module_prefix}/b", [int(output_dim)], dtype=dtype),
    ]


def build_muzero_param_signature_from_snapshot(
    config_snapshot: dict[str, Any],
    *,
    dtype: str = DEFAULT_PARAM_DTYPE,
) -> list[dict[str, Any]]:
    """Construit la signature attendue des poids MuZero sans instancier JAX.

    Args:
        config_snapshot (dict[str, Any]): Sous-ensemble stable de configuration.
        dtype (str): Type de poids attendu pour les feuilles.

    Returns:
        list[dict[str, Any]]: Signature complete des poids MuZero.
    """

    observation_shape = [
        int(value)
        for value in list(config_snapshot.get("observation_shape") or [])
    ]
    input_dim = int(np.prod(observation_shape, dtype=np.int64)) if observation_shape else 0
    hidden_dims = [
        int(value)
        for value in list(config_snapshot.get("network_hidden_dims") or [])
    ]
    hidden_state_size = int(config_snapshot.get("hidden_state_size") or 0)
    action_space_size = int(config_snapshot.get("action_space_size") or 0)
    support_size = int(config_snapshot.get("support_size") or 0)
    support_bins = 2 * support_size + 1

    use_jepa_encoder = bool(config_snapshot.get("use_jepa_encoder", False))
    jepa_latent_size = int(config_snapshot.get("jepa_latent_size", 128))

    signature: list[dict[str, Any]] = []

    if use_jepa_encoder:
        # L'encodeur de contexte JEPA a 3 couches linéaires :
        # representation_network/context_encoder/linear : [input_dim, 256]
        signature.extend(
            _build_linear_signature(
                "representation_network/context_encoder/linear",
                input_dim,
                256,
                dtype=dtype,
            )
        )
        # representation_network/context_encoder/linear_1 : [256, 256]
        signature.extend(
            _build_linear_signature(
                "representation_network/context_encoder/linear_1",
                256,
                256,
                dtype=dtype,
            )
        )
        # representation_network/context_encoder/linear_2 : [256, jepa_latent_size]
        signature.extend(
            _build_linear_signature(
                "representation_network/context_encoder/linear_2",
                256,
                jepa_latent_size,
                dtype=dtype,
            )
        )
        representation_input_dim = jepa_latent_size
    else:
        representation_input_dim = input_dim

    representation_dims = [representation_input_dim, *hidden_dims, hidden_state_size]
    for index, (source_dim, target_dim) in enumerate(zip(representation_dims, representation_dims[1:])):
        suffix = "" if index == 0 else f"_{index}"
        signature.extend(
            _build_linear_signature(
                f"representation_network/linear{suffix}",
                source_dim,
                target_dim,
                dtype=dtype,
            )
        )

    dynamics_input_dim = hidden_state_size + action_space_size
    dynamics_hidden_dims = [dynamics_input_dim, *hidden_dims]
    for index, (source_dim, target_dim) in enumerate(zip(dynamics_hidden_dims, dynamics_hidden_dims[1:])):
        suffix = "" if index == 0 else f"_{index}"
        signature.extend(
            _build_linear_signature(
                f"dynamics_network/linear{suffix}",
                source_dim,
                target_dim,
                dtype=dtype,
            )
        )
    dynamics_output_index = len(hidden_dims)
    dynamics_source_dim = hidden_dims[-1] if hidden_dims else dynamics_input_dim
    signature.extend(
        _build_linear_signature(
            f"dynamics_network/linear_{dynamics_output_index}",
            dynamics_source_dim,
            hidden_state_size,
            dtype=dtype,
        )
    )
    signature.extend(
        _build_linear_signature(
            f"dynamics_network/linear_{dynamics_output_index + 1}",
            dynamics_source_dim,
            support_bins,
            dtype=dtype,
        )
    )

    prediction_input_dim = hidden_state_size
    prediction_hidden_dims = [prediction_input_dim, *hidden_dims]
    for index, (source_dim, target_dim) in enumerate(zip(prediction_hidden_dims, prediction_hidden_dims[1:])):
        suffix = "" if index == 0 else f"_{index}"
        signature.extend(
            _build_linear_signature(
                f"prediction_network/linear{suffix}",
                source_dim,
                target_dim,
                dtype=dtype,
            )
        )
    prediction_output_index = len(hidden_dims)
    prediction_source_dim = hidden_dims[-1] if hidden_dims else prediction_input_dim
    signature.extend(
        _build_linear_signature(
            f"prediction_network/linear_{prediction_output_index}",
            prediction_source_dim,
            action_space_size,
            dtype=dtype,
        )
    )
    signature.extend(
        _build_linear_signature(
            f"prediction_network/linear_{prediction_output_index + 1}",
            prediction_source_dim,
            support_bins,
            dtype=dtype,
        )
    )
    return signature


def build_muzero_expected_context_from_config(config: Any) -> dict[str, Any]:
    """Construit le contexte attendu MuZero a partir de la seule configuration.

    Args:
        config (Any): Configuration MuZero resolue.

    Returns:
        dict[str, Any]: Contexte de compatibilite sans initialiser JAX.
    """

    config_snapshot = build_muzero_config_snapshot(config)
    return {
        "engine": "muzero",
        "horizon": str(getattr(config, "horizon", "intraday") or "intraday").lower(),
        "config_snapshot": config_snapshot,
        "config_fingerprint": build_muzero_config_fingerprint(config_snapshot),
        "param_signature": build_muzero_param_signature_from_snapshot(config_snapshot),
    }


def build_muzero_expected_context(
    *,
    config: Any,
    expected_params: Any,
) -> dict[str, Any]:
    """Construit le contexte de compatibilite attendu pour MuZero."""

    config_snapshot = build_muzero_config_snapshot(config)
    return {
        "engine": "muzero",
        "horizon": str(getattr(config, "horizon", "intraday") or "intraday").lower(),
        "config_snapshot": config_snapshot,
        "config_fingerprint": build_muzero_config_fingerprint(config_snapshot),
        "param_signature": _flatten_tree_signature(expected_params),
    }


def build_muzero_checkpoint_payload(
    *,
    config: Any,
    params: Any,
    opt_state: Any,
    training_step_count: int | None = None,
    artifact_kind: str = "checkpoint",
    lineage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construit le payload structure v2 d'un checkpoint MuZero."""

    dataset_descriptor = dict(getattr(config, "dataset_descriptor", {}) or {})
    feature_profile = dict(getattr(config, "feature_profile", {}) or {})
    feature_profile_name = (
        str(feature_profile.get("profile_name") or dataset_descriptor.get("feature_profile") or "")
        .strip()
        or None
    )
    mechanics_profile_version = (
        str(
            getattr(config, "mechanics_profile_version", "")
            or dataset_descriptor.get("mechanics_profile_version")
            or ""
        ).strip()
        or None
    )
    config_snapshot = build_muzero_config_snapshot(config)
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "engine": "muzero",
        "horizon": str(getattr(config, "horizon", "intraday") or "intraday").lower(),
        "artifact_kind": str(artifact_kind or "checkpoint").strip() or "checkpoint",
        "created_at": _now_iso(),
        "params": params,
        "opt_state": opt_state,
        "training_step_count": (
            int(training_step_count)
            if training_step_count is not None
            else None
        ),
        "config_snapshot": config_snapshot,
        "config_fingerprint": build_muzero_config_fingerprint(config_snapshot),
        "dataset_descriptor": dataset_descriptor,
        "dataset_id": (
            str(getattr(config, "dataset_id", "") or dataset_descriptor.get("dataset_id") or "").strip()
            or None
        ),
        "feature_profile": feature_profile_name,
        "mechanics_profile_version": mechanics_profile_version,
        "symbols": [
            str(symbol).strip()
            for symbol in list(getattr(config, "symbols", []) or [])
            if str(symbol).strip()
        ],
        "lineage": dict(lineage or {}),
    }


def extract_checkpoint_schema_version(payload: Any) -> int | None:
    """Retourne la version de schema si le payload en declare une."""

    if not isinstance(payload, dict):
        return None
    raw_version = payload.get("schema_version")
    try:
        return int(raw_version)
    except (TypeError, ValueError):
        return None


def load_checkpoint_payload(path: str | Path) -> Any:
    """Charge le payload pickled d'un checkpoint."""

    with Path(path).open("rb") as file_obj:
        return pickle.load(file_obj)


def inspect_muzero_checkpoint(
    path: str | Path,
    *,
    expected_context: dict[str, Any],
) -> tuple[Any | None, dict[str, Any]]:
    """Inspecte un checkpoint MuZero et retourne sa compatibilite detaillee."""

    checkpoint_path = Path(path)
    expected_fingerprint = dict(expected_context.get("config_fingerprint") or {})
    compatibility = {
        "allowed": False,
        "status": "missing",
        "reason": "Checkpoint MuZero introuvable.",
        "schema_version": None,
        "expected_fingerprint": expected_fingerprint,
        "artifact_fingerprint": None,
        "source_path": str(checkpoint_path),
    }
    if not checkpoint_path.exists():
        return None, compatibility

    try:
        payload = load_checkpoint_payload(checkpoint_path)
    except Exception as exc:
        compatibility.update(
            {
                "status": "error",
                "reason": f"Lecture du checkpoint impossible: {exc}",
            }
        )
        return None, compatibility

    schema_version = extract_checkpoint_schema_version(payload)
    compatibility["schema_version"] = schema_version

    if schema_version == CHECKPOINT_SCHEMA_VERSION:
        config_snapshot = dict(payload.get("config_snapshot") or {})
        if not config_snapshot:
            compatibility.update(
                {
                    "status": "incompatible",
                    "reason": "Checkpoint MuZero v2 invalide: config_snapshot absent.",
                }
            )
            return payload, compatibility

        # Inférence dynamique de use_jepa_encoder pour la rétrocompatibilité
        if "use_jepa_encoder" not in config_snapshot:
            params = payload.get("params")
            has_jepa = False
            def _has_jepa_weights(tree: Any) -> bool:
                if isinstance(tree, dict):
                    for k, v in tree.items():
                        if "context_encoder" in str(k):
                            return True
                        if _has_jepa_weights(v):
                            return True
                elif isinstance(tree, (list, tuple)):
                    for item in tree:
                        if _has_jepa_weights(item):
                            return True
                return False
            has_jepa = _has_jepa_weights(params)
            config_snapshot["use_jepa_encoder"] = has_jepa
            if has_jepa and "jepa_latent_size" not in config_snapshot:
                config_snapshot["jepa_latent_size"] = 128

        artifact_fingerprint = build_muzero_config_fingerprint(config_snapshot)
        compatibility["artifact_fingerprint"] = artifact_fingerprint

        expected_engine = str(expected_context.get("engine") or "muzero").strip().lower()
        artifact_engine = str(payload.get("engine") or "").strip().lower()
        if artifact_engine and artifact_engine != expected_engine:
            compatibility.update(
                {
                    "status": "incompatible",
                    "reason": (
                        "Checkpoint MuZero incompatible: moteur attendu="
                        f"{expected_engine}, obtenu={artifact_engine}."
                    ),
                }
            )
            return payload, compatibility

        expected_horizon = str(expected_context.get("horizon") or "").strip().lower()
        artifact_horizon = str(payload.get("horizon") or "").strip().lower()
        if expected_horizon and artifact_horizon and artifact_horizon != expected_horizon:
            compatibility.update(
                {
                    "status": "incompatible",
                    "reason": (
                        "Checkpoint MuZero incompatible: horizon attendu="
                        f"{expected_horizon}, obtenu={artifact_horizon}."
                    ),
                }
            )
            return payload, compatibility

        if artifact_fingerprint.get("sha256") != expected_fingerprint.get("sha256"):
            compatibility.update(
                {
                    "status": "incompatible",
                    "reason": "Checkpoint MuZero incompatible: empreinte de configuration differente.",
                }
            )
            return payload, compatibility

        params = payload.get("params")
        if params is None:
            compatibility.update(
                {
                    "status": "incompatible",
                    "reason": "Checkpoint MuZero v2 invalide: params absents.",
                }
            )
            return payload, compatibility

        artifact_signature = _flatten_tree_signature(params)
        shapes_match, reason = _compare_tree_signatures(
            expected_signature=list(expected_context.get("param_signature") or []),
            artifact_signature=artifact_signature,
        )
        compatibility["artifact_signature"] = artifact_signature
        if not shapes_match:
            compatibility.update({"status": "incompatible", "reason": reason})
            return payload, compatibility

        compatibility.update(
            {
                "allowed": True,
                "status": "compatible",
                "reason": "Checkpoint MuZero v2 compatible.",
            }
        )
        return payload, compatibility

    if not isinstance(payload, dict) or payload.get("params") is None:
        compatibility.update(
            {
                "status": "incompatible",
                "reason": "Checkpoint MuZero legacy invalide: format de payload inconnu.",
            }
        )
        return payload, compatibility

    legacy_signature = _flatten_tree_signature(payload.get("params"))
    compatibility["artifact_signature"] = legacy_signature
    shapes_match, reason = _compare_tree_signatures(
        expected_signature=list(expected_context.get("param_signature") or []),
        artifact_signature=legacy_signature,
    )
    if not shapes_match:
        compatibility.update({"status": "incompatible", "reason": reason})
        return payload, compatibility

    compatibility.update(
        {
            "allowed": True,
            "status": "legacy_compatible",
            "reason": "Checkpoint MuZero legacy compatible avec l'architecture courante.",
            "artifact_fingerprint": {
                "format": "legacy",
                "leaf_count": len(legacy_signature),
            },
        }
    )
    return payload, compatibility


def save_muzero_checkpoint(
    path: str | Path,
    *,
    config: Any,
    params: Any,
    opt_state: Any,
    training_step_count: int | None = None,
    artifact_kind: str = "checkpoint",
    lineage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialise un checkpoint MuZero v2 sur disque."""

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_muzero_checkpoint_payload(
        config=config,
        params=params,
        opt_state=opt_state,
        training_step_count=training_step_count,
        artifact_kind=artifact_kind,
        lineage=lineage,
    )
    with checkpoint_path.open("wb") as file_obj:
        pickle.dump(payload, file_obj, protocol=pickle.HIGHEST_PROTOCOL)
    return payload


def archive_muzero_artifacts(
    *,
    archive_root: str | Path,
    paths: list[str | Path],
    reason: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Archive un ou plusieurs artefacts MuZero dans un dossier horodate."""

    normalized_paths = [
        Path(path)
        for path in paths
        if str(path or "").strip()
    ]
    existing_paths = [path for path in normalized_paths if path.exists()]
    archive_base = Path(archive_root)
    archive_dir = archive_base / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_sanitize_token(reason, 'reason')}"
    archive_dir.mkdir(parents=True, exist_ok=True)

    archived_paths: list[dict[str, Any]] = []
    for source_path in existing_paths:
        target_path = archive_dir / source_path.name
        suffix_index = 1
        while target_path.exists():
            target_path = archive_dir / f"{source_path.stem}_{suffix_index}{source_path.suffix}"
            suffix_index += 1
        shutil.move(str(source_path), str(target_path))
        archived_paths.append(
            {
                "source_path": str(source_path),
                "archived_path": str(target_path),
            }
        )

    report = {
        "status": "archived" if archived_paths else "noop",
        "reason": str(reason or "").strip() or "archive",
        "archive_dir": str(archive_dir),
        "artifacts": archived_paths,
        "archived_at": _now_iso(),
    }
    metadata_path = archive_dir / "metadata.json"
    metadata_payload = {**report, "metadata": dict(metadata or {})}
    metadata_path.write_text(
        json.dumps(metadata_payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    report["metadata_path"] = str(metadata_path)
    return report


def _to_float(value: Any, default: float = 0.0) -> float:
    """Convertit une valeur libre en flottant robuste.

    Args:
        value (Any): Valeur brute a convertir.
        default (float): Valeur de repli si la conversion echoue.

    Returns:
        float: Valeur numerique exploitable.
    """

    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _extract_seed_metrics(candidate: dict[str, Any]) -> dict[str, Any]:
    """Extrait un bloc de metriques seed depuis une structure heterogene.

    Args:
        candidate (dict[str, Any]): Structure candidate provenant d'un screen,
            d'un rapport Arena ou d'un manifeste de relance.

    Returns:
        dict[str, Any]: Bloc de metriques le plus pertinent trouve.
    """

    direct_metrics = dict(candidate.get("metrics") or {})
    if direct_metrics:
        return direct_metrics
    battle_report = dict(candidate.get("battle_report") or {})
    challenger = dict(battle_report.get("challenger") or {})
    nested_metrics = dict(challenger.get("metrics") or {})
    if nested_metrics:
        return nested_metrics
    return {}


def _score_seed_for_v66(metrics: dict[str, Any]) -> tuple[float, str]:
    """Construit un score d'apprentissage V6.6 pour un seed MuZero.

    Args:
        metrics (dict[str, Any]): Metriques de training ou Arena associees
            au checkpoint.

    Returns:
        tuple[float, str]: Score plus eleve = meilleur seed probable et
            raison compacte de la decision.
    """

    loss_pol = _to_float(metrics.get("loss_pol"), default=9.99)
    loss_pol_per_head = _to_float(metrics.get("loss_pol_per_head"), default=loss_pol)
    root_mask_rate = _to_float(metrics.get("root_mask_rate"), default=1.0)
    policy_top1_share = _to_float(metrics.get("policy_top1_share"), default=0.0)
    close_quality_score = _to_float(metrics.get("close_quality_score"), default=0.0)
    split_opportunity_count = _to_float(metrics.get("split_opportunity_count"), default=0.0)
    pyramid_opportunity_count = _to_float(metrics.get("pyramid_opportunity_count"), default=0.0)
    directional_imbalance = _to_float(metrics.get("directional_imbalance"), default=1.0)

    mechanics_readiness = (
        min(split_opportunity_count, 6.0) * 0.20
        + min(pyramid_opportunity_count, 6.0) * 0.20
    )
    score = (
        (policy_top1_share * 10.0)
        + (close_quality_score * 6.0)
        + mechanics_readiness
        - (loss_pol_per_head * 6.5)
        - (root_mask_rate * 18.0)
        - (directional_imbalance * 4.0)
    )
    reason = (
        "score_seed_v66="
        f"top1={policy_top1_share:.2f}, close_q={close_quality_score:.2f}, "
        f"loss={loss_pol:.2f}, loss_head={loss_pol_per_head:.2f}, root_mask={root_mask_rate:.3f}, "
        f"split_opp={split_opportunity_count:.0f}, pyramid_opp={pyramid_opportunity_count:.0f}, "
        f"dir_imb={directional_imbalance:.3f}"
    )
    return score, reason


def recommend_muzero_seed_for_v66(
    *,
    weights_dir: str | Path,
    horizon: str,
    candidate_reports: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Recommande un seed V6.6 oriente apprentissage plutot que score court.

    Args:
        weights_dir (str | Path): Dossier local des checkpoints MuZero.
        horizon (str): Horizon concerne, par exemple ``scalp``.
        candidate_reports (list[dict[str, Any]] | None): Rapports candidats
            optionnels deja charges par l'orchestrateur.

    Returns:
        dict[str, Any]: Charge utile contenant le chemin recommande et la
            raison associee.
    """

    normalized_horizon = str(horizon or "scalp").strip().lower() or "scalp"
    weights_root = Path(weights_dir)
    ranked_candidates: list[dict[str, Any]] = []
    for raw_candidate in list(candidate_reports or []):
        candidate = dict(raw_candidate or {})
        checkpoint_step = int(candidate.get("checkpoint_step") or 0)
        checkpoint_path = str(candidate.get("checkpoint_path") or "").strip()
        if not checkpoint_path and checkpoint_step > 0:
            checkpoint_path = str(
                weights_root / f"muzero_{normalized_horizon}_ckpt_{checkpoint_step}.pkl"
            )
        path_obj = Path(checkpoint_path) if checkpoint_path else None
        if path_obj is None or not path_obj.exists():
            continue
        metrics = _extract_seed_metrics(candidate)
        if not metrics:
            continue
        score, reason = _score_seed_for_v66(metrics)
        ranked_candidates.append(
            {
                "checkpoint_path": str(path_obj),
                "checkpoint_step": checkpoint_step,
                "seed_score": score,
                "seed_selection_reason": reason,
            }
        )

    if ranked_candidates:
        ranked_candidates.sort(
            key=lambda item: (
                _to_float(item.get("seed_score"), default=-999.0),
                int(item.get("checkpoint_step") or 0),
            ),
            reverse=True,
        )
        selected = dict(ranked_candidates[0])
        selected["seed_source"] = "metrics"
        return {
            "recommended_seed_for_v66": selected["checkpoint_path"],
            "seed_selection_reason": str(selected["seed_selection_reason"]),
            "seed_score": float(selected["seed_score"]),
            "seed_source": "metrics",
            "candidates_considered": len(ranked_candidates),
        }

    fallback_steps = (23000, 6000, 10500)
    for fallback_step in fallback_steps:
        fallback_path = weights_root / f"muzero_{normalized_horizon}_ckpt_{int(fallback_step)}.pkl"
        if fallback_path.exists():
            return {
                "recommended_seed_for_v66": str(fallback_path),
                "seed_selection_reason": f"repli_v66_sur_ckpt{int(fallback_step)}",
                "seed_score": None,
                "seed_source": "fallback",
                "candidates_considered": len(ranked_candidates),
            }

    available_checkpoints = sorted(
        weights_root.glob(f"muzero_{normalized_horizon}_ckpt_*.pkl"),
        key=lambda path: int(str(path.stem).split("_ckpt_")[-1] or 0),
        reverse=True,
    )
    if available_checkpoints:
        return {
            "recommended_seed_for_v66": str(available_checkpoints[0]),
            "seed_selection_reason": "repli_v66_sur_checkpoint_disponible_le_plus_recent",
            "seed_score": None,
            "seed_source": "latest_available",
            "candidates_considered": len(ranked_candidates),
        }

    return {
        "recommended_seed_for_v66": None,
        "seed_selection_reason": "aucun_checkpoint_seed_disponible_pour_v66",
        "seed_score": None,
        "seed_source": "missing",
        "candidates_considered": len(ranked_candidates),
    }
