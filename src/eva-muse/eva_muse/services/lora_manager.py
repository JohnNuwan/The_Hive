"""
LoRA Manager — Reads the LoRA catalog and injects the correct LoRA nodes
into a ComfyUI workflow dict dynamically for a given niche.

The catalog is stored at infra/comfyui/loras_catalog.json.
LoRA files live at /mnt/data/comfyui/models/loras/ on Proxmox.
"""
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Path to the catalog on this machine (Docker mounts it at runtime)
CATALOG_PATH = os.getenv(
    "LORAS_CATALOG_PATH",
    "/mnt/data/comfyui/loras_catalog.json"
)

# Fallback catalog in case the file isn't mounted yet
FALLBACK_CATALOG: dict[str, dict] = {
    # mapping: filename -> { niches, strength, description }
    # NOTE: Add real filenames from /mnt/data/comfyui/models/loras/ after download
    "fitness_athletic_v2.safetensors": {
        "niches": ["fitness"],
        "strength": 0.75,
        "description": "Athletic physique enhancer"
    },
    "gfe_realistic_v3.safetensors": {
        "niches": ["girlfriend"],
        "strength": 0.8,
        "description": "Girlfriend experience realistic style"
    },
    "bdsm_dominant_lora.safetensors": {
        "niches": ["dominatrice"],
        "strength": 0.85,
        "description": "Dominant BDSM aesthetic"
    },
    "soft_submissive_v2.safetensors": {
        "niches": ["soumise"],
        "strength": 0.8,
        "description": "Soft submissive pastel aesthetic"
    },
    "foot_fetish_detail.safetensors": {
        "niches": ["pied"],
        "strength": 0.9,
        "description": "Foot detail enhancer"
    },
    "redhead_freckles_lora.safetensors": {
        "niches": ["rousse"],
        "strength": 0.85,
        "description": "Red hair and freckles"
    },
    "petite_cute_v1.safetensors": {
        "niches": ["petite"],
        "strength": 0.75,
        "description": "Petite and cute style"
    },
    "mature_elegant_v2.safetensors": {
        "niches": ["milf"],
        "strength": 0.8,
        "description": "Mature elegant woman"
    },
    "cosplay_detail_lora.safetensors": {
        "niches": ["cosplay"],
        "strength": 0.8,
        "description": "Cosplay costume detail"
    },
    "furry_anthro_v3.safetensors": {
        "niches": ["furry"],
        "strength": 0.9,
        "description": "Furry anthropomorphic art"
    },
}


def load_catalog() -> dict:
    """Loads the LoRA catalog from file or returns the fallback."""
    try:
        with open(CATALOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"LoRA catalog not found at {CATALOG_PATH}, using fallback.")
        return FALLBACK_CATALOG
    except Exception as e:
        logger.error(f"Error loading LoRA catalog: {e}")
        return FALLBACK_CATALOG


def get_loras_for_niche(niche_id: str) -> list[dict]:
    """Returns a list of {filename, strength} for the given niche."""
    catalog = load_catalog()
    result = []
    for filename, meta in catalog.items():
        if niche_id in meta.get("niches", []):
            result.append({
                "filename": filename,
                "strength": meta.get("strength", 0.8)
            })
    return result


def inject_loras_into_workflow(workflow: dict, loras: list[dict], model_node_id: str = "4") -> dict:
    """
    Dynamically chains LoRALoader nodes before the KSampler's model input.

    For each LoRA in the list, it inserts a LoRALoader node that reads from
    the previous model output (chaining them together).

    Args:
        workflow: the base ComfyUI workflow dict
        loras: list of {"filename": str, "strength": float}
        model_node_id: the node ID of the CheckpointLoaderSimple

    Returns:
        Updated workflow dict with LoRA nodes injected
    """
    if not loras:
        return workflow

    # The next free node key
    next_id = max(int(k) for k in workflow.keys()) + 1
    current_model_source = [model_node_id, 0]  # [node_id, output_slot]
    current_clip_source = [model_node_id, 1]

    for lora in loras:
        node_key = str(next_id)
        workflow[node_key] = {
            "class_type": "LoraLoader",
            "inputs": {
                "model": current_model_source,
                "clip": current_clip_source,
                "lora_name": lora["filename"],
                "strength_model": lora["strength"],
                "strength_clip": lora["strength"],
            }
        }
        current_model_source = [node_key, 0]
        current_clip_source = [node_key, 1]
        next_id += 1

    # Patch KSampler (node "3") to use the final LoRA chain output
    if "3" in workflow:
        workflow["3"]["inputs"]["model"] = current_model_source

    # Patch the CLIPTextEncode nodes (6, 7) to use the new CLIP source
    for clip_node_id in ["6", "7"]:
        if clip_node_id in workflow:
            workflow[clip_node_id]["inputs"]["clip"] = current_clip_source

    return workflow
