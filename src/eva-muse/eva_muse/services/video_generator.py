"""
Video Generator — Generates short animated video clips via AnimateDiff + VHS nodes in ComfyUI.
Optionally applies ReActor FaceSwap on every frame.

Requires ComfyUI custom nodes:
- AnimateDiff-Evolved: https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved
- ComfyUI-VideoHelperSuite (VHS): https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite
- comfyui-reactor-node (for face swap): https://github.com/Gourieff/comfyui-reactor-node
"""
import logging
import random
import asyncio
from eva_muse.services.comfy_client import ComfyUIClient

logger = logging.getLogger(__name__)

# AnimateDiff motion module - placed at /mnt/data/comfyui/models/animatediff_models/
MOTION_MODULE = "mm_sd_v15_v2.ckpt"


async def generate_video_clip(
    prompt: str,
    negative_prompt: str = "ugly, disfigured, low quality, bad anatomy, watermark",
    width: int = 512,
    height: int = 768,
    num_frames: int = 16,         # 16 frames = ~1s at 16fps
    fps: int = 8,
    steps: int = 20,
    cfg: float = 7.0,
    influencer_face: str | None = None,   # filename in ComfyUI input/ (e.g. "lois_face.jpg")
    checkpoint: str = "v1-5-pruned-emaonly.safetensors",
    loras: list[dict] | None = None,
) -> bytes | None:
    """
    Generates a short MP4 video clip using AnimateDiff.
    Returns raw video bytes or None on failure.
    """
    client = ComfyUIClient()
    seed = random.randint(1, 999999)

    # Base workflow with AnimateDiff loader
    workflow: dict = {
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": checkpoint}
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"batch_size": num_frames, "height": height, "width": width}
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["4", 1], "text": prompt}
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["4", 1], "text": negative_prompt}
        },
        # AnimateDiff: loads the motion module
        "12": {
            "class_type": "ADE_AnimateDiffLoaderWithContext",
            "inputs": {
                "model": ["4", 0],
                "motion_module": MOTION_MODULE,
                "beta_schedule": "sqrt_linear (AnimateDiff)",
                "context_options": None,
                "motion_lora": None,
                "ad_settings": None,
            }
        },
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "cfg": cfg,
                "denoise": 1,
                "latent_image": ["5", 0],
                "model": ["12", 0],   # Use AnimateDiff output model
                "negative": ["7", 0],
                "positive": ["6", 0],
                "sampler_name": "euler",
                "scheduler": "normal",
                "seed": seed,
                "steps": steps
            }
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": ["4", 2]}
        },
        # VHS Video Combine — outputs as MP4
        "20": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["8", 0],
                "frame_rate": fps,
                "loop_count": 0,
                "filename_prefix": "Hive_Video",
                "format": "video/h264-mp4",
                "pingpong": False,
                "save_output": True,
                "crf": 19,
            }
        }
    }

    # Inject LoRAs if provided
    if loras:
        from eva_muse.services.lora_manager import inject_loras_into_workflow
        workflow = inject_loras_into_workflow(workflow, loras, model_node_id="4")

    # Inject ReActor FaceSwap on decoded frames if requested
    if influencer_face:
        workflow["30"] = {
            "class_type": "LoadImage",
            "inputs": {"image": influencer_face}
        }
        workflow["31"] = {
            "class_type": "ReActorFaceSwap",
            "inputs": {
                "enabled": True,
                "input_image": ["8", 0],
                "source_image": ["30", 0],
                "swap_model": "inswapper_128.onnx",
                "facedetection": "retinaface_resnet50",
                "face_restore_model": "none",
                "face_restore_visibility": 1.0,
                "codeformer_weight": 0.5
            }
        }
        # Redirect VHS input to the face-swapped output
        workflow["20"]["inputs"]["images"] = ["31", 0]

    try:
        logger.info(f"Generating video clip: {num_frames} frames, {width}x{height}")
        # ComfyUI returns the video file via the /history endpoint after generation
        video_bytes = await client.generate_video_from_workflow(workflow)
        return video_bytes
    except Exception as e:
        logger.error(f"Video generation failed: {e}")
        return None
