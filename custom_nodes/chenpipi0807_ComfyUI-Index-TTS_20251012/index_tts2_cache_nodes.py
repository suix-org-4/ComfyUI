import json
from typing import Dict, Any


class IndexTTS2CacheControlNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Whether to keep models cached after a node call
                "keep_models_cached": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                # Reserved for future VRAM options (not wired yet)
                # "vram_mode": (["Balanced", "Low", "Performance"], {"default": "Balanced"}),
            },
        }

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("cache_control",)
    FUNCTION = "build"
    CATEGORY = "audio"

    def build(self, keep_models_cached: bool = False) -> Dict[str, Any]:
        ctrl = {
            "keep_cached": bool(keep_models_cached),
        }
        # Return as DICT for downstream nodes
        return (ctrl,)
