"""
ChatterBox Official 23-Lang Engine Node - Official multilingual ChatterBox configuration
Provides ChatterBox Official 23-Lang engine adapter with multilingual parameters
"""

import os
import sys
import importlib.util
from typing import Dict, Any

# Add project root directory to path for imports
current_dir = os.path.dirname(__file__)
nodes_dir = os.path.dirname(current_dir)  # nodes/
project_root = os.path.dirname(nodes_dir)  # project root
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load base_node module directly
base_node_path = os.path.join(nodes_dir, "base", "base_node.py")
base_spec = importlib.util.spec_from_file_location("base_node_module", base_node_path)
base_module = importlib.util.module_from_spec(base_spec)
sys.modules["base_node_module"] = base_module
base_spec.loader.exec_module(base_module)

# Import the base class
BaseTTSNode = base_module.BaseTTSNode


class ChatterBoxOfficial23LangEngineNode(BaseTTSNode):
    """
    ChatterBox Official 23-Lang TTS Engine configuration node.
    Provides multilingual ChatterBox parameters and creates engine adapter for unified nodes.
    """
    
    @classmethod
    def NAME(cls):
        return "⚙️ ChatterBox Official 23-Lang Engine"
    
    @classmethod
    def INPUT_TYPES(cls):
        # Import language models for dropdown
        try:
            from engines.chatterbox_official_23lang.language_models import get_supported_language_names, SUPPORTED_LANGUAGES
            available_languages = get_supported_language_names()
        except ImportError:
            available_languages = ["English"]

        return {
            "required": {
                "model_version": (["v1", "v2"], {
                    "default": "v2",
                    "tooltip": "ChatterBox model version. v2 adds special tokens for emotions ([giggle], [laughter], [sigh]), sounds ([cough], [sneeze]), vocal styles ([singing], [whisper]), and improved Russian support."
                }),
                "language": (available_languages, {
                    "default": "English",
                    "tooltip": "ChatterBox language model to use for text-to-speech generation. Local models are preferred over remote downloads."
                }),
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "Device to run ChatterBox model on. 'auto' selects best available (GPU if available, otherwise CPU)."
                }),
                "exaggeration": ("FLOAT", {
                    "default": 0.5, 
                    "min": 0.25, 
                    "max": 2.0, 
                    "step": 0.05,
                    "tooltip": "Speech exaggeration level for ChatterBox. Higher values create more dramatic and expressive speech."
                }),
                "temperature": ("FLOAT", {
                    "default": 0.8, 
                    "min": 0.05, 
                    "max": 5.0, 
                    "step": 0.05,
                    "tooltip": "Controls randomness in ChatterBox generation. Higher values = more creative/varied speech, lower values = more consistent speech."
                }),
                "cfg_weight": ("FLOAT", {
                    "default": 0.5, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.05,
                    "tooltip": "Classifier-Free Guidance weight for ChatterBox. Controls how strongly the model follows the text prompt."
                }),
                "repetition_penalty": ("FLOAT", {
                    "default": 2.0,
                    "min": 1.0,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Penalty for repeated tokens. Higher values reduce repetition in generated speech."
                }),
                "min_p": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Minimum probability threshold for token selection. Lower values allow more diverse tokens."
                }),
                "top_p": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Nucleus sampling threshold. Controls the probability mass of tokens to consider."
                }),
            }
        }

    RETURN_TYPES = ("TTS_ENGINE",)
    RETURN_NAMES = ("TTS_engine",)
    FUNCTION = "create_engine_adapter"
    CATEGORY = "TTS Audio Suite/⚙️ Engines"

    def create_engine_adapter(self, model_version: str, language: str, device: str, exaggeration: float,
                            temperature: float, cfg_weight: float, repetition_penalty: float,
                            min_p: float, top_p: float):
        """
        Create ChatterBox Official 23-Lang engine adapter with configuration.

        Args:
            model_version: Model version (v1 or v2)
            language: Language for multilingual generation
            device: Device to run model on
            exaggeration: Speech exaggeration level
            temperature: Generation randomness
            cfg_weight: Classifier-Free Guidance weight
            repetition_penalty: Penalty for repeated tokens
            min_p: Minimum probability threshold
            top_p: Nucleus sampling threshold

        Returns:
            Tuple containing ChatterBox Official 23-Lang engine adapter
        """
        try:
            # Import the adapter class
            from engines.adapters.chatterbox_official_23lang_adapter import ChatterBoxOfficial23LangEngineAdapter

            # Create configuration dictionary
            config = {
                "model_version": model_version,
                "language": language,
                "device": device,
                "exaggeration": exaggeration,
                "temperature": temperature,
                "cfg_weight": cfg_weight,
                "repetition_penalty": repetition_penalty,
                "min_p": min_p,
                "top_p": top_p,
                "engine_type": "chatterbox_official_23lang"
            }
            
            print(f"⚙️ ChatterBox Official 23-Lang {model_version}: Configured for {language} on {device}")
            print(f"   Settings: exaggeration={exaggeration}, temperature={temperature}, cfg_weight={cfg_weight}")
            print(f"   Advanced: repetition_penalty={repetition_penalty}, min_p={min_p}, top_p={top_p}")
            
            # For now, return the config dict. The actual adapter creation will happen 
            # in the consumer nodes when they have access to the node instance
            engine_data = {
                "engine_type": "chatterbox_official_23lang", 
                "config": config,
                "adapter_class": "ChatterBoxOfficial23LangEngineAdapter"
            }
            
            return (engine_data,)
            
        except Exception as e:
            print(f"❌ ChatterBox Engine error: {e}")
            import traceback
            traceback.print_exc()
            
            # Return a default config that indicates error state
            error_config = {
                "engine_type": "chatterbox_official_23lang",
                "config": {
                    "language": language,
                    "device": "cpu",  # Fallback to CPU
                    "exaggeration": 0.5,
                    "temperature": 0.8,
                    "cfg_weight": 0.5,
                    "repetition_penalty": 2.0,
                    "min_p": 0.05,
                    "top_p": 1.0,
                    "error": str(e)
                },
                "adapter_class": "ChatterBoxOfficial23LangEngineAdapter"
            }
            return (error_config,)


# Register the node class
NODE_CLASS_MAPPINGS = {
    "ChatterBoxOfficial23LangEngineNode": ChatterBoxOfficial23LangEngineNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ChatterBoxOfficial23LangEngineNode": "⚙️ ChatterBox Official 23-Lang Engine"
}