import torch
import torch.nn as nn
from accelerate import init_empty_weights
from comfy.ops import cast_bias_weight

#based on https://github.com/huggingface/diffusers/blob/main/src/diffusers/quantizers/gguf/utils.py
def _replace_linear(model, compute_dtype, state_dict, prefix="", patches=None, scale_weights=None):
   
    has_children = list(model.children())
    if not has_children:
        return
    for name, module in model.named_children():
        module_prefix = prefix + name + "."
        _replace_linear(module, compute_dtype, state_dict, module_prefix, patches, scale_weights)

        if isinstance(module, nn.Linear) and "loras" not in module_prefix:
            in_features = state_dict[module_prefix + "weight"].shape[1]
            out_features = state_dict[module_prefix + "weight"].shape[0]
            if scale_weights is not None:
                scale_key = f"{module_prefix}scale_weight"

            with init_empty_weights():
                model._modules[name] = CustomLinear(
                    in_features,
                    out_features,
                    module.bias is not None,
                    compute_dtype=compute_dtype,
                    scale_weight=scale_weights.get(scale_key) if scale_weights else None
                )
            #set_lora_params(model._modules[name], patches, module_prefix)
            model._modules[name].source_cls = type(module)
            # Force requires_grad to False to avoid unexpected errors
            model._modules[name].requires_grad_(False)

    return model

def set_lora_params(module, patches, module_prefix=""):
    remove_lora_from_module(module)
    # Recursively set lora_diffs and lora_strengths for all CustomLinear layers
    for name, child in module.named_children():
        child_prefix = (f"{module_prefix}{name}.")
        set_lora_params(child, patches, child_prefix)
    if isinstance(module, CustomLinear):
        key = f"diffusion_model.{module_prefix}weight"
        patch = patches.get(key, [])
        #print(f"Processing LoRA patches for {key}: {len(patch)} patches found")
        if len(patch) == 0:
            key = key.replace("_orig_mod.", "")
            patch = patches.get(key, [])
            #print(f"Processing LoRA patches for {key}: {len(patch)} patches found")
        if len(patch) != 0:
            lora_diffs = []
            for p in patch:
                lora_obj = p[1]
                if "head" in key:
                    continue  # For now skip LoRA for head layers
                elif hasattr(lora_obj, "weights"):
                    lora_diffs.append(lora_obj.weights)
                elif isinstance(lora_obj, tuple) and lora_obj[0] == "diff":
                    lora_diffs.append(lora_obj[1])
                else:
                    continue
            lora_strengths = [p[0] for p in patch]
            module.lora = (lora_diffs, lora_strengths)
            module.step = 0  # Initialize step for LoRA scheduling


class CustomLinear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=False,
        compute_dtype=None,
        device=None,
        scale_weight=None
    ) -> None:
        super().__init__(in_features, out_features, bias, device)
        self.compute_dtype = compute_dtype
        self.lora = None
        self.step = 0
        self.scale_weight = scale_weight
        self.bias_function = []
        self.weight_function = []

    def forward(self, input):
        weight, bias = cast_bias_weight(self, input)

        if self.scale_weight is not None:
            if weight.numel() < input.numel():
                weight = weight * self.scale_weight
            else:
                input = input * self.scale_weight

        if self.lora is not None:
            weight = self.apply_lora(weight).to(self.compute_dtype)

        return torch.nn.functional.linear(input, weight, bias)

    @torch.compiler.disable()
    def apply_lora(self, weight):
        for lora_diff, lora_strength in zip(self.lora[0], self.lora[1]):
            if isinstance(lora_strength, list):
                lora_strength = lora_strength[self.step]
                if lora_strength == 0.0:
                    continue
            elif lora_strength == 0.0:
                continue
            patch_diff = torch.mm(
                lora_diff[0].flatten(start_dim=1).to(weight.device),
                lora_diff[1].flatten(start_dim=1).to(weight.device)
            ).reshape(weight.shape)
            alpha = lora_diff[2] / lora_diff[1].shape[0] if lora_diff[2] is not None else 1.0
            scale = lora_strength * alpha
            weight = weight.add(patch_diff, alpha=scale)
        return weight
    
def remove_lora_from_module(module):
    for name, submodule in module.named_modules():
        submodule.lora = None