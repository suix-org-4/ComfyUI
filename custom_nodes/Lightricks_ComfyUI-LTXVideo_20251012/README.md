# ComfyUI-LTXVideo

[![GitHub](https://img.shields.io/badge/LTXV-Repo-blue?logo=github)](https://github.com/Lightricks/LTX-Video)
[![Website](https://img.shields.io/badge/Website-LTXV-181717?logo=google-chrome)](https://www.lightricks.com/ltxv)
[![Model](https://img.shields.io/badge/HuggingFace-Model-orange?logo=huggingface)](https://huggingface.co/Lightricks/LTX-Video)
[![LTXV Trainer](https://img.shields.io/badge/LTXV-Trainer%20Repo-9146FF)](https://github.com/Lightricks/LTX-Video-Trainer)
[![Demo](https://img.shields.io/badge/Demo-Try%20Now-brightgreen?logo=vercel)](https://app.ltx.studio/ltx-video)
[![Paper](https://img.shields.io/badge/Paper-arXiv-B31B1B?logo=arxiv)](https://arxiv.org/abs/2501.00103)
[![Discord](https://img.shields.io/badge/Join-Discord-5865F2?logo=discord)](https://discord.gg/Mn8BRgUKKy)

ComfyUI-LTXVideo is a collection of custom nodes for ComfyUI, designed to provide useful tools for working with the LTXV model.
The model itself is supported in the core ComfyUI [code](https://github.com/comfyanonymous/ComfyUI/tree/master/comfy/ldm/lightricks).
The main LTXVideo repository can be found [here](https://github.com/Lightricks/LTX-Video).



# ⭐ 16.07.2025 - LTXV 0.9.8 Release ⭐

### 🚀 What's New
1. **LTXV 0.9.8 Model**<br>
   The new model and its distilled variants offer improved prompt understanding and detail generation<br>
   👉 [13B Distilled model](https://huggingface.co/Lightricks/LTX-Video/blob/main/ltxv-13b-0.9.8-distilled.safetensors)<br>
   👉 [13B Distilled model 8-bit](https://huggingface.co/Lightricks/LTX-Video/blob/main/ltxv-13b-0.9.8-distilled-fp8.safetensors)<br>
   👉 [2B from 13B Distilled model](https://huggingface.co/Lightricks/LTX-Video/blob/main/ltxv-2b-0.9.8-distilled.safetensors)<br>
   👉 [2B from 13B Distilled model 8-bit](https://huggingface.co/Lightricks/LTX-Video/blob/main/ltxv-2b-0.9.8-distilled-fp8.safetensors)<br>
   👉 [IC Lora Detailer](https://huggingface.co/Lightricks/LTX-Video-ICLoRA-detailer-13b-0.9.8/resolve/main/ltxv-098-ic-lora-detailer-comfyui.safetensors)<br>


2. **Autoregressive Generation**
   Introducing new ComfyUI nodes that enable virtually infinite video generation. The new **LTXV Looping Sampler** node allows generation of videos with arbitrary length and consistent motion. **ICLoRAs** are supported as well—by providing guidance from existing videos (e.g., depth, pose, or Canny edges), you can generate long videos in a video-to-video manner.
   👉 [Long Img2Video Generation Flow](#long-video-generation)
   👉 [Long Video2Video Generation Flow](#long-video-generation)

3. **Detailer ICLoRA**
   Introducing the **Detailer ICLoRA**, which enhances generated latents with fine details by applying a few additional diffusion steps. This results in significantly more detailed generations.
   👉 [Detailer ICLoRA Flow](#video-upscaling)

# ⭐ 8.07.2025 - LTXVideo ICLora Release  ⭐
### 🚀 What's New in LTXVideo ICLoRA

1. **Three New ICLoRA Models**
   Introducing powerful in-context LoRA models that enable precise control over video generation:
   - Depth Control: [LTX-Video-ICLoRA-depth-13b-0.9.7](https://huggingface.co/Lightricks/LTX-Video-ICLoRA-depth-13b-0.9.7)
   - Pose Control: [LTX-Video-ICLoRA-pose-13b-0.9.7](https://huggingface.co/Lightricks/LTX-Video-ICLoRA-pose-13b-0.9.7)
   - Edge/Canny Control: [LTX-Video-ICLoRA-canny-13b-0.9.7](https://huggingface.co/Lightricks/LTX-Video-ICLoRA-canny-13b-0.9.7)

2. **New Node: 🅛🅣🅧 LTXV In Context Sampler**
   A dedicated node for seamlessly integrating ICLoRA models into your workflow, enabling fine-grained control over video generation using depth maps, pose estimation, or edge detection.

3. **Example Workflow**
   Check out [example workflow](#iclora) for a complete example demonstrating how to use the ICLoRA models effectively.

4. **Custom ICLoRA Training**
   We've released a trainer that allows you to create your own specialized ICLoRA models for custom control signals. Check out the [trainer repository](https://github.com/Lightricks/LTX-Video-Trainer) to get started.


# ⭐ 9.06.2025 – LTXVideo VAE Patcher, Mask manipulation and Q8 LoRA loader nodes.  ⭐
1. **LTXV Patcher VAE**<br> The new node improves VAE decoding performance by reducing runtime and cutting memory consumption by up to 50%. This allows generation of higher-resolution outputs on consumer-grade GPUs with limited VRAM, without needing to load the VAE partially or decode in tiles.<br>
⚠️ On *Windows*, you may need to add the paths to the *MSVC compiler (cl.exe)* and *ninja.exe* to your system environment PATH variable. <br>
2. **LTXV Preprocess Masks**<br>
Preprocesses masks for use with the LTXVideo model's latent masking. It validates mask dimensions based on VAE downscaling, supports optional inversion, handles the first frame mask separately, combines temporal masks via max pooling, applies morphological operations to grow or shrink masks, and clamps values to ensure correct opacity. The result is a set of masks ready for latent-space masking.
3. **LTXV Q8 Lora Model Loader**<br>
Applying LoRA to an FP8-quantized model requires special handling to preserve output quality. It's crucial to apply LoRA weights using the correct precision, as the current LoRA implementation in ComfyUI does so in a non-optimal way. This node addresses that limitation by ensuring LoRA weights are applied properly, resulting in significantly better quality. If you're working with an FP8 LTXV model, using this node guarantees that LoRA behaves as expected and delivers the intended effect.


# ⭐ 14.05.2025 – LTXVideo 13B 0.9.7 Distilled Release ⭐

### 🚀 What's New in LTXVideo 13B 0.9.7 Distilled
1. **LTXV 13B Distilled 🥳 0.9.7**<br>
   Delivers cinematic-quality videos at fraction of steps needed to run full model. Only 4 or 8 steps needed for single generation.<br>
   👉 [Download here](https://huggingface.co/Lightricks/LTX-Video/blob/main/ltxv-13b-0.9.7-distilled.safetensors)

2. **LTXV 13B Distilled Quantized 0.9.7**<br>
   Offers reduced memory requirements and even faster inference speeds.
   Ideal for consumer-grade GPUs (e.g., NVIDIA 4090, 5090).<br>
   ***Important:*** In order to get the best performance with the quantized version please install [q8_kernels](https://github.com/Lightricks/LTXVideo-Q8-Kernels) package and use dedicated flow below. <br>
   👉 [Download here](https://huggingface.co/Lightricks/LTX-Video/blob/main/ltxv-13b-0.9.7-distilled-fp8.safetensors)<br>
   🧩 Example ComfyUI flow available in the [Example Workflows](#example-workflows) section.

3. **Updated LTV 13B Quantized version**<br>
From now on all our 8bit quantized models are running natively in ComfyUI, still with our Q8 patcher node you will get the best inference speed.<br>
👉 [Download here](https://huggingface.co/Lightricks/LTX-Video/blob/main/ltxv-13b-0.9.7-dev-fp8.safetensors)<br>
# ⭐ 06.05.2025 – LTXVideo 13B 0.9.7 Release ⭐

### 🚀 What's New in LTXVideo 13B 0.9.7

1. **LTXV 13B 0.9.7**
   Delivers cinematic-quality videos at unprecedented speed.<br>
   👉 [Download here](https://huggingface.co/Lightricks/LTX-Video/blob/main/ltxv-13b-0.9.7-dev.safetensors)

2. **LTXV 13B Quantized 0.9.7**
   Offers reduced memory requirements and even faster inference speeds.
   Ideal for consumer-grade GPUs (e.g., NVIDIA 4090, 5090).
   Delivers outstanding quality with improved performance.<br>
   ***Important:*** In order to run the quantized version please install [LTXVideo-Q8-Kernels](https://github.com/Lightricks/LTXVideo-Q8-Kernels) package and use dedicated flow below. Loading the model in Comfy with LoadCheckpoint node won't work. <br>
   👉 [Download here](https://huggingface.co/Lightricks/LTX-Video/blob/main/ltxv-13b-0.9.7-dev-fp8.safetensors)<br>
   🧩 Example ComfyUI flow available in the [Example Workflows](#example-workflows) section.

3. **Latent Upscaling Models**
   Enables inference across multiple scales by upscaling latent tensors without decoding/encoding.
   Multiscale inference delivers high-quality results in a fraction of the time compared to similar models.<br>
   ***Important:*** Make sure you put the models below in **models/upscale_models** folder.<br>
   👉 Spatial upscaling: [Download here](https://huggingface.co/Lightricks/LTX-Video/blob/main/ltxv-spatial-upscaler-0.9.7.safetensors).<br>
   👉 Temporal upscaling: [Download here](https://huggingface.co/Lightricks/LTX-Video/blob/main/ltxv-temporal-upscaler-0.9.7.safetensors).<br>
   🧩 Example ComfyUI flow available in the [Example Workflows](#example-workflows) section.


### Technical Updates

1. ***New simplified flows and nodes***<br>
1.1. Simplified image to video: [Download here](example_workflows/ltxv-13b-i2v-base.json).<br>
1.2. Simplified image to video with extension: [Download here](example_workflows/ltxv-13b-i2v-extend.json).<br>
1.3. Simplified image to video with keyframes: [Download here](example_workflows/ltxv-13b-i2v-keyframes.json).<br>

# 17.04.2025 ⭐ LTXVideo 0.9.6 Release ⭐

### LTXVideo 0.9.6 introduces:

1. LTXV 0.9.6 – higher quality, faster, great for final output. Download from [here](https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltxv-2b-0.9.6-dev-04-25.safetensors).
2. LTXV 0.9.6 Distilled – our fastest model yet (only 8 steps for generation), lighter, great for rapid iteration. Download from [here](https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltxv-2b-0.9.6-distilled-04-25.safetensors).

### Technical Updates

We introduce the __STGGuiderAdvanced__ node, which applies different CFG and STG parameters at various diffusion steps. All flows have been updated to use this node and are designed to provide optimal parameters for the best quality.
See the [Example Workflows](#example-workflows) section.

# 5.03.2025 ⭐ LTXVideo 0.9.5 Release ⭐

### LTXVideo 0.9.5 introduces:

1. Improved quality with reduced artifacts.
2. Support for higher resolution and longer sequences.
3. Frame and sequence conditioning (beyond the first frame).
4. Enhanced prompt understanding.
5. Commercial license availability.

### Technical Updates

Since LTXVideo is now fully supported in the ComfyUI core, we have removed the custom model implementation. Instead, we provide updated workflows to showcase the new features:

1. **Frame Conditioning** – Enables interpolation between given frames.
2. **Sequence Conditioning** – Allows motion interpolation from a given frame sequence, enabling video extension from the beginning, end, or middle of the original video.
3. **Prompt Enhancer** – A new node that helps generate prompts optimized for the best model performance.
   See the [Example Workflows](#example-workflows) section for more details.

### LTXTricks Update

The LTXTricks code has been integrated into this repository (in the `/tricks` folder) and will be maintained here. The original [repo](https://github.com/logtd/ComfyUI-LTXTricks) is no longer maintained, but all existing workflows should continue to function as expected.

## 22.12.2024

Fixed a bug which caused the model to produce artifacts on short negative prompts when using a native CLIP Loader node.

## 19.12.2024 ⭐ Update ⭐

1. Improved model - removes "strobing texture" artifacts and generates better motion. Download from [here](https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltx-video-2b-v0.9.1.safetensors).
2. STG support
3. Integrated image degradation system for improved motion generation.
4. Additional initial latent optional input to chain latents for high res generation.
5. Image captioning in image to video [flow](example_workflows/ltxvideo-i2v.json).

## Installation

Installation via [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager) is preferred. Simply search for `ComfyUI-LTXVideo` in the list of nodes and follow installation instructions.

### Manual installation

1. Install ComfyUI
2. Clone this repository to `custom-nodes` folder in your ComfyUI installation directory.
3. Install the required packages:

```bash
cd custom_nodes/ComfyUI-LTXVideo && pip install -r requirements.txt
```

For portable ComfyUI installations, run

```
.\python_embeded\python.exe -m pip install -r .\ComfyUI\custom_nodes\ComfyUI-LTXVideo\requirements.txt
```

### Models

1. Download [ltx-video-2b-v0.9.1.safetensors](https://huggingface.co/Lightricks/LTX-Video/blob/main/ltx-video-2b-v0.9.1.safetensors) from Hugging Face and place it under `models/checkpoints`.
2. Install one of the t5 text encoders, for example [google_t5-v1_1-xxl_encoderonly](https://huggingface.co/mcmonkey/google_t5-v1_1-xxl_encoderonly/tree/main). You can install it using ComfyUI Model Manager.

## Example workflows

Note that to run the example workflows, you need to have some additional custom nodes, like [ComfyUI-VideoHelperSuite](https://github.com/kosinkadink/ComfyUI-VideoHelperSuite) and others, installed. You can do it by pressing "Install Missing Custom Nodes" button in ComfyUI Manager.

### Long Video Generation
🧩 [Image to Video Long Video](example_workflows/ltxv-13b-i2v-long-multi-prompt.json): Long video generation with support for multiple prompts along the video duration.
🧩 [Video to Video Long Video](example_workflows/ltxv-13b-v2v-long-depth.json): Long video-to-video generation. Given a guiding video—such as depth, pose, or edges—the flow generates a new video.

### Video Upscaling
🧩 [Video Upscaling](example_workflows/ltxv-13b-upscale.json): Upscales and adds fine details to any given video, increasing its spatial resolution by 2×.

### Easy to use multi scale generation workflows

🧩 [Image to video mixed](example_workflows/ltxv13b-i2v-mixed-multiscale.json): mixed flow with full and distilled model for best quality and speed trade-off.<br>

### 13B model<br>
🧩 [Image to video](example_workflows/ltxv-13b-i2v-base.json)<br>
🧩 [Image to video with keyframes](example_workflows/ltxv-13b-i2v-keyframes.json)<br>
🧩 [Image to video with duration extension](example_workflows/ltxv-13b-i2v-extend.json)<br>
🧩 [Image to video 8b quantized](example_workflows/ltxv-13b-i2v-base-fp8.json)

### 13B distilled model<br>
🧩 [Image to video](example_workflows/13b-distilled/ltxv-13b-dist-i2v-base.json)<br>
🧩 [Image to video with keyframes](example_workflows/13b-distilled/ltxv-13b-dist-i2v-keyframes.json)<br>
🧩 [Image to video with duration extension](example_workflows/13b-distilled/ltxv-13b-dist-i2v-extend.json)<br>
🧩 [Image to video 8b quantized](example_workflows/13b-distilled/ltxv-13b-dist-i2v-base-fp8.json)

### ICLora
🧩 [Download workflow](example_workflows/ic_lora/ic-lora.json)

### Inversion

#### Flow Edit

🧩 [Download workflow](example_workflows/tricks/ltxvideo-flow-edit.json)<br>
![workflow](example_workflows/tricks/ltxvideo-flow-edit.png)

#### RF Edit

🧩 [Download workflow](example_workflows/tricks/ltxvideo-rf-edit.json)<br>
![workflow](example_workflows/tricks/ltxvideo-rf-edit.png)
