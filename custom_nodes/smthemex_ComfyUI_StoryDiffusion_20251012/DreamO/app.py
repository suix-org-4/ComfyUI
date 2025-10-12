# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# import argparse

import cv2
# import gradio as gr
import numpy as np
import torch
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision.transforms.functional import normalize

from .dreamo.dreamo_pipeline import DreamOPipeline
from .dreamo.utils import img2tensor, resize_numpy_image_area, tensor2img
from .tools import BEN2

# parser = argparse.ArgumentParser()
# parser.add_argument('--port', type=int, default=8080)
# args = parser.parse_args()


class Generator:
    def __init__(self):
        device = torch.device('cuda')
        # preprocessing models
        # background remove model: BEN2
        self.bg_rm_model = BEN2.BEN_Base().to(device).eval()
        hf_hub_download(repo_id='PramaLLC/BEN2', filename='BEN2_Base.pth', local_dir='models')
        self.bg_rm_model.loadcheckpoints('models/BEN2_Base.pth')
        # face crop and align tool: facexlib
        self.face_helper = FaceRestoreHelper(
            upscale_factor=1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            device=device,
        )

        # load dreamo
        model_root = 'black-forest-labs/FLUX.1-dev'
        dreamo_pipeline = DreamOPipeline.from_pretrained(model_root, torch_dtype=torch.bfloat16)
        dreamo_pipeline.load_dreamo_model(device, use_turbo=True)
        self.dreamo_pipeline = dreamo_pipeline.to(device)

    @torch.no_grad()
    def get_align_face(self, img):
        # the face preprocessing code is same as PuLID
        self.face_helper.clean_all()
        image_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.face_helper.read_image(image_bgr)
        self.face_helper.get_face_landmarks_5(only_center_face=True)
        self.face_helper.align_warp_face()
        if len(self.face_helper.cropped_faces) == 0:
            return None
        align_face = self.face_helper.cropped_faces[0]

        input = img2tensor(align_face, bgr2rgb=True).unsqueeze(0) / 255.0
        input = input.to(torch.device("cuda"))
        parsing_out = self.face_helper.face_parse(normalize(input, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))[0]
        parsing_out = parsing_out.argmax(dim=1, keepdim=True)
        bg_label = [0, 16, 18, 7, 8, 9, 14, 15]
        bg = sum(parsing_out == i for i in bg_label).bool()
        white_image = torch.ones_like(input)
        # only keep the face features
        face_features_image = torch.where(bg, white_image, input)
        face_features_image = tensor2img(face_features_image, rgb2bgr=False)

        return face_features_image


# generator = Generator()


@torch.inference_mode()
def dreamo_generate_image(
    dreamo_pipeline,ref_conds,
    # ref_image1,
    # ref_image2,
    # ref_task1,
    # ref_task2,
    prompt,
    width,
    height,
    num_steps,
    guidance,
    seed,
    true_cfg,
    cfg_start_step,
    cfg_end_step,
    neg_prompt,
    neg_guidance,
    first_step_guidance,
):
    #print(prompt)
    # ref_conds = []
    # #debug_images = []

    # ref_images = [ref_image1, ref_image2]
    # ref_tasks = [ref_task1, ref_task2]

    # for idx, (ref_image, ref_task) in enumerate(zip(ref_images, ref_tasks)):
    #     if ref_image is not None:
    #         if ref_task == "id":
    #             ref_image = generator.get_align_face(ref_image)
    #         elif ref_task != "style":
    #             ref_image = generator.bg_rm_model.inference(Image.fromarray(ref_image))
    #         ref_image = resize_numpy_image_area(np.array(ref_image), ref_res * ref_res)
    #         #debug_images.append(ref_image)
    #         ref_image = img2tensor(ref_image, bgr2rgb=False).unsqueeze(0) / 255.0
    #         ref_image = 2 * ref_image - 1.0
    #         ref_conds.append(
    #             {
    #                 'img': ref_image,
    #                 'task': ref_task,
    #                 'idx': idx + 1,
    #             }
    #         )

    seed = int(seed)
    if seed == -1:
        seed = torch.Generator(device="cpu").seed()

    image = dreamo_pipeline(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=num_steps,
        guidance_scale=guidance,
        ref_conds=ref_conds,
        generator=torch.Generator(device="cpu").manual_seed(seed),
        true_cfg_scale=true_cfg,
        true_cfg_start_step=cfg_start_step,
        true_cfg_end_step=cfg_end_step,
        negative_prompt=neg_prompt,
        neg_guidance_scale=neg_guidance,
        first_step_guidance_scale=first_step_guidance if first_step_guidance > 0 else guidance,
    ).images[0]

    return image


# _HEADER_ = '''
# <div style="text-align: center; max-width: 650px; margin: 0 auto;">
#     <h1 style="font-size: 2.5rem; font-weight: 700; margin-bottom: 1rem; display: contents;">DreamO</h1>
#     <p style="font-size: 1rem; margin-bottom: 1.5rem;">Paper: <a href='https://arxiv.org/abs/2504.16915' target='_blank'>DreamO: A Unified Framework for Image Customization</a> | Codes: <a href='https://github.com/bytedance/DreamO' target='_blank'>GitHub</a></p>
# </div>

# ❗️❗️❗️**User Guide:**
# - The most important thing to do first is to try the examples provided below the demo, which will help you better understand the capabilities of the DreamO model and the types of tasks it currently supports
# - For each input, please select the appropriate task type. For general objects, characters, or clothing, choose IP — we will remove the background from the input image. If you select ID, we will extract the face region from the input image (similar to PuLID). If you select Style, the background will be preserved, and you must prepend the prompt with the instruction: 'generate a same style image.' to activate the style task.
# - To accelerate inference, we adopt FLUX-turbo LoRA, which reduces the sampling steps from 25 to 12 compared to FLUX-dev. Additionally, we distill a CFG LoRA, achieving nearly a twofold reduction in steps by eliminating the need for true CFG

# '''  # noqa E501

# _CITE_ = r"""
# If DreamO is helpful, please help to ⭐ the <a href='https://github.com/bytedance/DreamO' target='_blank'> Github Repo</a>. Thanks!
# ---

# 📧 **Contact**
# If you have any questions or feedbacks, feel free to open a discussion or contact <b>wuyanze123@gmail.com</b> and <b>eechongm@gmail.com</b>
# """  # noqa E501


# def create_demo():

#     with gr.Blocks() as demo:
#         gr.Markdown(_HEADER_)

#         with gr.Row():
#             with gr.Column():
#                 with gr.Row():
#                     ref_image1 = gr.Image(label="ref image 1", type="numpy", height=256)
#                     ref_image2 = gr.Image(label="ref image 2", type="numpy", height=256)
#                 with gr.Row():
#                     ref_task1 = gr.Dropdown(choices=["ip", "id", "style"], value="ip", label="task for ref image 1")
#                     ref_task2 = gr.Dropdown(choices=["ip", "id", "style"], value="ip", label="task for ref image 2")
#                 prompt = gr.Textbox(label="Prompt", value="a person playing guitar in the street")
#                 width = gr.Slider(768, 1024, 1024, step=16, label="Width")
#                 height = gr.Slider(768, 1024, 1024, step=16, label="Height")
#                 num_steps = gr.Slider(8, 30, 12, step=1, label="Number of steps")
#                 guidance = gr.Slider(1.0, 10.0, 3.5, step=0.1, label="Guidance")
#                 seed = gr.Textbox(label="Seed (-1 for random)", value="-1")
#                 with gr.Accordion("Advanced Options", open=False, visible=False):
#                     ref_res = gr.Slider(512, 1024, 512, step=16, label="resolution for ref image")
#                     neg_prompt = gr.Textbox(label="Neg Prompt", value="")
#                     neg_guidance = gr.Slider(1.0, 10.0, 3.5, step=0.1, label="Neg Guidance")
#                     true_cfg = gr.Slider(1, 5, 1, step=0.1, label="true cfg")
#                     cfg_start_step = gr.Slider(0, 30, 0, step=1, label="cfg start step")
#                     cfg_end_step = gr.Slider(0, 30, 0, step=1, label="cfg end step")
#                     first_step_guidance = gr.Slider(0, 10, 0, step=0.1, label="first step guidance")
#                 generate_btn = gr.Button("Generate")
#                 gr.Markdown(_CITE_)

#             with gr.Column():
#                 output_image = gr.Image(label="Generated Image", format='png')
#                 debug_image = gr.Gallery(
#                     label="Preprocessing output (including possible face crop and background remove)",
#                     elem_id="gallery",
#                 )
#                 seed_output = gr.Textbox(label="Used Seed")

#         with gr.Row(), gr.Column():
#             gr.Markdown("## Examples")
#             example_inps = [
#                 [
#                     'example_inputs/woman1.png',
#                     'ip',
#                     'profile shot dark photo of a 25-year-old female with smoke escaping from her mouth, the backlit smoke gives the image an ephemeral quality, natural face, natural eyebrows, natural skin texture, award winning photo, highly detailed face, atmospheric lighting, film grain, monochrome',  # noqa E501
#                     9180879731249039735,
#                 ],
#                 [
#                     'example_inputs/man1.png',
#                     'ip',
#                     'a man sitting on the cloud, playing guitar',
#                     1206523688721442817,
#                 ],
#                 [
#                     'example_inputs/toy1.png',
#                     'ip',
#                     'a purple toy holding a sign saying "DreamO", on the mountain',
#                     1563188099017016129,
#                 ],
#                 [
#                     'example_inputs/perfume.png',
#                     'ip',
#                     'a perfume under spotlight',
#                     116150031980664704,
#                 ],
#             ]
#             gr.Examples(examples=example_inps, inputs=[ref_image1, ref_task1, prompt, seed], label='IP task')

#             example_inps = [
#                 [
#                     'example_inputs/hinton.jpeg',
#                     'id',
#                     'portrait, Chibi',
#                     5443415087540486371,
#                 ],
#             ]
#             gr.Examples(
#                 examples=example_inps,
#                 inputs=[ref_image1, ref_task1, prompt, seed],
#                 label='ID task (similar to PuLID, will only refer to the face)',
#             )

#             example_inps = [
#                 [
#                     'example_inputs/mickey.png',
#                     'style',
#                     'generate a same style image. A rooster wearing overalls.',
#                     6245580464677124951,
#                 ],
#                 [
#                     'example_inputs/mountain.png',
#                     'style',
#                     'generate a same style image. A pavilion by the river, and the distant mountains are endless',
#                     5248066378927500767,
#                 ],
#             ]
#             gr.Examples(examples=example_inps, inputs=[ref_image1, ref_task1, prompt, seed], label='Style task')

#             example_inps = [
#                 [
#                     'example_inputs/shirt.png',
#                     'example_inputs/skirt.jpeg',
#                     'ip',
#                     'ip',
#                     'A girl is wearing a short-sleeved shirt and a short skirt on the beach.',
#                     9514069256241143615,
#                 ],
#                 [
#                     'example_inputs/woman2.png',
#                     'example_inputs/dress.png',
#                     'id',
#                     'ip',
#                     'the woman wearing a dress, In the banquet hall',
#                     7698454872441022867,
#                 ],
#             ]
#             gr.Examples(
#                 examples=example_inps,
#                 inputs=[ref_image1, ref_image2, ref_task1, ref_task2, prompt, seed],
#                 label='Try-On task',
#             )

#             example_inps = [
#                 [
#                     'example_inputs/dog1.png',
#                     'example_inputs/dog2.png',
#                     'ip',
#                     'ip',
#                     'two dogs in the jungle',
#                     6187006025405083344,
#                 ],
#                 [
#                     'example_inputs/woman3.png',
#                     'example_inputs/cat.png',
#                     'ip',
#                     'ip',
#                     'A girl rides a giant cat, walking in the noisy modern city. High definition, realistic, non-cartoonish. Excellent photography work, 8k high definition.',  # noqa E501
#                     11980469406460273604,
#                 ],
#                 [
#                     'example_inputs/man2.jpeg',
#                     'example_inputs/woman4.jpeg',
#                     'ip',
#                     'ip',
#                     'a man is dancing with a woman in the room',
#                     8303780338601106219,
#                 ],
#             ]
#             gr.Examples(
#                 examples=example_inps,
#                 inputs=[ref_image1, ref_image2, ref_task1, ref_task2, prompt, seed],
#                 label='Multi IP',
#             )

#         generate_btn.click(
#             fn=generate_image,
#             inputs=[
#                 ref_image1,
#                 ref_image2,
#                 ref_task1,
#                 ref_task2,
#                 prompt,
#                 width,
#                 height,
#                 ref_res,
#                 num_steps,
#                 guidance,
#                 seed,
#                 true_cfg,
#                 cfg_start_step,
#                 cfg_end_step,
#                 neg_prompt,
#                 neg_guidance,
#                 first_step_guidance,
#             ],
#             outputs=[output_image, debug_image, seed_output],
#         )

#     return demo


# if __name__ == '__main__':
#     demo = create_demo()
#     demo.queue().launch(server_name='0.0.0.0', server_port=args.port)
