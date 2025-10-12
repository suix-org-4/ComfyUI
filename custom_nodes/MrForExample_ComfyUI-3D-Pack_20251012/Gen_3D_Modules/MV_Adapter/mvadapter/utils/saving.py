import json
import math
import os
import re
import shutil
from typing import List, Optional, Union

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch

# import wandb
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw

from .typing import *


def image_to_tensor(
    images: List[Image.Image], device: str = "cuda"
) -> torch.FloatTensor:
    tensors = []
    for image in images:
        tensor = torch.from_numpy(np.array(image)).float() / 255.0
        tensor = tensor.to(device)
        tensors.append(tensor)
    return torch.stack(tensors)


def tensor_to_image(
    data: Union[Image.Image, torch.Tensor, np.ndarray],
    batched: bool = False,
    format: str = "HWC",
) -> Union[Image.Image, List[Image.Image]]:
    if isinstance(data, Image.Image):
        return data
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    if data.dtype == np.float32 or data.dtype == np.float16:
        data = (data * 255).astype(np.uint8)
    elif data.dtype == np.bool_:
        data = data.astype(np.uint8) * 255
    assert data.dtype == np.uint8
    if format == "CHW":
        if batched and data.ndim == 4:
            data = data.transpose((0, 2, 3, 1))
        elif not batched and data.ndim == 3:
            data = data.transpose((1, 2, 0))

    if batched:
        return [Image.fromarray(d) for d in data]
    return Image.fromarray(data)


def largest_factor_near_sqrt(n: int) -> int:
    """
    Finds the largest factor of n that is closest to the square root of n.

    Args:
        n (int): The integer for which to find the largest factor near its square root.

    Returns:
        int: The largest factor of n that is closest to the square root of n.
    """
    sqrt_n = int(math.sqrt(n))  # Get the integer part of the square root

    # First, check if the square root itself is a factor
    if sqrt_n * sqrt_n == n:
        return sqrt_n

    # Otherwise, find the largest factor by iterating from sqrt_n downwards
    for i in range(sqrt_n, 0, -1):
        if n % i == 0:
            return i

    # If n is 1, return 1
    return 1


def make_image_grid(
    images: List[Image.Image],
    rows: Optional[int] = None,
    cols: Optional[int] = None,
    resize: Optional[int] = None,
) -> Image.Image:
    """
    Prepares a single grid of images. Useful for visualization purposes.
    """
    if rows is None and cols is not None:
        assert len(images) % cols == 0
        rows = len(images) // cols
    elif cols is None and rows is not None:
        assert len(images) % rows == 0
        cols = len(images) // rows
    elif rows is None and cols is None:
        rows = largest_factor_near_sqrt(len(images))
        cols = len(images) // rows

    assert len(images) == rows * cols

    if resize is not None:
        images = [img.resize((resize, resize)) for img in images]

    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(images):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


class SaverMixin:
    _save_dir: Optional[str] = None
    _wandb_logger: Optional[Any] = None

    def set_save_dir(self, save_dir: str):
        self._save_dir = save_dir

    def get_save_dir(self):
        if self._save_dir is None:
            raise ValueError("Save dir is not set")
        return self._save_dir

    def convert_data(self, data):
        if data is None:
            return None
        elif isinstance(data, np.ndarray):
            return data
        elif isinstance(data, torch.Tensor):
            if data.dtype in [torch.float16, torch.bfloat16]:
                data = data.float()
            return data.detach().cpu().numpy()
        elif isinstance(data, list):
            return [self.convert_data(d) for d in data]
        elif isinstance(data, dict):
            return {k: self.convert_data(v) for k, v in data.items()}
        else:
            raise TypeError(
                "Data must be in type numpy.ndarray, torch.Tensor, list or dict, getting",
                type(data),
            )

    def get_save_path(self, filename):
        save_path = os.path.join(self.get_save_dir(), filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        return save_path

    DEFAULT_RGB_KWARGS = {"data_format": "HWC", "data_range": (0, 1)}
    DEFAULT_UV_KWARGS = {
        "data_format": "HWC",
        "data_range": (0, 1),
        "cmap": "checkerboard",
    }
    DEFAULT_GRAYSCALE_KWARGS = {"data_range": None, "cmap": "jet"}
    DEFAULT_GRID_KWARGS = {"align": "max"}

    def get_rgb_image_(self, img, data_format, data_range, rgba=False):
        img = self.convert_data(img)
        assert data_format in ["CHW", "HWC"]
        if data_format == "CHW":
            img = img.transpose(1, 2, 0)
        if img.dtype != np.uint8:
            img = img.clip(min=data_range[0], max=data_range[1])
            img = (
                (img - data_range[0]) / (data_range[1] - data_range[0]) * 255.0
            ).astype(np.uint8)
        nc = 4 if rgba else 3
        imgs = [img[..., start : start + nc] for start in range(0, img.shape[-1], nc)]
        imgs = [
            (
                img_
                if img_.shape[-1] == nc
                else np.concatenate(
                    [
                        img_,
                        np.zeros(
                            (img_.shape[0], img_.shape[1], nc - img_.shape[2]),
                            dtype=img_.dtype,
                        ),
                    ],
                    axis=-1,
                )
            )
            for img_ in imgs
        ]
        img = np.concatenate(imgs, axis=1)
        if rgba:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def _save_rgb_image(
        self,
        filename,
        img,
        data_format,
        data_range,
        name: Optional[str] = None,
        step: Optional[int] = None,
    ):
        img = self.get_rgb_image_(img, data_format, data_range)
        cv2.imwrite(filename, img)
        if name and self._wandb_logger:
            self._wandb_logger.log_image(
                key=name, images=[self.get_save_path(filename)], step=step
            )

    def save_rgb_image(
        self,
        filename,
        img,
        data_format=DEFAULT_RGB_KWARGS["data_format"],
        data_range=DEFAULT_RGB_KWARGS["data_range"],
        name: Optional[str] = None,
        step: Optional[int] = None,
    ) -> str:
        save_path = self.get_save_path(filename)
        self._save_rgb_image(save_path, img, data_format, data_range, name, step)
        return save_path

    def get_uv_image_(self, img, data_format, data_range, cmap):
        img = self.convert_data(img)
        assert data_format in ["CHW", "HWC"]
        if data_format == "CHW":
            img = img.transpose(1, 2, 0)
        img = img.clip(min=data_range[0], max=data_range[1])
        img = (img - data_range[0]) / (data_range[1] - data_range[0])
        assert cmap in ["checkerboard", "color"]
        if cmap == "checkerboard":
            n_grid = 64
            mask = (img * n_grid).astype(int)
            mask = (mask[..., 0] + mask[..., 1]) % 2 == 0
            img = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8) * 255
            img[mask] = np.array([255, 0, 255], dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif cmap == "color":
            img_ = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
            img_[..., 0] = (img[..., 0] * 255).astype(np.uint8)
            img_[..., 1] = (img[..., 1] * 255).astype(np.uint8)
            img_ = cv2.cvtColor(img_, cv2.COLOR_RGB2BGR)
            img = img_
        return img

    def save_uv_image(
        self,
        filename,
        img,
        data_format=DEFAULT_UV_KWARGS["data_format"],
        data_range=DEFAULT_UV_KWARGS["data_range"],
        cmap=DEFAULT_UV_KWARGS["cmap"],
    ) -> str:
        save_path = self.get_save_path(filename)
        img = self.get_uv_image_(img, data_format, data_range, cmap)
        cv2.imwrite(save_path, img)
        return save_path

    def get_grayscale_image_(self, img, data_range, cmap):
        img = self.convert_data(img)
        img = np.nan_to_num(img)
        if data_range is None:
            img = (img - img.min()) / (img.max() - img.min())
        else:
            img = img.clip(data_range[0], data_range[1])
            img = (img - data_range[0]) / (data_range[1] - data_range[0])
        assert cmap in [None, "jet", "magma", "spectral"]
        if cmap == None:
            img = (img * 255.0).astype(np.uint8)
            img = np.repeat(img[..., None], 3, axis=2)
        elif cmap == "jet":
            img = (img * 255.0).astype(np.uint8)
            img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        elif cmap == "magma":
            img = 1.0 - img
            base = cm.get_cmap("magma")
            num_bins = 256
            colormap = LinearSegmentedColormap.from_list(
                f"{base.name}{num_bins}", base(np.linspace(0, 1, num_bins)), num_bins
            )(np.linspace(0, 1, num_bins))[:, :3]
            a = np.floor(img * 255.0)
            b = (a + 1).clip(max=255.0)
            f = img * 255.0 - a
            a = a.astype(np.uint16).clip(0, 255)
            b = b.astype(np.uint16).clip(0, 255)
            img = colormap[a] + (colormap[b] - colormap[a]) * f[..., None]
            img = (img * 255.0).astype(np.uint8)
        elif cmap == "spectral":
            colormap = plt.get_cmap("Spectral")

            def blend_rgba(image):
                image = image[..., :3] * image[..., -1:] + (
                    1.0 - image[..., -1:]
                )  # blend A to RGB
                return image

            img = colormap(img)
            img = blend_rgba(img)
            img = (img * 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def _save_grayscale_image(
        self,
        filename,
        img,
        data_range,
        cmap,
        name: Optional[str] = None,
        step: Optional[int] = None,
    ):
        img = self.get_grayscale_image_(img, data_range, cmap)
        cv2.imwrite(filename, img)
        if name and self._wandb_logger:
            self._wandb_logger.log_image(
                key=name, images=[self.get_save_path(filename)], step=step
            )

    def save_grayscale_image(
        self,
        filename,
        img,
        data_range=DEFAULT_GRAYSCALE_KWARGS["data_range"],
        cmap=DEFAULT_GRAYSCALE_KWARGS["cmap"],
        name: Optional[str] = None,
        step: Optional[int] = None,
    ) -> str:
        save_path = self.get_save_path(filename)
        self._save_grayscale_image(save_path, img, data_range, cmap, name, step)
        return save_path

    def get_image_grid_(self, imgs, align):
        if isinstance(imgs[0], list):
            return np.concatenate(
                [self.get_image_grid_(row, align) for row in imgs], axis=0
            )
        cols = []
        for col in imgs:
            assert col["type"] in ["rgb", "uv", "grayscale"]
            if col["type"] == "rgb":
                rgb_kwargs = self.DEFAULT_RGB_KWARGS.copy()
                rgb_kwargs.update(col["kwargs"])
                cols.append(self.get_rgb_image_(col["img"], **rgb_kwargs))
            elif col["type"] == "uv":
                uv_kwargs = self.DEFAULT_UV_KWARGS.copy()
                uv_kwargs.update(col["kwargs"])
                cols.append(self.get_uv_image_(col["img"], **uv_kwargs))
            elif col["type"] == "grayscale":
                grayscale_kwargs = self.DEFAULT_GRAYSCALE_KWARGS.copy()
                grayscale_kwargs.update(col["kwargs"])
                cols.append(self.get_grayscale_image_(col["img"], **grayscale_kwargs))

        if align == "max":
            h = max([col.shape[0] for col in cols])
        elif align == "min":
            h = min([col.shape[0] for col in cols])
        elif isinstance(align, int):
            h = align
        else:
            raise ValueError(
                f"Unsupported image grid align: {align}, should be min, max, or int"
            )

        for i in range(len(cols)):
            if cols[i].shape[0] != h:
                w = int(cols[i].shape[1] * h / cols[i].shape[0])
                cols[i] = cv2.resize(cols[i], (w, h), interpolation=cv2.INTER_CUBIC)
        return np.concatenate(cols, axis=1)

    def save_image_grid(
        self,
        filename,
        imgs,
        align=DEFAULT_GRID_KWARGS["align"],
        name: Optional[str] = None,
        step: Optional[int] = None,
        texts: Optional[List[float]] = None,
    ):
        save_path = self.get_save_path(filename)
        img = self.get_image_grid_(imgs, align=align)

        if texts is not None:
            img = Image.fromarray(img)
            draw = ImageDraw.Draw(img)
            black, white = (0, 0, 0), (255, 255, 255)
            for i, text in enumerate(texts):
                draw.text((2, (img.size[1] // len(texts)) * i + 1), f"{text}", white)
                draw.text((0, (img.size[1] // len(texts)) * i + 1), f"{text}", white)
                draw.text((2, (img.size[1] // len(texts)) * i - 1), f"{text}", white)
                draw.text((0, (img.size[1] // len(texts)) * i - 1), f"{text}", white)
                draw.text((1, (img.size[1] // len(texts)) * i), f"{text}", black)
            img = np.asarray(img)

        cv2.imwrite(save_path, img)
        if name and self._wandb_logger:
            self._wandb_logger.log_image(key=name, images=[save_path], step=step)
        return save_path

    def save_image(self, filename, img) -> str:
        save_path = self.get_save_path(filename)
        img = self.convert_data(img)
        assert img.dtype == np.uint8 or img.dtype == np.uint16
        if img.ndim == 3 and img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif img.ndim == 3 and img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(save_path, img)
        return save_path

    def save_cubemap(self, filename, img, data_range=(0, 1), rgba=False) -> str:
        save_path = self.get_save_path(filename)
        img = self.convert_data(img)
        assert img.ndim == 4 and img.shape[0] == 6 and img.shape[1] == img.shape[2]

        imgs_full = []
        for start in range(0, img.shape[-1], 3):
            img_ = img[..., start : start + 3]
            img_ = np.stack(
                [
                    self.get_rgb_image_(img_[i], "HWC", data_range, rgba=rgba)
                    for i in range(img_.shape[0])
                ],
                axis=0,
            )
            size = img_.shape[1]
            placeholder = np.zeros((size, size, 3), dtype=np.float32)
            img_full = np.concatenate(
                [
                    np.concatenate(
                        [placeholder, img_[2], placeholder, placeholder], axis=1
                    ),
                    np.concatenate([img_[1], img_[4], img_[0], img_[5]], axis=1),
                    np.concatenate(
                        [placeholder, img_[3], placeholder, placeholder], axis=1
                    ),
                ],
                axis=0,
            )
            imgs_full.append(img_full)

        imgs_full = np.concatenate(imgs_full, axis=1)
        cv2.imwrite(save_path, imgs_full)
        return save_path

    def save_data(self, filename, data) -> str:
        data = self.convert_data(data)
        if isinstance(data, dict):
            if not filename.endswith(".npz"):
                filename += ".npz"
            save_path = self.get_save_path(filename)
            np.savez(save_path, **data)
        else:
            if not filename.endswith(".npy"):
                filename += ".npy"
            save_path = self.get_save_path(filename)
            np.save(save_path, data)
        return save_path

    def save_state_dict(self, filename, data) -> str:
        save_path = self.get_save_path(filename)
        torch.save(data, save_path)
        return save_path

    def save_img_sequence(
        self,
        filename,
        img_dir,
        matcher,
        save_format="mp4",
        fps=30,
        name: Optional[str] = None,
        step: Optional[int] = None,
    ) -> str:
        assert save_format in ["gif", "mp4"]
        if not filename.endswith(save_format):
            filename += f".{save_format}"
        save_path = self.get_save_path(filename)
        matcher = re.compile(matcher)
        img_dir = os.path.join(self.get_save_dir(), img_dir)
        imgs = []
        for f in os.listdir(img_dir):
            if matcher.search(f):
                imgs.append(f)
        imgs = sorted(imgs, key=lambda f: int(matcher.search(f).groups()[0]))
        imgs = [cv2.imread(os.path.join(img_dir, f)) for f in imgs]

        if save_format == "gif":
            imgs = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in imgs]
            imageio.mimsave(save_path, imgs, fps=fps, palettesize=256)
        elif save_format == "mp4":
            imgs = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in imgs]
            imageio.mimsave(save_path, imgs, fps=fps)
        if name and self._wandb_logger:
            from .core import warn

            warn("Wandb logger does not support video logging yet!")
        return save_path

    def save_img_sequences(
        self,
        seq_dir,
        matcher,
        save_format="mp4",
        fps=30,
        delete=True,
        name: Optional[str] = None,
        step: Optional[int] = None,
    ):
        seq_dir_ = os.path.join(self.get_save_dir(), seq_dir)
        for f in os.listdir(seq_dir_):
            img_dir_ = os.path.join(seq_dir_, f)
            if not os.path.isdir(img_dir_):
                continue
            try:
                self.save_img_sequence(
                    os.path.join(seq_dir, f),
                    os.path.join(seq_dir, f),
                    matcher,
                    save_format=save_format,
                    fps=fps,
                    name=f"{name}_{f}",
                    step=step,
                )
                if delete:
                    shutil.rmtree(img_dir_)
            except:
                from .core import warn

                warn(f"Video saving for directory {seq_dir_} failed!")

    def save_file(self, filename, src_path, delete=False) -> str:
        save_path = self.get_save_path(filename)
        shutil.copyfile(src_path, save_path)
        if delete:
            os.remove(src_path)
        return save_path

    def save_json(self, filename, payload) -> str:
        save_path = self.get_save_path(filename)
        with open(save_path, "w") as f:
            f.write(json.dumps(payload))
        return save_path
