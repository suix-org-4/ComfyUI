
import math
import numpy as np
import torch
import gc
from typing import List, Optional, Tuple, Union
import os
from functools import lru_cache


@lru_cache(maxsize=None)
def CHECK_ENABLE_PROFILING_DEBUG():
    ENABLE_PROFILING_DEBUG = os.getenv("ENABLE_PROFILING_DEBUG", "false").lower() == "true"
    return ENABLE_PROFILING_DEBUG


@lru_cache(maxsize=None)
def CHECK_ENABLE_GRAPH_MODE():
    ENABLE_GRAPH_MODE = os.getenv("ENABLE_GRAPH_MODE", "false").lower() == "true"
    return ENABLE_GRAPH_MODE


@lru_cache(maxsize=None)
def GET_RUNNING_FLAG():
    RUNNING_FLAG = os.getenv("RUNNING_FLAG", "infer")
    return RUNNING_FLAG


@lru_cache(maxsize=None)
def GET_DTYPE():
    RUNNING_FLAG = os.getenv("DTYPE")
    return RUNNING_FLAG

class BaseScheduler:
    def __init__(self, config):
        self.config = config
        self.step_index = 0
        self.latents = None
        self.infer_steps = config.infer_steps
        self.caching_records = [True] * config.infer_steps
        self.flag_df = False
        self.transformer_infer = None

    def step_pre(self, step_index):
        self.step_index = step_index
        if GET_DTYPE() == "BF16":
            self.latents = self.latents.to(dtype=torch.bfloat16)

    def clear(self):
        pass

    
class WanScheduler(BaseScheduler):
    def __init__(self, config):
        super().__init__(config)
        self.device = torch.device("cuda")
        self.infer_steps = self.config.infer_steps
        self.target_video_length = self.config.target_video_length
        self.sample_shift = self.config.sample_shift
        self.shift = 1
        self.num_train_timesteps = 1000
        self.disable_corrector = []
        self.solver_order = 2
        self.noise_pred = None

        self.caching_records_2 = [True] * self.config.infer_steps

    def prepare(self, image_encoder_output=None):
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(self.config.seed)

        self.prepare_latents(self.config.target_shape, dtype=torch.float32)

        if self.config.task in ["t2v"]:
            self.seq_len = math.ceil((self.config.target_shape[2] * self.config.target_shape[3]) / (self.config.patch_size[1] * self.config.patch_size[2]) * self.config.target_shape[1])
        elif self.config.task in ["i2v"]:
            self.seq_len = ((self.config.target_video_length - 1) // self.config.vae_stride[0] + 1) * self.config.lat_h * self.config.lat_w // (self.config.patch_size[1] * self.config.patch_size[2])

        alphas = np.linspace(1, 1 / self.num_train_timesteps, self.num_train_timesteps)[::-1].copy()
        sigmas = 1.0 - alphas
        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32)

        sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)

        self.sigmas = sigmas
        self.timesteps = sigmas * self.num_train_timesteps

        self.model_outputs = [None] * self.solver_order
        self.timestep_list = [None] * self.solver_order
        self.last_sample = None

        self.sigmas = self.sigmas.to("cpu")
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()

        self.set_timesteps(self.infer_steps, device=self.device, shift=self.sample_shift)

    def prepare_latents(self, target_shape, dtype=torch.float32):
        self.latents = torch.randn(
            target_shape[0],
            target_shape[1],
            target_shape[2],
            target_shape[3],
            dtype=dtype,
            device=self.device,
            generator=self.generator,
        )

    def set_timesteps(
        self,
        infer_steps: Union[int, None] = None,
        device: Union[str, torch.device] = None,
        sigmas: Optional[List[float]] = None,
        mu: Optional[Union[float, None]] = None,
        shift: Optional[Union[float, None]] = None,
    ):
        sigmas = np.linspace(self.sigma_max, self.sigma_min, infer_steps + 1).copy()[:-1]

        if shift is None:
            shift = self.shift
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        sigma_last = 0

        timesteps = sigmas * self.num_train_timesteps
        sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)

        self.sigmas = torch.from_numpy(sigmas)
        self.timesteps = torch.from_numpy(timesteps).to(device=device, dtype=torch.int64)

        assert len(self.timesteps) == self.infer_steps
        self.model_outputs = [
            None,
        ] * self.solver_order
        self.lower_order_nums = 0
        self.last_sample = None
        self._begin_index = None
        self.sigmas = self.sigmas.to("cpu")

    def _sigma_to_alpha_sigma_t(self, sigma):
        return 1 - sigma, sigma

    def convert_model_output(
        self,
        model_output: torch.Tensor,
        *args,
        sample: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        timestep = args[0] if len(args) > 0 else kwargs.pop("timestep", None)
        if sample is None:
            if len(args) > 1:
                sample = args[1]
            else:
                raise ValueError("missing `sample` as a required keyward argument")

        sigma = self.sigmas[self.step_index]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
        sigma_t = self.sigmas[self.step_index]
        x0_pred = sample - sigma_t * model_output
        return x0_pred

    def reset(self):
        self.model_outputs = [None] * self.solver_order
        self.timestep_list = [None] * self.solver_order
        self.last_sample = None
        self.noise_pred = None
        self.this_order = None
        self.lower_order_nums = 0
        self.prepare_latents(self.config.target_shape, dtype=torch.float32)
        gc.collect()
        torch.cuda.empty_cache()

    def multistep_uni_p_bh_update(
        self,
        model_output: torch.Tensor,
        *args,
        sample: torch.Tensor = None,
        order: int = None,
        **kwargs,
    ) -> torch.Tensor:
        prev_timestep = args[0] if len(args) > 0 else kwargs.pop("prev_timestep", None)
        if sample is None:
            if len(args) > 1:
                sample = args[1]
            else:
                raise ValueError(" missing `sample` as a required keyward argument")
        if order is None:
            if len(args) > 2:
                order = args[2]
            else:
                raise ValueError(" missing `order` as a required keyward argument")
        model_output_list = self.model_outputs

        s0 = self.timestep_list[-1]
        m0 = model_output_list[-1]
        x = sample

        sigma_t, sigma_s0 = (
            self.sigmas[self.step_index + 1],
            self.sigmas[self.step_index],
        )
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)

        h = lambda_t - lambda_s0
        device = sample.device

        rks = []
        D1s = []
        for i in range(1, order):
            si = self.step_index - i
            mi = model_output_list[-(i + 1)]
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)

        rks.append(1.0)
        rks = torch.tensor(rks, device=device)

        R = []
        b = []

        hh = -h
        h_phi_1 = torch.expm1(hh)  # h\phi_1(h) = e^h - 1
        h_phi_k = h_phi_1 / hh - 1

        factorial_i = 1

        B_h = torch.expm1(hh)

        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        R = torch.stack(R)
        b = torch.tensor(b, device=device)

        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1)  # (B, K)
            # for order 2, we use a simplified version
            if order == 2:
                rhos_p = torch.tensor([0.5], dtype=x.dtype, device=device)
            else:
                rhos_p = torch.linalg.solve(R[:-1, :-1], b[:-1]).to(device).to(x.dtype)
        else:
            D1s = None

        x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
        if D1s is not None:
            pred_res = torch.einsum("k,bkc...->bc...", rhos_p, D1s)
        else:
            pred_res = 0
        x_t = x_t_ - alpha_t * B_h * pred_res
        x_t = x_t.to(x.dtype)
        return x_t

    def multistep_uni_c_bh_update(
        self,
        this_model_output: torch.Tensor,
        *args,
        last_sample: torch.Tensor = None,
        this_sample: torch.Tensor = None,
        order: int = None,
        **kwargs,
    ) -> torch.Tensor:
        this_timestep = args[0] if len(args) > 0 else kwargs.pop("this_timestep", None)
        if last_sample is None:
            if len(args) > 1:
                last_sample = args[1]
            else:
                raise ValueError(" missing`last_sample` as a required keyward argument")
        if this_sample is None:
            if len(args) > 2:
                this_sample = args[2]
            else:
                raise ValueError(" missing`this_sample` as a required keyward argument")
        if order is None:
            if len(args) > 3:
                order = args[3]
            else:
                raise ValueError(" missing`order` as a required keyward argument")

        model_output_list = self.model_outputs

        m0 = model_output_list[-1]
        x = last_sample
        x_t = this_sample
        model_t = this_model_output

        sigma_t, sigma_s0 = (
            self.sigmas[self.step_index],
            self.sigmas[self.step_index - 1],
        )
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)

        h = lambda_t - lambda_s0
        device = this_sample.device

        rks = []
        D1s = []
        for i in range(1, order):
            si = self.step_index - (i + 1)
            mi = model_output_list[-(i + 1)]
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)

        rks.append(1.0)
        rks = torch.tensor(rks, device=device)

        R = []
        b = []

        hh = -h
        h_phi_1 = torch.expm1(hh)  # h\phi_1(h) = e^h - 1
        h_phi_k = h_phi_1 / hh - 1

        factorial_i = 1

        B_h = torch.expm1(hh)

        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        R = torch.stack(R)
        b = torch.tensor(b, device=device)

        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1)
        else:
            D1s = None

        # for order 1, we use a simplified version
        if order == 1:
            rhos_c = torch.tensor([0.5], dtype=x.dtype, device=device)
        else:
            rhos_c = torch.linalg.solve(R, b).to(device).to(x.dtype)

        x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
        if D1s is not None:
            corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], D1s)
        else:
            corr_res = 0
        D1_t = model_t - m0
        x_t = x_t_ - alpha_t * B_h * (corr_res + rhos_c[-1] * D1_t)
        x_t = x_t.to(x.dtype)
        return x_t

    def step_post(self):
        model_output = self.noise_pred.to(torch.float32)
        timestep = self.timesteps[self.step_index]
        sample = self.latents.to(torch.float32)

        use_corrector = self.step_index > 0 and self.step_index - 1 not in self.disable_corrector and self.last_sample is not None

        model_output_convert = self.convert_model_output(model_output, sample=sample)
        if use_corrector:
            sample = self.multistep_uni_c_bh_update(
                this_model_output=model_output_convert,
                last_sample=self.last_sample,
                this_sample=sample,
                order=self.this_order,
            )

        for i in range(self.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
            self.timestep_list[i] = self.timestep_list[i + 1]

        self.model_outputs[-1] = model_output_convert
        self.timestep_list[-1] = timestep

        this_order = min(self.solver_order, len(self.timesteps) - self.step_index)

        self.this_order = min(this_order, self.lower_order_nums + 1)  # warmup for multistep
        assert self.this_order > 0

        self.last_sample = sample
        prev_sample = self.multistep_uni_p_bh_update(
            model_output=model_output,
            sample=sample,
            order=self.this_order,
        )

        if self.lower_order_nums < self.solver_order:
            self.lower_order_nums += 1

        self.latents = prev_sample



class WanStepDistillScheduler(WanScheduler):
    def __init__(self, config):
        super().__init__(config)
        self.denoising_step_list = config.denoising_step_list
        self.infer_steps = len(self.denoising_step_list)
        self.sample_shift = self.config.sample_shift
        self.order = 1
        self.num_train_timesteps = 1000
        self.sigma_max = 1.0
        self.sigma_min = 0.0

    def prepare(self, image_encoder_output):
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(self.config.seed)

        self.prepare_latents(self.config.target_shape, dtype=torch.float32)

        if self.config.task in ["t2v"]:
            self.seq_len = math.ceil((self.config.target_shape[2] * self.config.target_shape[3]) / (self.config.patch_size[1] * self.config.patch_size[2]) * self.config.target_shape[1])
        elif self.config.task in ["i2v"]:
            self.seq_len = self.config.lat_h * self.config.lat_w // (self.config.patch_size[1] * self.config.patch_size[2]) * self.config.target_shape[1]

        self.set_denoising_timesteps(device=self.device)

    def set_denoising_timesteps(self, device: Union[str, torch.device] = None):
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min)
        self.sigmas = torch.linspace(sigma_start, self.sigma_min, self.num_train_timesteps + 1)[:-1]
        self.sigmas = self.sample_shift * self.sigmas / (1 + (self.sample_shift - 1) * self.sigmas)
        self.timesteps = self.sigmas * self.num_train_timesteps

        self.denoising_step_index = [self.num_train_timesteps - x for x in self.denoising_step_list]
        self.timesteps = self.timesteps[self.denoising_step_index].to(device)
        self.sigmas = self.sigmas[self.denoising_step_index].to("cpu")

    def reset(self):
        self.prepare_latents(self.config.target_shape, dtype=torch.float32)

    def add_noise(self, original_samples, noise, sigma):
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample.type_as(noise)

    def step_post(self):
        flow_pred = self.noise_pred.to(torch.float32)
        sigma = self.sigmas[self.step_index].item()
        noisy_image_or_video = self.latents.to(torch.float32) - sigma * flow_pred
        if self.step_index < self.infer_steps - 1:
            sigma = self.sigmas[self.step_index + 1].item()
            noisy_image_or_video = self.add_noise(noisy_image_or_video, torch.randn_like(noisy_image_or_video), self.sigmas[self.step_index + 1].item())
        self.latents = noisy_image_or_video.to(self.latents.dtype)
        
    def step(self, model_output, timestep, sample, generator=None, return_dict=True):
        """
        使用模型输出预测数据并执行去噪步骤。
        
        Args:
            model_output (`torch.Tensor`): 直接输出来自模型的预测。
            timestep (`int`): 当前的离散时间步。
            sample (`torch.Tensor`): 在时间步t处的输入样本。
            generator (`torch.Generator`, optional): 用于采样的随机数生成器。
            return_dict (`bool`, optional): 是否返回字典格式的结果。
            
        Returns:
            `torch.Tensor` 或 `Dict[str, torch.Tensor]`: 更新后的样本。
        """
        # 设置当前步骤索引
        step_index = torch.where(self.timesteps == timestep)[0].item()
        
        # 保存必要的属性以供step_post使用
        self.noise_pred = model_output
        self.latents = sample
        self.step_index = step_index
        
        # 执行后处理步骤
        self.step_post()
        
        # 返回更新后的样本
        if return_dict:
            return {"prev_sample": self.latents}
        else:
            return (self.latents,)
