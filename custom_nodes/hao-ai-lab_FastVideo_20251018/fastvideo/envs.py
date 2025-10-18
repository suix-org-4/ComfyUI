# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/envs.py

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    FASTVIDEO_RINGBUFFER_WARNING_INTERVAL: int = 60
    FASTVIDEO_NCCL_SO_PATH: str | None = None
    LD_LIBRARY_PATH: str | None = None
    LOCAL_RANK: int = 0
    CUDA_VISIBLE_DEVICES: str | None = None
    FASTVIDEO_CACHE_ROOT: str = os.path.expanduser("~/.cache/fastvideo")
    FASTVIDEO_CONFIG_ROOT: str = os.path.expanduser("~/.config/fastvideo")
    FASTVIDEO_CONFIGURE_LOGGING: int = 1
    FASTVIDEO_RAY_PER_WORKER_GPUS: float = 1.0
    FASTVIDEO_LOGGING_LEVEL: str = "INFO"
    FASTVIDEO_LOGGING_PREFIX: str = ""
    FASTVIDEO_LOGGING_CONFIG_PATH: str | None = None
    FASTVIDEO_TRACE_FUNCTION: int = 0
    FASTVIDEO_ATTENTION_BACKEND: str | None = None
    FASTVIDEO_ATTENTION_CONFIG: str | None = None
    FASTVIDEO_WORKER_MULTIPROC_METHOD: str = "fork"
    FASTVIDEO_TARGET_DEVICE: str = "cuda"
    MAX_JOBS: str | None = None
    NVCC_THREADS: str | None = None
    CMAKE_BUILD_TYPE: str | None = None
    VERBOSE: bool = False
    FASTVIDEO_TORCH_PROFILER_DIR: str | None = None
    FASTVIDEO_TORCH_PROFILER_RECORD_SHAPES: bool = False
    FASTVIDEO_TORCH_PROFILER_WITH_PROFILE_MEMORY: bool = False
    FASTVIDEO_TORCH_PROFILER_WITH_STACK: bool = True
    FASTVIDEO_TORCH_PROFILER_WITH_FLOPS: bool = False
    FASTVIDEO_TORCH_PROFILE_REGIONS: str = ""
    FASTVIDEO_SERVER_DEV_MODE: bool = False
    FASTVIDEO_STAGE_LOGGING: bool = False
    FASTVIDEO_HOST_IP: str = ""
    FASTVIDEO_LOOPBACK_IP: str = ""


def get_default_cache_root() -> str:
    return os.getenv(
        "XDG_CACHE_HOME",
        os.path.join(os.path.expanduser("~"), ".cache"),
    )


def get_default_config_root() -> str:
    return os.getenv(
        "XDG_CONFIG_HOME",
        os.path.join(os.path.expanduser("~"), ".config"),
    )


def maybe_convert_int(value: str | None) -> int | None:
    if value is None:
        return None
    return int(value)


# The begin-* and end* here are used by the documentation generator
# to extract the used env vars.

# begin-env-vars-definition

environment_variables: dict[str, Callable[[], Any]] = {

    # ================== Installation Time Env Vars ==================

    # Target device of FastVideo, supporting [cuda (by default),
    # rocm, neuron, cpu, openvino]
    "FASTVIDEO_TARGET_DEVICE":
    lambda: os.getenv("FASTVIDEO_TARGET_DEVICE", "cuda"),

    # Maximum number of compilation jobs to run in parallel.
    # By default this is the number of CPUs
    "MAX_JOBS":
    lambda: os.getenv("MAX_JOBS", None),

    # Number of threads to use for nvcc
    # By default this is 1.
    # If set, `MAX_JOBS` will be reduced to avoid oversubscribing the CPU.
    "NVCC_THREADS":
    lambda: os.getenv("NVCC_THREADS", None),

    # If set, fastvideo will use precompiled binaries (*.so)
    "FASTVIDEO_USE_PRECOMPILED":
    lambda: bool(os.environ.get("FASTVIDEO_USE_PRECOMPILED")) or bool(
        os.environ.get("FASTVIDEO_PRECOMPILED_WHEEL_LOCATION")),

    # CMake build type
    # If not set, defaults to "Debug" or "RelWithDebInfo"
    # Available options: "Debug", "Release", "RelWithDebInfo"
    "CMAKE_BUILD_TYPE":
    lambda: os.getenv("CMAKE_BUILD_TYPE"),

    # If set, fastvideo will print verbose logs during installation
    "VERBOSE":
    lambda: bool(int(os.getenv('VERBOSE', '0'))),

    # Root directory for FASTVIDEO configuration files
    # Defaults to `~/.config/fastvideo` unless `XDG_CONFIG_HOME` is set
    # Note that this not only affects how fastvideo finds its configuration files
    # during runtime, but also affects how fastvideo installs its configuration
    # files during **installation**.
    "FASTVIDEO_CONFIG_ROOT":
    lambda: os.path.expanduser(
        os.getenv(
            "FASTVIDEO_CONFIG_ROOT",
            os.path.join(get_default_config_root(), "fastvideo"),
        )),

    # ================== Runtime Env Vars ==================

    # Root directory for FASTVIDEO cache files
    # Defaults to `~/.cache/fastvideo` unless `XDG_CACHE_HOME` is set
    "FASTVIDEO_CACHE_ROOT":
    lambda: os.path.expanduser(
        os.getenv(
            "FASTVIDEO_CACHE_ROOT",
            os.path.join(get_default_cache_root(), "fastvideo"),
        )),

    # used in distributed environment to determine the ip address
    # of the current node, when the node has multiple network interfaces.
    # If you are using multi-node inference, you should set this differently
    # on each node.
    "FASTVIDEO_HOST_IP":
    lambda: os.getenv("FASTVIDEO_HOST_IP", ""),

    # Used to force set up loopback IP
    "FASTVIDEO_LOOPBACK_IP":
    lambda: os.getenv("FASTVIDEO_LOOPBACK_IP", ""),

    # Number of GPUs per worker in Ray, if it is set to be a fraction,
    # it allows ray to schedule multiple actors on a single GPU,
    # so that users can colocate other actors on the same GPUs as FastVideo.
    "FASTVIDEO_RAY_PER_WORKER_GPUS":
    lambda: float(os.getenv("FASTVIDEO_RAY_PER_WORKER_GPUS", "1.0")),

    # Interval in seconds to log a warning message when the ring buffer is full
    "FASTVIDEO_RINGBUFFER_WARNING_INTERVAL":
    lambda: int(os.environ.get("FASTVIDEO_RINGBUFFER_WARNING_INTERVAL", "60")),

    # Path to the NCCL library file. It is needed because nccl>=2.19 brought
    # by PyTorch contains a bug: https://github.com/NVIDIA/nccl/issues/1234
    "FASTVIDEO_NCCL_SO_PATH":
    lambda: os.environ.get("FASTVIDEO_NCCL_SO_PATH", None),

    # when `FASTVIDEO_NCCL_SO_PATH` is not set, fastvideo will try to find the nccl
    # library file in the locations specified by `LD_LIBRARY_PATH`
    "LD_LIBRARY_PATH":
    lambda: os.environ.get("LD_LIBRARY_PATH", None),

    # Internal flag to enable Dynamo fullgraph capture
    "FASTVIDEO_TEST_DYNAMO_FULLGRAPH_CAPTURE":
    lambda: bool(
        os.environ.get("FASTVIDEO_TEST_DYNAMO_FULLGRAPH_CAPTURE", "1") != "0"),

    # local rank of the process in the distributed setting, used to determine
    # the GPU device id
    "LOCAL_RANK":
    lambda: int(os.environ.get("LOCAL_RANK", "0")),

    # used to control the visible devices in the distributed setting
    "CUDA_VISIBLE_DEVICES":
    lambda: os.environ.get("CUDA_VISIBLE_DEVICES", None),

    # timeout for each iteration in the engine
    "FASTVIDEO_ENGINE_ITERATION_TIMEOUT_S":
    lambda: int(os.environ.get("FASTVIDEO_ENGINE_ITERATION_TIMEOUT_S", "60")),

    # Logging configuration
    # If set to 0, fastvideo will not configure logging
    # If set to 1, fastvideo will configure logging using the default configuration
    #    or the configuration file specified by FASTVIDEO_LOGGING_CONFIG_PATH
    "FASTVIDEO_CONFIGURE_LOGGING":
    lambda: int(os.getenv("FASTVIDEO_CONFIGURE_LOGGING", "1")),
    "FASTVIDEO_LOGGING_CONFIG_PATH":
    lambda: os.getenv("FASTVIDEO_LOGGING_CONFIG_PATH"),

    # this is used for configuring the default logging level
    "FASTVIDEO_LOGGING_LEVEL":
    lambda: os.getenv("FASTVIDEO_LOGGING_LEVEL", "INFO"),

    # if set, FASTVIDEO_LOGGING_PREFIX will be prepended to all log messages
    "FASTVIDEO_LOGGING_PREFIX":
    lambda: os.getenv("FASTVIDEO_LOGGING_PREFIX", ""),

    # Trace function calls
    # If set to 1, fastvideo will trace function calls
    # Useful for debugging
    "FASTVIDEO_TRACE_FUNCTION":
    lambda: int(os.getenv("FASTVIDEO_TRACE_FUNCTION", "0")),

    # Backend for attention computation
    # Available options:
    # - "TORCH_SDPA": use torch.nn.MultiheadAttention
    # - "FLASH_ATTN": use FlashAttention
    # - "SLIDING_TILE_ATTN" : use Sliding Tile Attention
    # - "VIDEO_SPARSE_ATTN": use Video Sparse Attention
    # - "SAGE_ATTN": use Sage Attention
    # - "SAGE_ATTN_THREE": use Sage Attention 3
    "FASTVIDEO_ATTENTION_BACKEND":
    lambda: os.getenv("FASTVIDEO_ATTENTION_BACKEND", None),

    # Path to the attention configuration file. Only used for sliding tile
    # attention for now.
    "FASTVIDEO_ATTENTION_CONFIG":
    lambda: (None if os.getenv("FASTVIDEO_ATTENTION_CONFIG", None) is None else
             os.path.expanduser(os.getenv("FASTVIDEO_ATTENTION_CONFIG", "."))),

    # Use dedicated multiprocess context for workers.
    # Both spawn and fork work
    "FASTVIDEO_WORKER_MULTIPROC_METHOD":
    lambda: os.getenv("FASTVIDEO_WORKER_MULTIPROC_METHOD", "fork"),

    # Enables torch profiler if set. Path to the directory where torch profiler
    # traces are saved. Note that it must be an absolute path.
    "FASTVIDEO_TORCH_PROFILER_DIR":
    lambda: (None
             if os.getenv("FASTVIDEO_TORCH_PROFILER_DIR", None) is None else os.
             path.expanduser(os.getenv("FASTVIDEO_TORCH_PROFILER_DIR", "."))),

    # Enable torch profiler to record shapes if set
    # FASTVIDEO_TORCH_PROFILER_RECORD_SHAPES=1. If not set, torch profiler will
    # not record shapes.
    "FASTVIDEO_TORCH_PROFILER_RECORD_SHAPES":
    lambda: bool(
        os.getenv("FASTVIDEO_TORCH_PROFILER_RECORD_SHAPES", "0") != "0"),

    # Enable torch profiler to profile memory if set
    # FASTVIDEO_TORCH_PROFILER_WITH_PROFILE_MEMORY=1. If not set, torch profiler
    # will not profile memory.
    "FASTVIDEO_TORCH_PROFILER_WITH_PROFILE_MEMORY":
    lambda: bool(
        os.getenv("FASTVIDEO_TORCH_PROFILER_WITH_PROFILE_MEMORY", "0") != "0"),

    # Enable torch profiler to profile stack if set
    # FASTVIDEO_TORCH_PROFILER_WITH_STACK=1. If not set, torch profiler WILL
    # profile stack by default.
    "FASTVIDEO_TORCH_PROFILER_WITH_STACK":
    lambda: bool(os.getenv("FASTVIDEO_TORCH_PROFILER_WITH_STACK", "1") != "0"),

    # Enable torch profiler to profile flops if set
    # FASTVIDEO_TORCH_PROFILER_WITH_FLOPS=1. If not set, torch profiler will
    # not profile flops.
    "FASTVIDEO_TORCH_PROFILER_WITH_FLOPS":
    lambda: bool(os.getenv("FASTVIDEO_TORCH_PROFILER_WITH_FLOPS", "0") != "0"),
    "FASTVIDEO_TORCH_PROFILE_REGIONS":
    lambda: os.getenv("FASTVIDEO_TORCH_PROFILE_REGIONS", ""),

    # If set, fastvideo will run in development mode, which will enable
    # some additional endpoints for developing and debugging,
    # e.g. `/reset_prefix_cache`
    "FASTVIDEO_SERVER_DEV_MODE":
    lambda: bool(int(os.getenv("FASTVIDEO_SERVER_DEV_MODE", "0"))),

    # If set, fastvideo will enable stage logging, which will print the time
    # taken for each stage
    "FASTVIDEO_STAGE_LOGGING":
    lambda: bool(int(os.getenv("FASTVIDEO_STAGE_LOGGING", "0"))),
}

# end-env-vars-definition


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())
