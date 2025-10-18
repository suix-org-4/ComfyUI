# SPDX-License-Identifier: Apache-2.0
# Adapt from https://github.com/vllm-project/vllm/blob/releases/v0.11.0/vllm/executor/ray_utils.py

from collections import defaultdict
import os
import time

from fastvideo.utils import get_ip
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.worker.worker_base import WorkerWrapperBase
from fastvideo.logger import init_logger

logger = init_logger(__name__)
PG_WAIT_TIMEOUT = 1800

try:
    import ray
    from ray.util import placement_group_table
    from ray.util.placement_group import PlacementGroup
    try:
        from ray._private.state import available_resources_per_node
    except ImportError:
        # Ray 2.9.x doesn't expose `available_resources_per_node`
        from ray._private.state import state as _state
        available_resources_per_node = _state._available_resources_per_node

    class RayWorkerWrapper(WorkerWrapperBase):
        """Ray wrapper for fastvideo.worker.Worker, allowing Worker to be
        lazily initialized after Ray sets CUDA_VISIBLE_DEVICES."""

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)

        def get_node_ip(self) -> str:
            return get_ip()

        def get_node_and_gpu_ids(self) -> tuple[str, list[int]]:
            from fastvideo.platforms import current_platform

            node_id = ray.get_runtime_context().get_node_id()
            device_key = current_platform.ray_device_key
            if not device_key:
                raise RuntimeError("current platform %s does not support ray.",
                                   current_platform.device_name)
            gpu_ids = ray.get_runtime_context().get_accelerator_ids(
            )[device_key]
            return node_id, gpu_ids

        def override_env_vars(self, vars: dict[str, str]):
            os.environ.update(vars)

    ray_import_err = None
except ImportError as e:
    ray = None
    ray_import_err = str(e)


def assert_ray_available() -> None:
    """Raise an exception if Ray is not available."""
    if ray is None:
        raise ValueError(f"Failed to import Ray: {ray_import_err}."
                         "Please install Ray with `pip install ray`.")


def _verify_bundles(placement_group: "PlacementGroup",
                    fastvideo_args: FastVideoArgs, device_str: str):
    """Verify a given placement group has bundles located in the right place.

    There are 2 rules.
    - Warn if all tensor parallel workers cannot fit in a single node.
    - Fail if driver node is not included in a placement group.
    """
    assert ray.is_initialized(), (
        "Ray is not initialized although distributed-executor-backend is ray.")
    pg_data = placement_group_table(placement_group)
    # bundle_idx -> node_id
    bundle_to_node_ids = pg_data["bundles_to_node_id"]
    # bundle_idx -> bundle (e.g., {"GPU": 1})
    bundles = pg_data["bundles"]
    # node_id -> List of bundle (e.g., {"GPU": 1})
    node_id_to_bundle: dict[str, list[dict[str, float]]] = defaultdict(list)

    for bundle_idx, node_id in bundle_to_node_ids.items():
        node_id_to_bundle[node_id].append(bundles[bundle_idx])
    driver_node_id = ray.get_runtime_context().get_node_id()

    if driver_node_id not in node_id_to_bundle:
        raise RuntimeError(
            f"driver node id {driver_node_id} is not included in a placement "
            f"group {placement_group.id}. Node id -> bundles "
            f"{node_id_to_bundle}. "
            "You don't have enough GPUs available in a current node. Check "
            "`ray status` and `ray list nodes` to see if you have available "
            "GPUs in a node `{driver_node_id}` before starting an FastVideo engine."
        )

    for node_id, bundles in node_id_to_bundle.items():
        if len(bundles) < fastvideo_args.tp_size:
            logger.warning(
                "tensor_parallel_size=%d "
                "is bigger than a reserved number of %ss (%d "
                "%ss) in a node %s. Tensor parallel workers can be "
                "spread out to 2+ nodes which can degrade the performance "
                "unless you have fast interconnect across nodes, like "
                "Infiniband. To resolve this issue, make sure you have more "
                "than %d GPUs available at each node.", fastvideo_args.tp_size,
                device_str, len(bundles), device_str, node_id,
                fastvideo_args.tp_size)


def _wait_until_pg_ready(current_placement_group: "PlacementGroup"):
    """Wait until a placement group is ready.

    It prints the informative log messages if the placement group is
    not created within time.

    """
    # Wait until PG is ready - this will block until all
    # requested resources are available, and will timeout
    # if they cannot be provisioned.
    placement_group_specs = current_placement_group.bundle_specs

    s = time.time()
    pg_ready_ref = current_placement_group.ready()
    wait_interval = 10
    while time.time() - s < PG_WAIT_TIMEOUT:
        ready, _ = ray.wait([pg_ready_ref], timeout=wait_interval)
        if len(ready) > 0:
            break

        # Exponential backoff for warning print.
        wait_interval *= 2
        logger.info(
            "Waiting for creating a placement group of specs for "
            "%d seconds. specs=%s. Check `ray status` and "
            "`ray list nodes` to see if you have enough resources,"
            " and make sure the IP addresses used by ray cluster"
            " are the same as FASTVIDEO_HOST_IP environment variable"
            " specified in each node if you are running on a multi-node.",
            int(time.time() - s), placement_group_specs)

    try:
        ray.get(pg_ready_ref, timeout=0)
    except ray.exceptions.GetTimeoutError:
        raise ValueError(
            "Cannot provide a placement group of "
            f"{placement_group_specs=} within {PG_WAIT_TIMEOUT} seconds. See "
            "`ray status` and `ray list nodes` to make sure the cluster has "
            "enough resources.") from None


def initialize_ray_cluster(
    fastvideo_args: FastVideoArgs,
    ray_address: str | None = None,
):
    """Initialize the distributed cluster with Ray.

    it will connect to the Ray cluster and create a placement group
    for the workers, which includes the specification of the resources
    for each distributed worker.

    Args:
        parallel_config: The configurations for parallel execution.
        ray_address: The address of the Ray cluster. If None, uses
            the default Ray cluster address.
    """
    assert_ray_available()
    from fastvideo.platforms import current_platform

    if ray.is_initialized():
        logger.info("Ray is already initialized. Skipping Ray initialization.")
    elif current_platform.is_rocm() or current_platform.is_xpu():
        # Try to connect existing ray instance and create a new one if not found
        try:
            ray.init("auto")
        except ConnectionError:
            logger.warning(
                "No existing RAY instance detected. "
                "A new instance will be launched with current node resources.")
            ray.init(address=ray_address,
                     num_gpus=fastvideo_args.num_gpus,
                     runtime_env=fastvideo_args.ray_runtime_env)
    else:
        ray.init(address=ray_address,
                 runtime_env=fastvideo_args.ray_runtime_env)

    device_str = current_platform.ray_device_key
    if not device_str:
        raise ValueError(
            f"current platform {current_platform.device_name} does not "
            "support ray.")

    # Create or get the placement group for worker processes
    if fastvideo_args.ray_placement_group:
        current_placement_group = fastvideo_args.ray_placement_group
    else:
        current_placement_group = ray.util.get_current_placement_group()

    if current_placement_group:
        logger.info("Using the existing placement group")

        # We are in a placement group
        bundles = current_placement_group.bundle_specs
        # Verify that we can use the placement group.
        device_bundles = 0
        for bundle in bundles:
            bundle_devices = bundle.get(device_str, 0)
            if bundle_devices > 1:
                raise ValueError(
                    "Placement group bundle cannot have more than 1 "
                    f"{device_str}.")
            if bundle_devices:
                device_bundles += 1
        if fastvideo_args.num_gpus > device_bundles:
            raise ValueError(
                f"The number of required {device_str}s exceeds the total "
                f"number of available {device_str}s in the placement group. "
                f"Required number of devices: {fastvideo_args.num_gpus}. "
                f"Total number of devices: {device_bundles}.")
    else:
        logger.info("No current placement group found. "
                    "Creating a new placement group.")
        num_devices_in_cluster = ray.cluster_resources().get(device_str, 0)
        # Log a warning message and delay resource allocation failure response.
        # Avoid immediate rejection to allow user-initiated placement group
        # created and wait cluster to be ready
        if fastvideo_args.num_gpus > num_devices_in_cluster:
            logger.warning(
                "The number of required %ss exceeds the total "
                "number of available %ss in the placement group.", device_str,
                device_str)
        # Create a new placement group
        placement_group_specs: list[dict[str, float]] = ([{
            device_str: 1.0
        } for _ in range(fastvideo_args.num_gpus)])

        # FastVideo engine is also a worker to execute model with an accelerator,
        # so it requires to have the device in a current node. Check if
        # the current node has at least one device.
        current_ip = get_ip()
        current_node_id = ray.get_runtime_context().get_node_id()
        current_node_resource = available_resources_per_node()[current_node_id]
        if current_node_resource.get(device_str, 0) < 1:
            raise ValueError(
                f"Current node has no {device_str} available. "
                f"{current_node_resource=}. FastVideo engine cannot start without "
                f"{device_str}. Make sure you have at least 1 {device_str} "
                f"available in a node {current_node_id=} {current_ip=}.")
        # This way, at least bundle is required to be created in a current
        # node.
        placement_group_specs[0][f"node:{current_ip}"] = 0.001

        # By default, Ray packs resources as much as possible.
        current_placement_group = ray.util.placement_group(
            placement_group_specs, strategy="PACK")
        _wait_until_pg_ready(current_placement_group)

    assert current_placement_group is not None
    _verify_bundles(current_placement_group, fastvideo_args, device_str)
    # Set the placement group in the fastvideo args
    fastvideo_args.ray_placement_group = current_placement_group


def is_in_ray_actor():
    """Check if we are in a Ray actor."""

    try:
        import ray
        return (ray.is_initialized()
                and ray.get_runtime_context().get_actor_id() is not None)
    except ImportError:
        return False
