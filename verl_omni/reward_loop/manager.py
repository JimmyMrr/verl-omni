# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

import os

import ray
from verl.experimental.reward_loop import RewardLoopManager


class NPUAwareRewardLoopManager(RewardLoopManager):
    """RewardLoopManager that exposes accelerator devices to reward workers.

    The upstream ``RewardLoopManager`` creates workers as CPU-only Ray actors
    (no ``num_gpus`` / NPU resource).  On Ascend NPU this means Ray sets
    ``ASCEND_RT_VISIBLE_DEVICES`` to an empty string, so ``aclInit`` fails
    with error 107001 when a reward function calls ``.to("npu")``.

    This subclass keeps workers as CPU-only actors (to avoid reserving
    accelerator resources already held by actor/rollout workers) but injects
    the platform's visible-devices env var via ``runtime_env`` so the NPU
    runtime can find devices.  It also sets the platform's
    ``RAY_EXPERIMENTAL_NOSET_*`` env vars to prevent Ray from overriding
    the injected visible-devices value for CPU-only actors.  Workers are
    distributed round-robin across available accelerators.

    When the launcher restricts visible devices via the platform's
    visible-devices env var (e.g. ``ASCEND_RT_VISIBLE_DEVICES=8,9,...,15``),
    workers are round-robined *within that subset* instead of over the full
    node, so reward workers co-locate with actor/rollout workers on the
    user-selected devices rather than spilling onto die 0..N.
    """

    def _init_reward_loop_workers(self):
        from verl.plugin.platform import get_platform

        self.reward_loop_workers = []
        num_workers = self.config.reward.num_workers
        alive_nodes = [node for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]
        node_ids = [node["NodeID"] for node in alive_nodes]

        platform = get_platform()
        env_var = platform.visible_devices_envvar()
        resource_name = platform.ray_resource_name()

        num_devices = 0
        for node in alive_nodes:
            num_devices = max(num_devices, int(node["Resources"].get(resource_name, 0)))
        if num_devices == 0:
            num_devices = int(self.config.trainer.n_gpus_per_node)

        # Respect the launcher's visible-devices restriction (e.g.
        # ASCEND_RT_VISIBLE_DEVICES="8,9,...,15") so reward workers land on the
        # same die subset as actor/rollout workers instead of always starting
        # from die 0.  Fall back to round-robin over all node devices when the
        # env var is unset/empty.
        visible_env = os.environ.get(env_var, "").strip()
        if visible_env:
            visible_pool = [int(d.strip()) for d in visible_env.split(",") if d.strip() != ""]
            if not visible_pool:  # malformed value, fall back
                visible_pool = list(range(num_devices))
        else:
            visible_pool = list(range(num_devices))

        # Prevent Ray from overriding the visible-devices env var for CPU-only
        # actors.  Without these flags Ray sets ASCEND_RT_VISIBLE_DEVICES="" for
        # actors with num_gpus=0, hiding all NPU devices even though we inject
        # the env var via runtime_env above.
        noset_vars = {var: "1" for var in platform.ray_noset_envvars()}

        # If every configured reward function targets CPU, reward workers never
        # call ``.to("npu")`` / ``aclInit``, so they don't need accelerator
        # visibility at all.  Skip injecting the visible-devices env var and the
        # NOSET flags — let Ray's default behavior hide all NPU devices from
        # these CPU-only actors, avoiding any NPU runtime initialization and
        # the associated HBM reservation.
        all_cpu = True
        reward_functions_cfg = self.config.reward.get("reward_functions", {})
        for _name, entry in reward_functions_cfg.items():
            if entry.get("device", "cuda") != "cpu":
                all_cpu = False
                break

        for i in range(num_workers):
            node_id = node_ids[i % len(node_ids)]

            if all_cpu:
                env_vars = {}
            else:
                device_id = visible_pool[i % len(visible_pool)]
                env_vars = {env_var: str(device_id), **noset_vars}

            opts = {
                "name": f"reward_loop_worker_{i}",
                "scheduling_strategy": ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=True,
                ),
                "runtime_env": {"env_vars": env_vars},
            }

            self.reward_loop_workers.append(
                self.reward_loop_workers_class.options(**opts).remote(self.config, self.reward_router_address)
            )
