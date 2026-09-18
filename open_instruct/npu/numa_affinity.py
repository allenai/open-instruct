# Copyright 2024 AllenAI. All rights reserved.
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
"""NPU NUMA affinity helpers.

Ray actors only see the devices assigned to them (``ASCEND_RT_VISIBLE_DEVICES``
is set per actor), so the actor-visible device count cannot be used to split
the host's devices across NUMA nodes. The bind index must instead be derived
from the physical device id and the host device count.
"""

import glob


def parse_visible_devices(env_value: str) -> list[int]:
    """Parse an ``ASCEND_RT_VISIBLE_DEVICES`` value into physical device ids."""
    return [int(x) for x in env_value.split(",") if x.strip() != ""]


def host_npu_device_count() -> int:
    """Count NPUs on the host via ``/dev/davinci*``, independent of visibility masks."""
    return len(glob.glob("/dev/davinci[0-9]*"))


def npu_numa_node_index(local_rank: int, visible_devices: list[int], host_device_count: int, numa_nodes: int) -> int:
    """NUMA node index for an NPU rank, derived from its physical device id.

    ``local_rank`` indexes into the actor-visible devices; the physical id is
    what determines the NUMA node on the host.
    """
    physical_device_id = visible_devices[local_rank] if local_rank < len(visible_devices) else local_rank
    # Guard a zero node count and clamp so non-divisible topologies never hand
    # `numa_bind` a node id that does not exist.
    devices_per_numa_node = max(host_device_count // max(numa_nodes, 1), 1)
    node_index = physical_device_id // devices_per_numa_node
    return min(node_index, max(numa_nodes - 1, 0))
