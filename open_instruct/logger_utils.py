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

import logging

import torch.distributed as dist


def setup_logger(name: str | None = None) -> logging.Logger:
    """Set up a logger with consistent formatting across the project.

    Only rank 0 logs at INFO; all other ranks log at WARNING.
    Rank is auto-detected from torch.distributed.

    Args:
        name: Logger name (typically __name__). If None, returns root logger.

    Returns:
        Logger instance with the specified name
    """
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    level = logging.INFO if rank == 0 else logging.WARNING
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    else:
        logging.getLogger().setLevel(level)

    return logging.getLogger(name)
