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


def setup_logger(name: str | None = None, rank: int = 0) -> logging.Logger:
    """Set up a logger with consistent formatting across the project.

    This function configures logging.basicConfig with a standard format
    that includes timestamp, level, filename, line number, and message.
    It only configures basicConfig once to avoid overwriting existing config.

    Args:
        name: Logger name (typically __name__). If None, returns root logger.
        rank: Process rank in distributed training. Only rank 0 logs INFO.

    Returns:
        Logger instance with the specified name
    """
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO if rank == 0 else logging.WARNING,
            format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    return logging.getLogger(name)
