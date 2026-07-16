import logging

import torch

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(
            "Using GPU: %s",
            torch.cuda.get_device_name(device),
        )
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    return device
