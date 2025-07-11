import logging
import torch
import numpy as np
from torch.profiler import record_function
from einops import rearrange, repeat, reduce

logger = logging.getLogger(__name__)
logger.info("Begin to load kmeans operator...")
try:
    from .libKMCUDA import kmeans_cuda  # type: ignore
except ImportError as e:
    logger.error("Fail to load kmeans operator from local path.")
    logger.exception(e)
    print("Please use libKMCUDA built from https://github.com/duanyll/kmcuda. The built libKMCUDA.so file should be placed in the same directory as this file. Do not use the official libKMCUDA from pip.")
    raise e
logger.info("Finish loading kmeans operator.")

seed = 42


def kmeans(samples: torch.Tensor, cluster_num: int, cached_center=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run kmeans on samples. Result is on the same device as samples. If cached_center is not None, it will be used as the initial cluster center.
    Args:
        samples: (sample_num, feature_dim)
        cluster_num: int
        cached_center: (cluster_num, feature_dim)
    Returns:
        cluster_idx: (sample_num)
        cluster_centers: (cluster_num, feature_dim)
        cluster_loss: (1)
    """
    if cluster_num <= 1:
        if samples.shape[0] == 0:
            _cluster_num_eff = max(cluster_num, 0)
            return torch.empty(0, dtype=torch.long, device=samples.device), \
                   torch.empty((_cluster_num_eff, samples.shape[1] if samples.ndim > 1 else 1), dtype=samples.dtype, device=samples.device), \
                   torch.tensor(0.0, device=samples.device)

        idx = torch.zeros(samples.shape[0], dtype=torch.long, device=samples.device)
        centers = samples.mean(dim=0, keepdim=True)
        loss = ((samples - centers) ** 2).mean()
        return idx, centers, loss

    if cluster_num > samples.shape[0]:
        logger.warning(
            f"cluster_num ({cluster_num}) > sample_num ({samples.shape[0]}). Setting cluster_num = sample_num.")
        cluster_num = samples.shape[0]
        if samples.shape[0] == 0:
            return torch.empty(0, dtype=torch.long, device=samples.device), \
                   torch.empty((0, samples.shape[1] if samples.ndim > 1 else 1), dtype=samples.dtype, device=samples.device), \
                   torch.tensor(0.0, device=samples.device)
        
        idx = torch.arange(samples.shape[0], device=samples.device, dtype=torch.long)
        centers = samples.clone()
        return idx, centers, torch.tensor(0.0, device=samples.device) # Loss is 0 as each point is a center

    with record_function("kmeans"):
        if cached_center is None:
            idx, centers = kmeans_cuda(samples, cluster_num, seed=seed)
        else:
            idx, centers = kmeans_cuda(samples, cluster_num,
                                 initial_centroids=cached_center, seed=seed)
    
    # Calculate loss
    loss = ((samples - centers[idx]) ** 2).mean()
    return idx.long(), centers, loss
