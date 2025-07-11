import torch
from einops import rearrange, repeat, reduce
from .cluster_center import get_cluster_centers_scatter
from .kmeans import kmeans

store = {}
"""
module_id -> (cache_size, sample_num)
Stored on the same device as samples.
"""
cache_mode = "disable"  # "disable", "init", "update", "ready",
"""
Cache is used to reduce kmeans computation by reusing previous results between a few epochs.
Disable: No cache is read or written to, kmeans is run from scratch every time.
Init: Run kmeans with random cluster center initialization and cache the result. Used for the first epoch.
Update: Run kmeans with previous cluster center initialization and cache the result. Used for the following epochs.
Ready: Use cached result directly. Used for the following epochs.
"""
cache_size = 0
"""
Defines size to allocate for cache. Usually set to the number of samples in the dataset.
"""


def reset_cache(size):
    global store
    store = {}

    global cache_size
    cache_size = size


def kmeans_batched(samples: torch.Tensor, cluster_num: int, cache_indice=None, module_id=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run kmeans on batched samples. Will use cache if cache_indice and module_id is not None. Result is on the same device as samples.
    Args:
        samples: (batch_size, sample_num, feature_dim)
        cluster_num: int
        cache_indice: int tensor (batch_size) or None
    Returns:
        cluster_idx: (batch_size, sample_num)
        cluster_centers: (batch_size, cluster_num, feature_dim)
        cluster_loss: (1)
    """
    dev = samples.device
    batch_size, sample_num, feature_dim = samples.shape
    all_indices = torch.empty(batch_size, sample_num, dtype=torch.long, device=dev)
    all_centers = torch.empty(batch_size, cluster_num, feature_dim, dtype=samples.dtype, device=dev)
    total_loss = 0.0

    if cache_mode != "disable" and cache_indice is not None and module_id is not None and batch_size > 0:
        if module_id not in store:
            store[module_id] = torch.zeros(
                (cache_size, sample_num), dtype=torch.long, device=dev)
        if cache_mode == "init":
            for i in range(batch_size):
                idx_i, centers_i, loss_i = kmeans(samples[i, :, :], cluster_num)
                total_loss += loss_i
                store[module_id][cache_indice[i], :] = idx_i
                all_indices[i, :] = idx_i
                actual_k_i = centers_i.shape[0]
                if actual_k_i == cluster_num:
                    all_centers[i, :, :] = centers_i
                elif actual_k_i < cluster_num:
                    all_centers[i, :actual_k_i, :] = centers_i
                    if actual_k_i > 0:
                        all_centers[i, actual_k_i:, :] = centers_i[actual_k_i-1:actual_k_i, :].repeat(cluster_num - actual_k_i, 1)
                    else:
                        all_centers[i, :, :] = 0
                else:
                    all_centers[i, :, :] = centers_i[:cluster_num, :]
            return all_indices, all_centers, total_loss / batch_size
        elif cache_mode == "update":
            for i in range(batch_size):
                last_idx_i = store[module_id][cache_indice[i]].to(dev)
                initial_centers_i = None
                if last_idx_i.max() < cluster_num:
                    try:
                        initial_centers_i_batch = get_cluster_centers_scatter(
                            samples[i].unsqueeze(0), last_idx_i.unsqueeze(0), cluster_num)
                        initial_centers_i = initial_centers_i_batch.squeeze(0)
                    except:
                        initial_centers_i = None
                idx_i, centers_i, loss_i = kmeans(samples[i, :, :], cluster_num, cached_center=initial_centers_i)
                total_loss += loss_i
                store[module_id][cache_indice[i], :] = idx_i.detach()
                all_indices[i, :] = idx_i
                actual_k_i = centers_i.shape[0]
                if actual_k_i == cluster_num:
                    all_centers[i, :, :] = centers_i
                elif actual_k_i < cluster_num:
                    all_centers[i, :actual_k_i, :] = centers_i
                    if actual_k_i > 0:
                        all_centers[i, actual_k_i:, :] = centers_i[actual_k_i-1:actual_k_i, :].repeat(cluster_num - actual_k_i, 1)
                    else:
                        all_centers[i, :, :] = 0
                else:
                    all_centers[i, :, :] = centers_i[:cluster_num, :]
            return all_indices, all_centers, total_loss / batch_size
        elif cache_mode == "ready":
            cached_indices_batch = store[module_id][cache_indice]
            all_indices = cached_indices_batch.to(dev)
            for i in range(batch_size):
                pass
            current_centers_batch = get_cluster_centers_scatter(samples, all_indices, cluster_num)
            all_centers = current_centers_batch
            loss = ((samples - all_centers[torch.arange(batch_size).unsqueeze(1), all_indices]) ** 2).mean()
            return all_indices, all_centers, loss
        else:
            for i in range(batch_size):
                idx_i, centers_i, loss_i = kmeans(samples[i, :, :], cluster_num)
                total_loss += loss_i
                all_indices[i, :] = idx_i
                actual_k_i = centers_i.shape[0]
                if actual_k_i == cluster_num:
                    all_centers[i, :, :] = centers_i
                elif actual_k_i < cluster_num:
                    all_centers[i, :actual_k_i, :] = centers_i
                    if actual_k_i > 0:
                        all_centers[i, actual_k_i:, :] = centers_i[actual_k_i-1:actual_k_i, :].repeat(cluster_num - actual_k_i, 1)
                    else:
                        all_centers[i, :, :] = 0
                else:
                    all_centers[i, :, :] = centers_i[:cluster_num, :]
            return all_indices, all_centers, total_loss / batch_size
    else:
        if batch_size == 0:
            return all_indices, all_centers, torch.tensor(0.0, device=dev)
        for i in range(batch_size):
            idx_i, centers_i, loss_i = kmeans(samples[i, :, :], cluster_num)
            total_loss += loss_i
            all_indices[i, :] = idx_i
            actual_k_i = centers_i.shape[0]
            if actual_k_i == cluster_num:
                all_centers[i, :, :] = centers_i
            elif actual_k_i < cluster_num:
                all_centers[i, :actual_k_i, :] = centers_i
                if actual_k_i > 0:
                    all_centers[i, actual_k_i:, :] = centers_i[actual_k_i-1:actual_k_i, :].repeat(cluster_num - actual_k_i, 1)
                else:
                    all_centers[i, :, :] = 0
            else:
                all_centers[i, :, :] = centers_i[:cluster_num, :]
        return all_indices, all_centers, total_loss / batch_size


class KMeansCacheScheduler:
    def __init__(self, policy):
        assert isinstance(policy, int) or isinstance(
            policy, list), "policy must be int or list"
        self.policy = policy
        self.current_epoch = 0

    def step(self):
        """
        Example: policy = [(100, 10), (300, 20), 50]
        1. At epoch 1, cache_mode = "init"
        2. For epoch 2 to 100, Every 10 epochs, cache_mode = "update", else cache_mode = "ready"
        3. For epoch 101 to 300, Every 20 epochs, cache_mode = "update", else cache_mode = "ready"
        4. For epoch > 300, Every 50 epochs, cache_mode = "update", else cache_mode = "ready"

        Example: policy = 10
        1. At epoch 1, cache_mode = "init"
        2. Every 10 epochs, cache_mode = "update", else cache_mode = "ready"
        """
        global cache_mode
        self.current_epoch += 1
        if self.current_epoch == 1:
            cache_mode = "init"
        elif isinstance(self.policy, int):
            if self.current_epoch % self.policy == 0:
                cache_mode = "update"
            else:
                cache_mode = "ready"
        else:
            for i in range(len(self.policy)):
                if isinstance(self.policy[i], int):
                    if self.current_epoch % self.policy[i] == 0:
                        cache_mode = "update"
                        break
                    else:
                        cache_mode = "ready"
                        break
                else:
                    if self.current_epoch > self.policy[i][0]:
                        pass
                    elif self.current_epoch % self.policy[i][1] == 0:
                        cache_mode = "update"
                        break
                    else:
                        cache_mode = "ready"
                        break
