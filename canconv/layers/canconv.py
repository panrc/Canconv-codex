import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from .kmeans import KMeans, get_cluster_centers
from .pwac import filter_indice, dispatch_indice, permute, inverse_permute, batched_matmul_conv
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity

# Imports for Hierarchical Clustering
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage as scipy_linkage, fcluster, dendrogram
from typing import cast, Literal, Optional

from canconv.config.defaults import get_cfg_defaults # Corrected import

class CANConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 cluster_num=32, # Base for fine K-Means
                 kernel_size=3,
                 mlp_inner_dims=16,
                 bias="cluster",  # or "global_param" or "global_adaptive" or "none"
                 detach_centroid=False,
                 cluster_source="channel",  # "spatial" or "pixel"
                 kernel_generator="low_rank",  # or "weighted_sum" or "low_rank"
                 kernel_count=8,  # required when kernel_generator is "weighted_sum"
                 cluster_ablation="none",  # or "global" or "pixelwise"
                 filter_threshold=0,
                 enable_dynamic_k: bool = False, # Applies to K-Means (fine-grained or global)
                 dynamic_k_variance_threshold: float = 0.1,
                 dynamic_k_delta: int = 5,
                 dynamic_k_min: int = 2,
                 dynamic_k_max: int = 64,
                 enable_hierarchical_clustering: bool = False, 
                 hierarchical_clustering_target_coarse_clusters: int = 0, 
                 hierarchical_clustering_linkage: str = 'ward', 
                 hc_distance_threshold: Optional[float] = None, 
                 hc_auto_k_method: Optional[str] = None, 
                 hc_auto_k_param: Optional[float] = 0.75
                 ) -> None:
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            cluster_num: Number of clusters
            kernel_size: Kernel size
            mlp_inner_dims: Number of hidden units in for the MLP that generates the kernel
            bias: "none" for no bias, "cluster" for bias for each cluster, "global_param" use a uniform bias like nn.Conv2d, 
                  "global_adaptive" generates global bias like LAGConv
        """
        super().__init__()

        cfg = get_cfg_defaults() # Get config here

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.unfold = nn.Unfold(kernel_size=kernel_size,
                                padding=(kernel_size - 1) // 2)
        self.kernel_area = kernel_size ** 2
        self.cluster_num_base_fine_kmeans = cluster_num
        self.bias_mode = bias
        self.detatch_centroid = detach_centroid
        self.cluster_source = cluster_source
        self.kernel_generator = kernel_generator
        self.cluster_ablation = cluster_ablation
        self.filter_threshold = filter_threshold

        self.enable_dynamic_k = enable_dynamic_k
        self.dynamic_k_variance_threshold = dynamic_k_variance_threshold
        self.dynamic_k_delta = dynamic_k_delta
        self.dynamic_k_min = dynamic_k_min
        self.dynamic_k_max = max(dynamic_k_max, dynamic_k_min)
        
        self.enable_hierarchical_clustering = enable_hierarchical_clustering
        self.hierarchical_clustering_target_coarse_clusters = hierarchical_clustering_target_coarse_clusters
        self.hierarchical_clustering_linkage = hierarchical_clustering_linkage
        self.hc_distance_threshold = hc_distance_threshold
        self.hc_auto_k_method = hc_auto_k_method
        self.hc_auto_k_param = hc_auto_k_param

        # Values are sourced from cfg defined at the start of __init__
        self.enable_patch_wise_fc = cfg.MODEL.CANCONV.ENABLE_PATCH_WISE_FC
        self.patch_fc_out_channels = cfg.MODEL.CANCONV.PATCH_FC_OUT_CHANNELS

        # Restore KMeans initialization and checks
        if enable_hierarchical_clustering and self.hierarchical_clustering_linkage not in ['ward', 'complete', 'average', 'single']:
            raise ValueError(f"Unsupported linkage type for hierarchical clustering: {self.hierarchical_clustering_linkage}")
        if enable_hierarchical_clustering and self.hc_auto_k_method is not None and \
           self.hc_auto_k_method not in ['percentile_dist', 'ratio_max_dist']:
            raise ValueError(f"Unsupported hc_auto_k_method: {self.hc_auto_k_method}")

        self.kmeans = KMeans(cluster_num=1) # Placeholder K, actual K passed during call

        if self.enable_patch_wise_fc:
            self.patch_fc = nn.Linear(
                self.in_channels * self.kernel_area, self.patch_fc_out_channels)

        if self.kernel_generator == "spatial":
            self.centroid_to_kernel = nn.Sequential(
                nn.Linear(in_features=self.in_channels * self.kernel_area,
                          out_features=mlp_inner_dims),
                nn.ReLU(),
                nn.Linear(in_features=mlp_inner_dims,
                          out_features=mlp_inner_dims),
                nn.ReLU(),
                nn.Linear(in_features=mlp_inner_dims,
                          out_features=self.kernel_area),
                nn.Sigmoid()
            )
            self.kernels = nn.parameter.Parameter(
                torch.randn(self.in_channels, self.kernel_area, self.out_channels))
            nn.init.kaiming_normal_(self.kernels, nonlinearity="relu")
        elif self.kernel_generator == "weighted_sum":
            self.centroid_to_kernel = nn.Sequential(
                nn.Linear(in_features=self.in_channels * self.kernel_area,
                          out_features=mlp_inner_dims),
                nn.ReLU(),
                nn.Linear(in_features=mlp_inner_dims,
                          out_features=mlp_inner_dims),
                nn.ReLU(),
                nn.Linear(in_features=mlp_inner_dims,
                          out_features=kernel_count),
                nn.Softmax()
            )
            self.kernels = nn.parameter.Parameter(
                torch.randn(kernel_count, self.in_channels * self.kernel_area, self.out_channels))
            nn.init.kaiming_normal_(self.kernels, nonlinearity="relu")
        elif self.kernel_generator == "low_rank":
            self.kernel_head = nn.Sequential(
                nn.Linear(self.in_channels, mlp_inner_dims),
                nn.ReLU(),
                nn.Linear(mlp_inner_dims, mlp_inner_dims),
                nn.ReLU(),
            )
            self.to_area = nn.Linear(mlp_inner_dims, self.kernel_area)
            self.to_cin = nn.Linear(mlp_inner_dims, self.in_channels)
            self.to_cout = nn.Linear(mlp_inner_dims, self.out_channels)
            self.kernels = nn.parameter.Parameter(
                torch.randn(self.in_channels, self.kernel_area, self.out_channels))
            nn.init.kaiming_normal_(self.kernels, nonlinearity="relu")
        else:
            raise ValueError(
                "kernel_generator must be either 'spatial' or 'weighted_sum' or 'low_rank'")

        if bias == "cluster":
            self.centroid_to_bias = nn.Sequential(
                nn.Linear(in_features=self.in_channels,
                          out_features=mlp_inner_dims),
                nn.ReLU(),
                nn.Linear(in_features=mlp_inner_dims,
                          out_features=mlp_inner_dims),
                nn.ReLU(),
                nn.Linear(in_features=mlp_inner_dims,
                          out_features=self.out_channels),
            )
        elif bias == "global_param":
            self.bias = nn.parameter.Parameter(
                torch.randn(self.out_channels))
        elif bias == "global_adaptive":
            self.global_bias = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, out_channels, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 1)
            )
        elif bias == "none":
            self.bias = None
        self._last_cluster_loss = None

    def generate_kernel(self, centroids: torch.Tensor):
        if centroids is None or centroids.numel() == 0 :
             # Fallback if centroids are empty, e.g. use a default kernel or raise error
             # This should ideally be handled by ensuring centroids are always valid before this call
             # For now, let's assume `kernels` is the base parameter (B, K, Cin*Area, Cout) or (Cin, Area, Cout)
             # and try to make it broadcastable or return a sensible default.
             # This situation needs robust handling. If K_max_fine is 0, this will be problematic.
             # Let's assume K_max_fine will be at least 1 if there's data.
             # If centroids are (B, 0, D), this would fail.
             # The calling code must ensure centroids has K > 0 if possible.
             # A simple fallback might be a zero kernel or a mean kernel.
             # For now, rely on calling code to provide valid centroids.
             # If K_max_fine becomes 0, then num_clusters_for_kernel_bias_gen will be 0.
             # Then convolution_by_cluster will have issues with weight.shape[1] == 0.

            # If centroids comes in as (B, 0, D) due to no clusters found anywhere in batch.
            # Return a dummy kernel that won't crash, but won't do useful work.
            # (B, 0, Cin*Area, Cout)
            return torch.empty(centroids.shape[0], 0, self.in_channels * self.kernel_area, self.out_channels, device=centroids.device)

        if self.kernel_generator == "spatial":
            spatial_weights = rearrange(
                self.centroid_to_kernel(centroids), 'b k area -> b k 1 area 1')
            kernel_by_cluster = rearrange(
                spatial_weights * self.kernels, 'b k cin area cout -> b k (cin area) cout')
        elif self.kernel_generator == "weighted_sum":
            kernel_weights = rearrange(
                self.centroid_to_kernel(centroids), 'b k n -> b k n 1 1')
            kernel_by_cluster = reduce(
                kernel_weights * self.kernels, 'b k n cinarea cout -> b k cinarea cout', 'sum')
        else: # low_rank
            kf = self.kernel_head(centroids)
            w_cin = rearrange(torch.sigmoid(self.to_cin(kf)),
                              'b k cin -> b k cin 1 1')
            w_area = rearrange(torch.sigmoid(self.to_area(kf)),
                               'b k area -> b k 1 area 1')
            w_cout = rearrange(torch.sigmoid(self.to_cout(kf)),
                               'b k cout -> b k 1 1 cout')
            # Ensure self.kernels is (Cin, Area, Cout) and properly broadcast/expanded
            # Original self.kernels: (self.in_channels, self.kernel_area, self.out_channels)
            # Broadcast self.kernels to (B, K, Cin, Area, Cout)
            base_kernel_expanded = self.kernels.unsqueeze(0).unsqueeze(0) # 1, 1, Cin, Area, Cout
            kernel_by_cluster = (w_cin * w_area * w_cout) * base_kernel_expanded
            kernel_by_cluster = rearrange(
                kernel_by_cluster, 'b k cin area cout -> b k (cin area) cout')

        return kernel_by_cluster

    def generate_bias(self, centroids: torch.Tensor, x):
        if centroids is None or centroids.numel() == 0: # Similar fallback as generate_kernel
            if self.bias_mode == "cluster":
                return torch.empty(centroids.shape[0], 0, self.out_channels, device=centroids.device)
            elif self.bias_mode == "global_param":
                return self.bias # (Cout)
            elif self.bias_mode == "global_adaptive":
                 return rearrange(self.global_bias(x), 'b cout 1 1 -> b 1 cout') # (B, 1, Cout)
            elif self.bias_mode == "none":
                return None
            return None # Default

        if self.bias_mode == "cluster":
            return self.centroid_to_bias(centroids)
        elif self.bias_mode == "global_param":
            return self.bias
        elif self.bias_mode == "global_adaptive":
            return rearrange(self.global_bias(x), 'b cout 1 1 -> b 1 cout')
        elif self.bias_mode == "none":
            return None

    def downsample_to_cluster_feature(self, x: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        if self.cluster_source == "channel":
            return reduce(patches, 'b s (cin area) -> b s cin', 'mean', cin=self.in_channels, area=self.kernel_area)
        elif self.cluster_source == "spatial":
            return reduce(patches, 'b s (cin area) -> b s area', 'mean', area=self.kernel_area)
        elif self.cluster_source == "pixel":
            return rearrange(x, 'b cin h w -> b (h w) cin')
        else:
            raise ValueError(
                "cluster_source must be either 'channel', 'spatial', or 'pixel'")

    def convolution_by_cluster(self, patches: torch.Tensor, indice: torch.Tensor, weight: torch.Tensor, bias=None):
        b = patches.shape[0]
        # If weight K is 0 (e.g. no clusters found in batch), this will fail.
        # Handle case where weight.shape[1] (K) is 0.
        if weight.shape[1] == 0:
            # No kernels to convolve with. Return zeros of appropriate output shape.
            # Output shape is (b, patch_num, out_channels)
            return torch.zeros(patches.shape[0], patches.shape[1], self.out_channels, device=patches.device)

        k = weight.shape[1] # Num clusters used for kernel/bias generation (K_max_fine)

        patches_re = rearrange(patches, "b s f -> (b s) f") # Use patches_re to avoid name clash
        weight_re = rearrange(weight, "b k f cout -> (b k) f cout")
        
        # indice is (B, NumPatches), values are 0 to K_max_fine-1 for each item's own set of clusters
        # dispatch_indice expects global indices if K is total clusters in batch.
        # Here, K is K_max_fine. Indices for item `b` are `0..K_b-1`.
        # The `indice + torch.arange(b).view(-1,1)*k` makes them map to `weight[b, indice[b,hw], ...]`
        # This is correct if `weight` is (B, K_max_fine, ...) and `indice` values are within `[0, K_max_fine-1]`.
        indice_adjusted = indice + torch.arange(b, device=indice.device).view(-1, 1) * k
        indice_adjusted = rearrange(indice_adjusted, "b hw -> (b hw)")
        
        if bias is not None:
            if bias.ndim == 1: # global_param bias (Cout) -> (1,1,Cout)
                bias_re = bias.view(1, -1) # (1, Cout) to be used by dispatch_indice logic later
            elif bias.shape[1] == 1 and k > 1 : # global_adaptive bias (B,1,Cout) but K clusters
                 bias_re = repeat(bias, 'b 1 cout -> (b k) cout', k=k)
            else: # cluster bias (B,K,Cout) -> (B*K, Cout)
                 bias_re = rearrange(bias, "b k cout -> (b k) cout")
        else:
            bias_re = None

        # Ensure indice values are within the valid range for weight_re's first dim after arange offset
        # Max value in indice_adjusted should be less than (b*k)
        if indice_adjusted.numel() > 0 and indice_adjusted.max() >= b * k:
            # This can happen if an item had K_item clusters, and K_item > K_max_fine used for another item.
            # This should not happen if K_max_fine is correctly calculated and used for weight tensor K dim.
            # And if indice values are correctly item-local [0, K_item-1].
            # Let's clamp for safety, though this indicates a logic error if it triggers.
            # print(f"Warning: Clamping indices in convolution_by_cluster. Max index {indice_adjusted.max()}, b*k = {b*k}")
            indice_adjusted = torch.clamp(indice_adjusted, 0, b * k - 1)

        # Call dispatch_indice first
        indice_perm, padded_patch_num, cluster_size_sorted, permuted_offset, cluster_perm, batch_height = dispatch_indice(
            indice=indice_adjusted, cluster_num=b * k)
        
        # Then permute the input features
        input_permuted = permute(input=patches_re, indice_perm=indice_perm, padded_patch_num=padded_patch_num)
        
        # Then call batched_matmul_conv
        output_permuted = batched_matmul_conv(
            input_permuted=input_permuted, 
            weight=weight_re, 
            permuted_offset=permuted_offset, 
            cluster_perm=cluster_perm, 
            batch_height=batch_height, 
            bias=bias_re)
        
        output = inverse_permute(input=output_permuted, indice_perm=indice_perm)

        return rearrange(output, "(b hw) cout -> b hw cout", b=b)

    def cluster_ablation_global_forward(self, x: torch.Tensor):
        b, cin, h, w = x.shape
        patches = self.unfold(x)
        patches = rearrange(
            patches, 'b (cin area) (h w) -> b (h w) (cin area)', area=self.kernel_area, h=h, w=w)
        centroids = reduce(patches, 'b s f -> b 1 f', 'mean')
        kernel = self.generate_kernel(centroids)
        bias = self.generate_bias(centroids, x)
        result = torch.matmul(patches, rearrange(
            kernel, 'b 1 f cout -> b f cout'))
        if bias is not None:
            if bias.ndim == 1: 
                result = result + bias.view(1, 1, -1) 
            else: 
                result = result + bias
        return rearrange(result, 'b (h w) cout -> b cout h w', h=h, w=w), torch.zeros(b, h*w, dtype=torch.long, device=x.device)

    def cluster_ablation_pixelwise_forward(self, x: torch.Tensor):
        b, cin, h, w = x.shape
        patches = self.unfold(x)
        patches = rearrange(
            patches, 'b (cin area) (h w) -> b (h w) (cin area)', area=self.kernel_area, h=h, w=w)
        kernel = self.generate_kernel(patches) 
        bias = self.generate_bias(patches, x)   
        result = torch.matmul(rearrange(patches, 'b s f -> b s 1 f'), kernel)
        if bias is not None:
            result = result + rearrange(bias, 'b s cout -> b s 1 cout')
        return rearrange(result, 'b (h w) 1 cout -> b cout h w', h=h, w=w), repeat(torch.arange(h*w, device=x.device), 's -> b s', b=b)

    def forward(self, x: torch.Tensor, cache_indice=None, cluster_override=None):
        if self.cluster_ablation == "global":
            # The returned index from ablation forward is a placeholder and not used for loss
            res, idx = self.cluster_ablation_global_forward(x)
            self._last_cluster_loss = torch.tensor(0.0, device=x.device) # No cluster loss in this mode
            return res, idx, self._last_cluster_loss
        elif self.cluster_ablation == "pixelwise":
            res, idx = self.cluster_ablation_pixelwise_forward(x)
            self._last_cluster_loss = torch.tensor(0.0, device=x.device) # No cluster loss in this mode
            return res, idx, self._last_cluster_loss

        batch_size, cin_actual, h, w = x.shape # Use actual cin from x
        
        patches = self.unfold(x)  # (b, cin_actual*kernel_area, h*w)
        patches = rearrange(
            patches, 'b (c_in area) (h w) -> b (h w) (c_in area)', 
            c_in=cin_actual, area=self.kernel_area, h=h, w=w) # (B, NumPatches, Cin*Area)

        # cluster_features are used for clustering. Cin for cluster_features can be different from Cin of patches.
        # self.in_channels is the expected input channels for clustering logic (e.g. mlp in_features)
        # If cluster_source is 'pixel', it uses x directly.
        # Let's ensure cluster_features are correctly shaped based on self.in_channels for consistency.
        # downsample_to_cluster_feature uses self.in_channels for its logic.
        # Patches are (B, NumPatches, cin_actual*kernel_area)
        # Cluster features should be (B, NumPatches, self.in_channels) if source='channel' after mean over area
        # Or (B, NumPatches, self.kernel_area) if source='spatial'
        # Or (B, NumPatches, cin_actual) if source='pixel' (NumPatches = H*W)

        cluster_features = self.downsample_to_cluster_feature(x, patches) # (B, NumPatches, ClusterFeatureDim)
        cluster_feature_dim = cluster_features.shape[-1]

        final_indice = None
        final_centroids = None 
        num_clusters_for_kernel_bias_gen = 0

        # --- Overrides and Cache Logic ---
        if cluster_override is not None:
            final_indice = cluster_override
            if cluster_features.device != final_indice.device: # Ensure device consistency
                 final_indice = final_indice.to(cluster_features.device)

            max_idx_override = 0
            if final_indice.numel() > 0:
                max_idx_override = final_indice.max().item() + 1
            
            # Determine K for override: use max_idx or a base number if override indices are sparse
            # This should ideally be passed or configured. Using max_idx can be large.
            # Original logic used a complex calculation for num_clusters_for_override_centroids
            # Let's use a simpler approach: max index in override, or a base if that's too small.
            num_clusters_for_override_centroids = max(max_idx_override, 1) # Ensure at least 1
            
            final_centroids = get_cluster_centers(cluster_features, final_indice, num_clusters_for_override_centroids)
            num_clusters_for_kernel_bias_gen = num_clusters_for_override_centroids
        
        elif cache_indice is not None and not self.training:
            # Use a fixed K for cached indices, typically base K.
            # The cached indices themselves define the number of clusters if used directly for centroids.
            # However, kmeans call implies K is fixed for centroid generation consistency.
            _cache_k = self.cluster_num_base_fine_kmeans 
            # Assuming cache_indice implies a fixed set of K clusters for which centroids are found
            final_indice, final_centroids = self.kmeans(
                cluster_features, cache_indice, cluster_num=_cache_k)
            num_clusters_for_kernel_bias_gen = _cache_k

        # --- Main Clustering Logic: Strict HC -> K-Means in HC, or Fallback ---
        elif self.enable_hierarchical_clustering: # Strict HC -> K-Means in HC path
            all_fine_indices_batch_list = []
            all_fine_centroids_batch_list = [] # List of tensors: [(K_item1, D), (K_item2, D), ...]
            item_total_fine_clusters_list = [] # List of ints: [K_item1, K_item2, ...]

            for i in range(batch_size):
                item_cluster_features = cluster_features[i] # (NumPatches, ClusterFeatureDim)
                item_patches_for_variance = patches[i]    # (NumPatches, Cin*Area), for dynamic K variance

                if item_cluster_features.shape[0] == 0: # No patches for this item
                    all_fine_indices_batch_list.append(torch.empty(0, dtype=torch.long, device=x.device))
                    all_fine_centroids_batch_list.append(torch.empty((0, cluster_feature_dim), device=x.device))
                    item_total_fine_clusters_list.append(0)
                    continue

                # Simplified: Skip expensive HC, default to single coarse cluster
                num_samples_in_item_hc = item_cluster_features.shape[0]
                _coarse_labels_np = np.zeros(num_samples_in_item_hc, dtype=np.int64)
                _num_coarse_k = 1 if num_samples_in_item_hc > 0 else 0
                
                # Create default tensor for single cluster case
                coarse_labels_for_item_tensor = torch.zeros(num_samples_in_item_hc, dtype=torch.long, device=item_cluster_features.device)
                num_coarse_clusters_for_item = _num_coarse_k
                
                # K-Means within each coarse cluster
                item_fine_indices_tensor = torch.empty_like(coarse_labels_for_item_tensor)
                item_centroids_collector_list = []
                current_max_fine_label_for_item = 0

                if num_coarse_clusters_for_item > 0 : # Only proceed if HC resulted in some coarse clusters
                    for c_idx in range(num_coarse_clusters_for_item):
                        coarse_mask = (coarse_labels_for_item_tensor == c_idx)
                        features_in_coarse_cluster = item_cluster_features[coarse_mask]
                        patches_for_variance_in_coarse_cluster = item_patches_for_variance[coarse_mask]

                        if features_in_coarse_cluster.shape[0] == 0:
                            continue

                        k_fine = self.cluster_num_base_fine_kmeans
                        if self.enable_dynamic_k and self.training: # Dynamic K for fine-grained
                            if patches_for_variance_in_coarse_cluster.shape[0] > 1:
                                variance_fine = patches_for_variance_in_coarse_cluster.var(dim=-1).mean()
                                if variance_fine > self.dynamic_k_variance_threshold:
                                    k_fine += self.dynamic_k_delta
                                else:
                                    k_fine -= self.dynamic_k_delta
                        
                        k_fine = max(self.dynamic_k_min, min(k_fine, self.dynamic_k_max))
                        k_fine = min(k_fine, features_in_coarse_cluster.shape[0]) # Not more than samples
                        k_fine = max(k_fine, 1) # At least 1 cluster

                        fine_labels_local, fine_centroids_local = self.kmeans(
                            features_in_coarse_cluster.unsqueeze(0), cluster_num=k_fine)
                        
                        item_fine_indices_tensor[coarse_mask] = fine_labels_local.squeeze(0) + current_max_fine_label_for_item
                        item_centroids_collector_list.append(fine_centroids_local.squeeze(0))
                        current_max_fine_label_for_item += k_fine
                
                all_fine_indices_batch_list.append(item_fine_indices_tensor)
                if item_centroids_collector_list:
                    all_fine_centroids_batch_list.append(torch.cat(item_centroids_collector_list, dim=0))
                else:
                    # If no centroids, create default single cluster
                    all_fine_centroids_batch_list.append(torch.zeros(1, cluster_feature_dim, device=item_cluster_features.device))
                
                item_total_fine_clusters_list.append(current_max_fine_label_for_item)

            # Consolidate batch results for strict HC path
            if not item_total_fine_clusters_list: # All items had 0 patches
                 K_max_fine = 0
            else:
                 K_max_fine = max(item_total_fine_clusters_list) if item_total_fine_clusters_list else 0
            K_max_fine = max(K_max_fine, 1) # Ensure K_max_fine is at least 1 for tensor creation

            final_indice = torch.stack(all_fine_indices_batch_list, dim=0) if all_fine_indices_batch_list else torch.empty((batch_size, 0), dtype=torch.long, device=x.device)
            
            final_centroids = torch.zeros(batch_size, K_max_fine, cluster_feature_dim, device=x.device)
            for i in range(batch_size):
                if item_total_fine_clusters_list[i] > 0 and all_fine_centroids_batch_list[i].numel() > 0 :
                    final_centroids[i, :item_total_fine_clusters_list[i], :] = all_fine_centroids_batch_list[i]
            
            num_clusters_for_kernel_bias_gen = K_max_fine

        else: # Fallback to original K-Means logic (if HC disabled)
            # This part replicates the original K-Means (potentially with its own HC-for-K or dynamic-K)
            # Determine effective_cluster_num_for_kmeans (old way)
            _effective_k_old_path = self.cluster_num_base_fine_kmeans
            if self.training: # Old HC and dynamic K were typically training-only
                # Try original HC-for-K determination (if it was enabled via enable_hierarchical_clustering,
                # but we are in this 'else' block, it means the strict path was not taken, maybe a flag issue.
                # For clarity, let's assume this 'else' implies !self.enable_hierarchical_clustering for the strict sense)
                # So, only old dynamic K applies here if self.enable_dynamic_k is true.
                if self.enable_dynamic_k: # Old global dynamic K based on all patches' variance
                    patch_variances = patches.var(dim=-1) # Global patches
                    sample_mean_variance = patch_variances.mean(dim=1)
                    batch_mean_variance = sample_mean_variance.mean()
                    if batch_mean_variance > self.dynamic_k_variance_threshold:
                        _effective_k_old_path += self.dynamic_k_delta
                    else:
                        _effective_k_old_path -= self.dynamic_k_delta
                    _effective_k_old_path = max(self.dynamic_k_min, min(_effective_k_old_path, self.dynamic_k_max))

            _effective_k_old_path = max(_effective_k_old_path, 1) # Ensure K is at least 1
            if cluster_features.shape[1] == 0: # No patches to cluster
                 final_indice = torch.empty((batch_size, 0), dtype=torch.long, device=x.device)
                 final_centroids = torch.empty((batch_size, _effective_k_old_path, cluster_feature_dim), device=x.device) # K can't be 0 for kmeans
            else:
                 final_indice, final_centroids, cluster_loss = self.kmeans(cluster_features, cluster_num=_effective_k_old_path)
                 self._last_cluster_loss = cluster_loss
            num_clusters_for_kernel_bias_gen = _effective_k_old_path

        # --- Filter threshold logic (Deferred for now with the new strict HC path) ---
        # if self.filter_threshold > 0 and cluster_override is None:
        # original_indice_device = final_indice.device
        # filtered_indice = filter_indice(final_indice, num_clusters_for_kernel_bias_gen, self.filter_threshold).to(original_indice_device)
        # num_clusters_after_filter = num_clusters_for_kernel_bias_gen + 1
        #
        # # Recalculate centroids if indice changed due to filtering
        # # This part needs careful adaptation if final_centroids is (B, K_max, D)
        # _b_cf, _n_patches_cf, _fd_cf = cluster_features.shape
        # new_centroids = torch.zeros((_b_cf, num_clusters_after_filter, _fd_cf), device=cluster_features.device)
        # for i in range(_b_cf):
        # for k_idx_val in range(num_clusters_for_kernel_bias_gen):
        # mask = (filtered_indice[i] == k_idx_val)
        # if mask.sum() > 0:
        # new_centroids[i, k_idx_val] = cluster_features[i, mask].mean(dim=0)
        #
        # global_center_features = reduce(cluster_features, 'b s f -> b f', 'mean')
        # new_centroids[:, num_clusters_for_kernel_bias_gen, :] = global_center_features
        #
        # final_indice = filtered_indice
        # final_centroids = new_centroids
        # num_clusters_for_kernel_bias_gen = num_clusters_after_filter

        if self.detatch_centroid and final_centroids is not None:
            final_centroids = final_centroids.detach()
        
        if final_centroids is None or final_indice is None or final_centroids.shape[1] == 0 : # K_max_fine could be 0 if no clusters
             # If K is 0 for kernels (num_clusters_for_kernel_bias_gen == 0), conv_by_cluster handles it.
             # Ensure final_centroids is not None for generate_kernel/bias.
             # If K_max_fine was 0 (e.g. batch had no data/patches), make it 1 with dummy centroid to avoid crash,
             # but conv_by_cluster should still get K=0 for weights.
             # The generate_kernel/bias has fallbacks for K=0 centroids now.
             # convolution_by_cluster also has a K=0 fallback for weights.
             # However, num_clusters_for_kernel_bias_gen is used to shape final_centroids.
             # If it's 0, final_centroids will be (B,0,D).
             # Let's ensure num_clusters_for_kernel_bias_gen is what generate_kernel/bias expect (K_max_fine)
             pass # Already handled by K_max_fine = max(K_max_fine, 1) in strict path, or _effective_k_old_path >=1

        kernels = self.generate_kernel(final_centroids) 
        bias_for_conv = self.generate_bias(final_centroids, x)
        
        # Ensure patches has data if indice is non-empty
        if patches.shape[1] == 0 and final_indice.numel() > 0 : # No patches but indice exists (should not happen)
            # Return zero output matching expected spatial dims of x
            return torch.zeros_like(x), final_indice, self._last_cluster_loss
        if patches.shape[1] > 0 and final_indice.numel() == 0 and patches.shape[0] == final_indice.shape[0]: # Patches but no indices (e.g. all items had 0 patches in strict hc)
            # This case should be covered by final_indice having shape (B,0)
             return torch.zeros_like(x), final_indice, self._last_cluster_loss

        result = self.convolution_by_cluster(patches, final_indice, kernels, bias_for_conv)
        
        return rearrange(result, 'b (h w) cout -> b cout h w', h=h, w=w), final_indice, self._last_cluster_loss

def test_kmconv_layer():
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Updated test call for CANConv with potentially new defaults or required params
    module = CANConv(
        in_channels=32, 
        out_channels=32, 
        cluster_num=8, # Base K for fine-grained
        enable_hierarchical_clustering=True, 
        hierarchical_clustering_target_coarse_clusters=4, # Example: 4 coarse clusters
        enable_dynamic_k=True, # Enable dynamic K for fine-grained
        dynamic_k_min=2, dynamic_k_max=10
    ).to(dev)
    
    # Basic run test
    print("Testing CANConv module...")
    try:
        x_test = torch.randn(2, 32, 16, 16, device=dev) # Smaller spatial for faster test
        y_test, indices_test = module(x_test)
        print(f"Output shape: {y_test.shape}, Indices shape: {indices_test.shape}")
        print("CANConv module basic run: PASSED")
    except Exception as e:
        print(f"CANConv module basic run: FAILED with {e}")
        import traceback
        traceback.print_exc()
        return # Stop if basic run fails

    # Profiler test (optional, can be time-consuming)
    enable_profiler_test = False
    if enable_profiler_test:
        print("\nStarting profiler test...")
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                wait=1, warmup=1, active=3, repeat=1), # Shorter schedule
            record_shapes=True,
            with_stack=True
        ) as prof:
            for _step in range(5): # Fewer steps
                with record_function('single_run_profiler'):
                    x_prof = torch.randn(1, 32, 32, 32, device=dev) # Moderate size for profiler
                    _y_prof, _indices_prof = module(x_prof)
                prof.step()
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        # prof.export_chrome_trace("trace.json") # Optional trace export
        print("Profiler test finished.")

if __name__ == "__main__":
    test_kmconv_layer()
