import torch
import torch.nn as nn

from .model import CANNet
from canconv.losses.combined_loss import CombinedLoss

from canconv.util.trainer import SimplePanTrainer
from canconv.layers.kmeans import reset_cache, KMeansCacheScheduler


class CANNetTrainer(SimplePanTrainer):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

    def _create_model(self, cfg):
        loss_type = cfg["loss"].lower()
        if loss_type == "l1":
            self.criterion = nn.L1Loss(reduction='mean').to(self.dev)
        elif loss_type in ("l2", "mse"):
            self.criterion = nn.MSELoss(reduction='mean').to(self.dev)
        elif loss_type == "sam":
            from canconv.losses.sam_loss import SAMLoss
            self.criterion = SAMLoss().to(self.dev)
        elif loss_type == "combined":
            self.criterion = CombinedLoss(
                base_loss=cfg.get("base_rec_loss", "l1"), 
                alpha=cfg.get("alpha_cluster_loss", 0.1),
                beta=cfg.get("beta_decouple_loss", 0.05)
            ).to(self.dev)
        else:
            raise NotImplementedError(f"Loss {cfg['loss']} not implemented")
        self.model = CANNet(
            spectral_num=cfg['spectral_num'], 
            channels=cfg['channels'], 
            cluster_num=cfg['cluster_num'], 
            filter_threshold=cfg["filter_threshold"],
            
            shallow_enable_dynamic_k=False,
            shallow_dynamic_k_variance_threshold=cfg.get('shallow_dynamic_k_variance_threshold', 0.1),
            shallow_dynamic_k_delta=cfg.get('shallow_dynamic_k_delta', 5),
            shallow_dynamic_k_min=cfg.get('shallow_dynamic_k_min', 2),
            shallow_dynamic_k_max=cfg.get('shallow_dynamic_k_max', 32),
            shallow_enable_hierarchical_clustering=False,
            shallow_hierarchical_clustering_target_coarse_clusters=cfg.get('shallow_hierarchical_clustering_target_coarse_clusters', 0),
            shallow_hierarchical_clustering_linkage=cfg.get('shallow_hierarchical_clustering_linkage', 'ward'),
            shallow_hc_distance_threshold=cfg.get('shallow_hc_distance_threshold'),
            shallow_hc_auto_k_method=cfg.get('shallow_hc_auto_k_method'),
            shallow_hc_auto_k_param=cfg.get('shallow_hc_auto_k_param', 0.75),

            deep_enable_dynamic_k=False,
            deep_dynamic_k_variance_threshold=cfg.get('deep_dynamic_k_variance_threshold', 0.1),
            deep_dynamic_k_delta=cfg.get('deep_dynamic_k_delta', 5),
            deep_dynamic_k_min=cfg.get('deep_dynamic_k_min', 2),
            deep_dynamic_k_max=cfg.get('deep_dynamic_k_max', 32),
            deep_enable_hierarchical_clustering=False,
            deep_hierarchical_clustering_target_coarse_clusters=cfg.get('deep_hierarchical_clustering_target_coarse_clusters', 0),
            deep_hierarchical_clustering_linkage=cfg.get('deep_hierarchical_clustering_linkage', 'ward'),
            deep_hc_distance_threshold=cfg.get('deep_hc_distance_threshold'),
            deep_hc_auto_k_method=cfg.get('deep_hc_auto_k_method'),
            deep_hc_auto_k_param=cfg.get('deep_hc_auto_k_param', 0.75)
        ).to(self.dev)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=cfg["learning_rate"], weight_decay=0)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=cfg["lr_step_size"])

        self.km_scheduler = KMeansCacheScheduler(cfg['kmeans_cache_update'])

    def _on_train_start(self):
        reset_cache(len(self.train_dataset))

    def _on_epoch_start(self, epoch):
        self.km_scheduler.step()

    def forward(self, data):
        pan_image = data['pan'].to(self.dev)
        lms_image = data['lms'].to(self.dev)
        
        if "index" in data and self.model.training:
            cache_indice = data['index'].to(self.dev)
            return self.model(pan_image, lms_image, cache_indice)
        else:
            output, _ = self.model(pan_image, lms_image)
            return output
