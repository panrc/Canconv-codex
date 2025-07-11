
import torch
import torch.nn as nn
import torch.nn.functional as F
from .sam_loss import SAMLoss

class CombinedLoss(nn.Module):
    def __init__(self, base_loss='l1', alpha=0.1, beta=0.05, reduction='mean'):
        super(CombinedLoss, self).__init__()
        if base_loss == 'l1':
            self.recon_loss = nn.L1Loss(reduction=reduction)
        elif base_loss == 'l2':
            self.recon_loss = nn.MSELoss(reduction=reduction)
        else:
            raise ValueError(f"Unsupported base loss: {base_loss}")
        self.alpha = alpha
        self.beta = beta
        self.sam_loss = SAMLoss()

    def forward(self, pred, target, model_outputs):
        # Reconstruction Loss
        recon_loss = self.recon_loss(pred, target)

        # Clustering Loss
        cluster_loss = model_outputs.get('cluster_loss', torch.tensor(0.0, device=pred.device))

        # Spatial-Spectral Decoupling Loss
        decouple_loss = self.sam_loss(pred, target)

        # Combined Loss
        total_loss = recon_loss + self.alpha * cluster_loss + self.beta * decouple_loss
        
        return total_loss, {'total_loss': total_loss, 'recon_loss': recon_loss, 'cluster_loss': cluster_loss, 'decouple_loss': decouple_loss}
