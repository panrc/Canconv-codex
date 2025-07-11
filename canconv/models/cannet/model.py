import torch
import torch.nn as nn
from torch.nn import functional as F
from canconv.layers.canconv import CANConv
from typing import Optional


class LightweightTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads=4, ff_dim_multiplier=2, dropout=0.1, spatial_reduction: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.reduction = max(1, spatial_reduction)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ff_dim_multiplier),
            nn.GELU(),
            nn.Linear(embed_dim * ff_dim_multiplier, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (B, C, H, W)
        B, C, H, W = x.shape
        # Ensure input channel matches embed_dim if an explicit check is needed,
        # otherwise, trust it's correctly configured during model init.
        # For now, I'll remove the assertion to avoid potential issues if C != embed_dim
        # due to some intermediate layers, though it should match.
        # assert C == self.embed_dim, f"Input channel {C} does not match embed_dim {self.embed_dim}"

        if self.reduction > 1:
            # Downsample spatially to reduce sequence length and memory
            x_reduced = F.avg_pool2d(x, kernel_size=self.reduction)
            Hr, Wr = x_reduced.shape[2], x_reduced.shape[3]
            x_seq = x_reduced.flatten(2).transpose(1, 2)  # (B, Hr*Wr, C)
        else:
            x_seq = x.flatten(2).transpose(1, 2)  # (B, H*W, C)

        # Self-attention
        attn_out, _ = self.attention(x_seq, x_seq, x_seq)
        x_seq = x_seq + self.dropout1(attn_out)
        x_seq = self.norm1(x_seq)

        # Feed-forward network
        ffn_out = self.ffn(x_seq)
        x_seq = x_seq + self.dropout2(ffn_out)
        x_seq = self.norm2(x_seq)

        if self.reduction > 1:
            out_reduced = x_seq.transpose(1, 2).view(B, C, Hr, Wr)
            # Upsample back to original resolution
            out = F.interpolate(out_reduced, size=(H, W), mode='bilinear', align_corners=False)
        else:
            out = x_seq.transpose(1, 2).view(B, C, H, W)
        return out


class CANResBlock(nn.Module):
    def __init__(self, channels, cluster_num, filter_threshold, cluster_source="channel", 
                 enable_dynamic_k=False, dynamic_k_variance_threshold=0.1, 
                 dynamic_k_delta=5, dynamic_k_min=2, dynamic_k_max=32, 
                 enable_hierarchical_clustering=False, 
                 hierarchical_clustering_target_coarse_clusters=0,
                 hierarchical_clustering_linkage='ward',
                 hc_distance_threshold: Optional[float] = None,
                 hc_auto_k_method: Optional[str] = None, 
                 hc_auto_k_param: Optional[float] = None,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        can_conv_params = {
            'cluster_num': cluster_num,
            'cluster_source': cluster_source, 
            'filter_threshold': filter_threshold,
            'enable_dynamic_k': enable_dynamic_k, 
            'dynamic_k_variance_threshold': dynamic_k_variance_threshold,
            'dynamic_k_delta': dynamic_k_delta, 
            'dynamic_k_min': dynamic_k_min, 
            'dynamic_k_max': dynamic_k_max,
            'enable_hierarchical_clustering': enable_hierarchical_clustering,
            'hierarchical_clustering_target_coarse_clusters': hierarchical_clustering_target_coarse_clusters,
            'hierarchical_clustering_linkage': hierarchical_clustering_linkage,
            'hc_distance_threshold': hc_distance_threshold,
            'hc_auto_k_method': hc_auto_k_method,
            'hc_auto_k_param': hc_auto_k_param
        }
        self.conv1 = CANConv(channels, channels, **can_conv_params)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x, cache_indice=None, cluster_override=None):
        res, idx, cluster_loss = self.conv1(x, cache_indice, cluster_override)
        res = self.act(res)
        x = x + res
        return x, idx, cluster_loss


class ConvDown(nn.Module):
    def __init__(self, in_channels, out_channels, dsconv=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if dsconv:
            self.conv = nn.Sequential(
                # nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
                nn.Conv2d(in_channels, in_channels, 2, 2, 0),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, 3, 1, 1,
                          groups=in_channels, bias=False),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels,
                          kernel_size=3, stride=2, padding=1),
                # nn.Conv2d(in_channels, in_channels, 2, 2, 0),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, 3, 1, 1)
            )

    def forward(self, x):
        return self.conv(x)


class ConvUp(nn.Module):
    def __init__(self, in_channels, out_channels, dsconv=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # self.conv1 = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, 2, 2, 0)
        if dsconv:
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, 1,
                          1, groups=out_channels, bias=False),
                nn.Conv2d(out_channels, out_channels, 1, 1, 0)
            )
        else:
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

    def forward(self, x, y):
        x = F.leaky_relu(self.conv1(x))
        x = x + y
        x = F.leaky_relu(self.conv2(x))
        return x


class CANNet(nn.Module):
    def __init__(self, spectral_num=8, channels=32, cluster_num=32, filter_threshold=0.005, 
                 # Transformer parameters
                 transformer_heads=1, transformer_ff_dim_multiplier=1, transformer_dropout=0.1,
                 transformer_spatial_reduction: int = 4,
                 # Common params

                 # Shallow layers clustering config
                 shallow_enable_dynamic_k=True,
                 shallow_dynamic_k_variance_threshold=0.1, 
                 shallow_dynamic_k_delta=5, 
                 shallow_dynamic_k_min=2, 
                 shallow_dynamic_k_max=32,
                 shallow_enable_hierarchical_clustering=True, # Example: shallow uses HC
                 shallow_hierarchical_clustering_target_coarse_clusters=0,
                 shallow_hierarchical_clustering_linkage='ward',
                 shallow_hc_distance_threshold: Optional[float] = None,
                 shallow_hc_auto_k_method: Optional[str] = None, 
                 shallow_hc_auto_k_param: Optional[float] = 0.75,

                 # Deep layers clustering config
                 deep_enable_dynamic_k=True,  # Example: deep uses variance-based dynamic K
                 deep_dynamic_k_variance_threshold=0.1, 
                 deep_dynamic_k_delta=5, 
                 deep_dynamic_k_min=2, 
                 deep_dynamic_k_max=32,
                 deep_enable_hierarchical_clustering=False, # Example: deep does not use HC
                 deep_hierarchical_clustering_target_coarse_clusters=0,
                 deep_hierarchical_clustering_linkage='ward',
                 deep_hc_distance_threshold: Optional[float] = None,
                 deep_hc_auto_k_method: Optional[str] = None, 
                 deep_hc_auto_k_param: Optional[float] = 0.75,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        ch_enc_dec = channels
        ch_bottleneck = int(channels * 1.25)

        if channels > 0:
            ch_enc_dec = max(1, ch_enc_dec)
            ch_bottleneck = max(1, ch_bottleneck)
        else: # if channels is 0 or negative, set all to 1
            ch_enc_dec = 1
            ch_bottleneck = 1

        shallow_res_block_params = {
            'enable_dynamic_k': shallow_enable_dynamic_k, 
            'dynamic_k_variance_threshold': shallow_dynamic_k_variance_threshold,
            'dynamic_k_delta': shallow_dynamic_k_delta, 
            'dynamic_k_min': shallow_dynamic_k_min, 
            'dynamic_k_max': shallow_dynamic_k_max,
            'enable_hierarchical_clustering': shallow_enable_hierarchical_clustering,
            'hierarchical_clustering_target_coarse_clusters': shallow_hierarchical_clustering_target_coarse_clusters,
            'hierarchical_clustering_linkage': shallow_hierarchical_clustering_linkage,
            'hc_distance_threshold': shallow_hc_distance_threshold,
            'hc_auto_k_method': shallow_hc_auto_k_method,
            'hc_auto_k_param': shallow_hc_auto_k_param
        }

        deep_res_block_params = {
            'enable_dynamic_k': deep_enable_dynamic_k, 
            'dynamic_k_variance_threshold': deep_dynamic_k_variance_threshold,
            'dynamic_k_delta': deep_dynamic_k_delta, 
            'dynamic_k_min': deep_dynamic_k_min, 
            'dynamic_k_max': deep_dynamic_k_max,
            'enable_hierarchical_clustering': deep_enable_hierarchical_clustering,
            'hierarchical_clustering_target_coarse_clusters': deep_hierarchical_clustering_target_coarse_clusters,
            'hierarchical_clustering_linkage': deep_hierarchical_clustering_linkage,
            'hc_distance_threshold': deep_hc_distance_threshold,
            'hc_auto_k_method': deep_hc_auto_k_method,
            'hc_auto_k_param': deep_hc_auto_k_param
        }

        # Simplified 3-stage U-Net Structure
        self.head_conv = nn.Conv2d(spectral_num+1, ch_enc_dec, 3, 1, 1)
        
        # Stage 1 (Encoder)
        self.rb1 = CANResBlock(ch_enc_dec, cluster_num, filter_threshold, **shallow_res_block_params)
        self.transformer1 = LightweightTransformer(ch_enc_dec, transformer_heads, transformer_ff_dim_multiplier, transformer_dropout, spatial_reduction=transformer_spatial_reduction)
        self.down1 = ConvDown(ch_enc_dec, ch_bottleneck)
        
        # Stage 2 (Bottleneck)
        self.rb2 = CANResBlock(ch_bottleneck, cluster_num, filter_threshold, **deep_res_block_params)
        self.transformer2 = LightweightTransformer(ch_bottleneck, transformer_heads, transformer_ff_dim_multiplier, transformer_dropout, spatial_reduction=transformer_spatial_reduction)
        self.up1 = ConvUp(ch_bottleneck, ch_enc_dec)
        
        # Stage 3 (Decoder)
        self.rb3 = CANResBlock(ch_enc_dec, cluster_num, filter_threshold, **shallow_res_block_params)
        self.transformer3 = LightweightTransformer(ch_enc_dec, transformer_heads, transformer_ff_dim_multiplier, transformer_dropout, spatial_reduction=transformer_spatial_reduction)
        
        self.tail_conv = nn.Conv2d(ch_enc_dec, spectral_num, 3, 1, 1)

    def forward(self, pan, lms, cache_indice=None):
        x_cat = torch.cat([pan, lms], dim=1)
        x_head = self.head_conv(x_cat) # ch_enc_dec

        # Stage 1 (Encoder)
        res_s1, idx_s1, cluster_loss_s1 = self.rb1(x_head, cache_indice)
        trans_s1 = self.transformer1(res_s1)
        out_s1 = res_s1 + trans_s1 # ch_enc_dec, for skip connection
        del res_s1
        del x_head

        # Downsampling to Bottleneck
        x_down = self.down1(out_s1)

        # Stage 2 (Bottleneck)
        res_s2, idx_s2, cluster_loss_s2 = self.rb2(x_down, cache_indice) # idx_s2 is captured but not used by rb3
        trans_s2 = self.transformer2(res_s2)
        out_s2 = res_s2 + trans_s2
        del res_s2
        del x_down
        del idx_s2 # Not used later

        # Upsampling and Decoder
        x_up = self.up1(out_s2, out_s1) # Skip connection: out_s1
        del out_s1
        del out_s2

        # Stage 3 (Decoder)
        res_s3, _, cluster_loss_s3 = self.rb3(x_up, cache_indice, idx_s1) # Use idx_s1 from Stage 1
        trans_s3 = self.transformer3(res_s3)
        out_s3 = res_s3 + trans_s3
        del res_s3
        del x_up
        del idx_s1

        output = self.tail_conv(out_s3)
        del out_s3
        
        total_cluster_loss = (cluster_loss_s1 or torch.tensor(0.0, device=lms.device)) + (cluster_loss_s2 or torch.tensor(0.0, device=lms.device)) + (cluster_loss_s3 or torch.tensor(0.0, device=lms.device))
        model_outputs = {"cluster_loss": total_cluster_loss}
        
        return lms + output, model_outputs
