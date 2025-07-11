import torch
import torch.nn as nn
import torch.nn.functional as F

class SAMLoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(SAMLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        """
        Compute Spectral Angle Mapper Loss.
        Args:
            y_pred: Predicted tensor, shape (B, C, H, W) or (B, N, C)
            y_true: Ground truth tensor, shape (B, C, H, W) or (B, N, C)
        Returns:
            SAM loss (scalar tensor)
        """
        # Normalize to prevent numerical instability
        # y_pred_norm = F.normalize(y_pred, p=2, dim=1, eps=self.epsilon)
        # y_true_norm = F.normalize(y_true, p=2, dim=1, eps=self.epsilon)
        
        # # Dot product
        # dot_product = (y_pred_norm * y_true_norm).sum(dim=1)

        # # Ensure dot_product is within [-1, 1] to avoid acos(nan)
        # dot_product = torch.clamp(dot_product, -1.0 + self.epsilon, 1.0 - self.epsilon)
        
        # angle = torch.acos(dot_product) # Output in radians
        # sam = torch.mean(angle) # Mean over all pixels/vectors and batch

        # Alternative calculation that might be more stable for backprop
        # Or handle shapes like (B, N, C) where C is the spectral dimension

        original_shape_len = len(y_pred.shape)
        if original_shape_len == 4: # B, C, H, W
            b, c, h, w = y_pred.shape
            y_pred = y_pred.reshape(b, c, h * w).permute(0, 2, 1) # -> (B, H*W, C)
            y_true = y_true.reshape(b, c, h * w).permute(0, 2, 1) # -> (B, H*W, C)
        elif original_shape_len == 3: # B, N, C
            pass # Already in correct shape
        else:
            raise ValueError(f"Unsupported input shape: {y_pred.shape}. Expected (B, C, H, W) or (B, N, C)")

        # cos_angle = F.cosine_similarity(y_pred, y_true, dim=2, eps=self.epsilon)
        
        # Numerical stability for acos:
        # Ensure vectors are non-zero, otherwise SAM is undefined or zero.
        # Let's compute dot product and norms manually for better control.
        
        dot_product = (y_pred * y_true).sum(dim=2) # (B, N)
        
        norm_pred = torch.linalg.norm(y_pred, dim=2, ord=2) # (B, N)
        norm_true = torch.linalg.norm(y_true, dim=2, ord=2) # (B, N)
        
        # Product of norms. Add epsilon to prevent division by zero if a vector is zero.
        norm_product = norm_pred * norm_true + self.epsilon
        
        cos_angle = dot_product / norm_product
        
        # Clamp to avoid acos(out_of_bounds) due to floating point inaccuracies
        cos_angle_clamped = torch.clamp(cos_angle, -1.0 + self.epsilon, 1.0 - self.epsilon)
        
        angle = torch.acos(cos_angle_clamped) # Radians, shape (B, N)
        
        # Remove NaNs that might occur if a true vector was zero (norm_true = 0)
        # and y_pred was also zero at that position.
        # In such cases, dot_product is 0, norm_product is epsilon. cos_angle approx 0. angle approx pi/2.
        # If a ground truth vector is zero, its angle with any prediction is ill-defined or could be considered 0.
        # Let's consider pixels/vectors where norm_true is zero to have zero loss contribution for SAM.
        # This means we only penalize angles where there's actual spectral information in the ground truth.
        
        # Create a mask for valid (non-zero norm) true vectors
        valid_mask = norm_true > self.epsilon
        
        # Apply mask to angles. If not valid, angle contributes 0 to mean.
        # Summing masked angles and dividing by count of valid elements.
        if torch.any(valid_mask):
            sam = angle[valid_mask].mean()
        else: # Should not happen with real data, but as a safeguard
            sam = torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype)
            
        # If all true vectors are zero, sam will be nan without the check above.
        # If sam is nan due to all true vectors being zero, return 0.
        if torch.isnan(sam):
             sam = torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype)

        return sam

# Example usage (for testing)
if __name__ == '__main__':
    loss_fn = SAMLoss()
    # Test with B, C, H, W
    y_p_4d = torch.rand(2, 8, 16, 16) * 2 - 1 # Random values between -1 and 1
    y_t_4d = torch.rand(2, 8, 16, 16) * 2 - 1
    y_t_4d[0, :, 0, 0] = 0 # Test a zero vector in true
    
    sam_val_4d = loss_fn(y_p_4d, y_t_4d)
    print(f"SAM Loss (4D input): {sam_val_4d.item()}")

    # Test with B, N, C
    y_p_3d = torch.rand(2, 256, 8) * 2 -1
    y_t_3d = torch.rand(2, 256, 8) * 2 -1
    y_t_3d[0, 0, :] = 0 # Test a zero vector in true

    sam_val_3d = loss_fn(y_p_3d, y_t_3d)
    print(f"SAM Loss (3D input): {sam_val_3d.item()}")

    # Test identical vectors (should be 0 loss)
    identical_pred = torch.tensor([[[0.1, 0.2, 0.7]]], dtype=torch.float32)
    identical_true = torch.tensor([[[0.1, 0.2, 0.7]]], dtype=torch.float32)
    sam_identical = loss_fn(identical_pred, identical_true)
    print(f"SAM Loss (identical): {sam_identical.item()}") # Expect ~0

    # Test orthogonal vectors (should be pi/2 ~ 1.57 loss)
    orthogonal_pred = torch.tensor([[[1.0, 0.0]]], dtype=torch.float32)
    orthogonal_true = torch.tensor([[[0.0, 1.0]]], dtype=torch.float32)
    sam_orthogonal = loss_fn(orthogonal_pred, orthogonal_true)
    print(f"SAM Loss (orthogonal): {sam_orthogonal.item()}") # Expect ~pi/2

    # Test opposite vectors (should be pi ~ 3.14 loss)
    opposite_pred = torch.tensor([[[1.0, 0.0]]], dtype=torch.float32)
    opposite_true = torch.tensor([[[-1.0, 0.0]]], dtype=torch.float32)
    sam_opposite = loss_fn(opposite_pred, opposite_true)
    print(f"SAM Loss (opposite): {sam_opposite.item()}") # Expect ~pi
    
    # Test with one true vector being zero
    pred_zero_true = torch.tensor([[[1.0, 0.5], [0.2, 0.8]]], dtype=torch.float32) # B=1, N=2, C=2
    true_zero_true = torch.tensor([[[0.0, 0.0], [0.3, 0.7]]], dtype=torch.float32)
    sam_zero_true = loss_fn(pred_zero_true, true_zero_true)
    print(f"SAM Loss (one true zero): {sam_zero_true.item()}")

    # Test with all true vectors being zero
    pred_all_zero_true = torch.tensor([[[1.0, 0.5], [0.2, 0.8]]], dtype=torch.float32) # B=1, N=2, C=2
    true_all_zero_true = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]], dtype=torch.float32)
    sam_all_zero_true = loss_fn(pred_all_zero_true, true_all_zero_true)
    print(f"SAM Loss (all true zero): {sam_all_zero_true.item()}") # Expect 0

    # Test with some pred vectors being zero
    pred_some_zero_pred = torch.tensor([[[0.0, 0.0], [0.2, 0.8]]], dtype=torch.float32) # B=1, N=2, C=2
    true_some_zero_pred = torch.tensor([[[0.1, 0.9], [0.3, 0.7]]], dtype=torch.float32)
    sam_some_zero_pred = loss_fn(pred_some_zero_pred, true_some_zero_pred)
    print(f"SAM Loss (some pred zero): {sam_some_zero_pred.item()}") 