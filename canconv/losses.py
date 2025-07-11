import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralAngleMapper(nn.Module):
    def __init__(self, epsilon: float = 1e-8):
        """
        光谱角损失函数 (Spectral Angle Mapper, SAM)

        Args:
            epsilon (float): 一个小的常数，用于防止除以零和acos的数值不稳定性。
        """
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        计算SAM损失。

        Args:
            y_pred (torch.Tensor): 预测的光谱图像，形状为 (B, C, H, W)
                                   B: 批次大小, C: 光谱通道数, H: 高度, W: 宽度
            y_true (torch.Tensor): 真实的光谱图像，形状为 (B, C, H, W)

        Returns:
            torch.Tensor: SAM损失值 (一个标量)
        """
        if not (y_pred.ndim == 4 and y_true.ndim == 4):
            raise ValueError(
                f"Inputs must be 4D tensors. Got y_pred: {y_pred.ndim}D, y_true: {y_true.ndim}D"
            )
        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"Input shapes must match. Got y_pred: {y_pred.shape}, y_true: {y_true.shape}"
            )

        # 沿着光谱通道维度 (dim=1) 计算点积
        # (B, C, H, W) * (B, C, H, W) -> (B, C, H, W) --sum(dim=1)--> (B, H, W)
        dot_product = (y_pred * y_true).sum(dim=1)

        # 计算每个光谱向量的L2范数 (模)
        # norm(dim=1) -> (B, H, W)
        norm_pred = torch.linalg.norm(y_pred, ord=2, dim=1)
        norm_true = torch.linalg.norm(y_true, ord=2, dim=1)

        # 计算余弦角
        # (B, H, W) / ((B, H, W) * (B, H, W) + eps)
        cos_angle = dot_product / (norm_pred * norm_true + self.epsilon)

        # 防止 acos 的输入超出 [-1, 1] 范围
        cos_angle_clamped = torch.clamp(cos_angle, -1.0 + self.epsilon, 1.0 - self.epsilon)

        # 计算角度 (弧度)
        # acos(...) -> (B, H, W)
        angle = torch.acos(cos_angle_clamped)

        # 返回所有像素角度的平均值
        return angle.mean()

if __name__ == '__main__':
    # 示例用法
    B, C, H, W = 2, 8, 16, 16  # 批次大小, 通道数, 高度, 宽度
    pred = torch.rand(B, C, H, W, dtype=torch.float32)
    true = torch.rand(B, C, H, W, dtype=torch.float32)

    sam_loss_fn = SpectralAngleMapper()
    loss_value = sam_loss_fn(pred, true)
    print(f"SAM Loss: {loss_value.item()}")

    # 测试一些特殊情况
    # 1. 完美匹配 (理论上loss应为0)
    perfect_match_pred = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.float32) # (1,1,2,2) C=1
    perfect_match_true = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.float32)
    
    # 为了让SAM有意义，C应该大于1，这里我们模拟多通道
    pm_pred_c3 = torch.cat([perfect_match_pred * 0.5, perfect_match_pred, perfect_match_pred * 1.5], dim=1) # (1,3,2,2)
    pm_true_c3 = torch.cat([perfect_match_pred * 0.5, perfect_match_pred, perfect_match_pred * 1.5], dim=1)
    
    loss_perfect = sam_loss_fn(pm_pred_c3, pm_true_c3)
    print(f"SAM Loss (Perfect Match): {loss_perfect.item()}") # 应该接近0

    # 2. 正交向量 (理论上loss应为 pi/2)
    # v1 = [1,0,0], v2 = [0,1,0]
    ortho_pred = torch.zeros(1, 3, 1, 1, dtype=torch.float32)
    ortho_pred[0, 0, 0, 0] = 1.0
    ortho_true = torch.zeros(1, 3, 1, 1, dtype=torch.float32)
    ortho_true[0, 1, 0, 0] = 1.0
    loss_orthogonal = sam_loss_fn(ortho_pred, ortho_true)
    print(f"SAM Loss (Orthogonal): {loss_orthogonal.item()} (pi/2 is approx {torch.pi/2})")

    # 3. 相反向量 (理论上loss应为 pi)
    # v1 = [1,0,0], v2 = [-1,0,0]
    opposite_pred = torch.zeros(1, 3, 1, 1, dtype=torch.float32)
    opposite_pred[0, 0, 0, 0] = 1.0
    opposite_true = torch.zeros(1, 3, 1, 1, dtype=torch.float32)
    opposite_true[0, 0, 0, 0] = -1.0
    loss_opposite = sam_loss_fn(opposite_pred, opposite_true)
    print(f"SAM Loss (Opposite): {loss_opposite.item()} (pi is approx {torch.pi})")

    print("Testing with CANNet output shapes:")
    spectral_num = 8 # from CANNet
    mock_output = torch.rand(B, spectral_num, H, W)
    mock_target = torch.rand(B, spectral_num, H, W)
    loss_cannet_shape = sam_loss_fn(mock_output, mock_target)
    print(f"SAM Loss (CANNet-like shapes): {loss_cannet_shape.item()}") 