import torch
import sys
sys.path.insert(0, '.')
from canconv.layers.canconv import CANConv

print('Testing CANConv forward method...')
try:
    layer = CANConv(in_channels=4, out_channels=4, cluster_num=2)
    x = torch.randn(1, 4, 4, 4)
    result = layer(x)
    print(f'Forward returned {len(result)} values: SUCCESS\!')
    print('CANConv fix verified\!')
except Exception as e:
    print(f'Error: {e}')
