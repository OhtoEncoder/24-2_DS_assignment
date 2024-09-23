import torch.nn as nn
from torch import Tensor  # 추가된 임포트

class LayerNormalization(nn.Module):
    def __init__(self, d_model: int) -> None:
        super(LayerNormalization, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)  #입력 차원에 맞춘 Layer Normalization 초기화
    
    def forward(self, x: Tensor) -> Tensor:
        return self.layer_norm(x) 