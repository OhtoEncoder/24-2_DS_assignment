import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class FeedForwardLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout:any) -> None:
        super(FeedForwardLayer, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)  
        self.linear2 = nn.Linear(d_ff, d_model)  
    
    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(F.relu(self.linear1(x)))  # 첫 번째 변환 후 ReLU 활성화, 두 번째 변환

class DropoutLayer(nn.Module):
    def __init__(self, p: float) -> None:
        super(DropoutLayer, self).__init__()
        self.dropout = nn.Dropout(p)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(x)

class ActivationLayer(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.relu(x)  