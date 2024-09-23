import torch.nn as nn
from torch import Tensor
from typing import Optional
from .attention import MultiHeadAttention
from .feedforward import FeedForwardLayer, DropoutLayer
from .normalization import LayerNormalization
from .residual import ResidualConnection

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForwardLayer(d_model, d_ff)
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)
        self.dropout1 = DropoutLayer(dropout)
        self.dropout2 = DropoutLayer(dropout)
        self.residual1 = ResidualConnection()
        self.residual2 = ResidualConnection()
    
    def forward(self, x: Tensor) -> Tensor:
        mask = None  #인코더에서는 마스크가 필요하지 않으니까~
        
        #셀프어텐션
        self_attn_output = self.self_attn(x, x, x, mask)  #Q, K, V 모두 x로 동일하게 입력
        x = self.residual1(x, self.dropout1(self_attn_output))  #드롭아웃과 잔차 연결
        x = self.norm1(x)  #정규화

        #피드포워드
        ff_output = self.ff(x) 
        x = self.residual2(x, self.dropout2(ff_output))  #드롭아웃과 잔차 연결
        x = self.norm2(x)  #정규화

        return x