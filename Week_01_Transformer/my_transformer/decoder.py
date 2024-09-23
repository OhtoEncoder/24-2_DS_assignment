import torch
import torch.nn as nn
from typing import Optional
from .attention import MultiHeadAttention
from .feedforward import FeedForwardLayer, DropoutLayer
from .normalization import LayerNormalization
from .residual import ResidualConnection


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)  #셀프어텐션
        self.enc_dec_attn = MultiHeadAttention(d_model, n_heads)  #인코더-디코더 어텐션
        self.feed_forward = FeedForwardLayer(d_model, n_heads, d_ff, dropout)  #피드포워드
        self.dropout = DropoutLayer(dropout)  #드롭아웃 레이어
        self.norm1 = LayerNormalization(d_model)  #레이어 정규화, 첫 셀프어텐션
        self.norm2 = LayerNormalization(d_model)  #레이어 정규화, 인코더디코더 연결
        self.norm3 = LayerNormalization(d_model)  #레이어 정규화, 피드포워드
        self.residual1= ResidualConnection()  #잔차 연결
        self.residual2= ResidualConnection()  #잔차 연결
        self.residual3= ResidualConnection()  #잔차 연결


    
    def forward(self, 
                x: torch.Tensor, 
                memory: torch.Tensor, 
                src_mask: Optional[torch.Tensor] = None, 
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        #첫 번째 셀프 어텐션
        self_attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.residual1(x, self.dropout(self_attn_output))
        x = self.norm1(x)
        
        #인코더-디코더 어텐션
        enc_dec_attn_output = self.enc_dec_attn(x, memory, memory, src_mask)
        x = self.residual2(x, self.dropout(enc_dec_attn_output))
        x = self.norm2(x)
        
        #피드포워드
        feed_forward_output = self.feed_forward(x)
        x = self.residual3(x, self.dropout(feed_forward_output))
        x = self.norm3(x)
        
        return x