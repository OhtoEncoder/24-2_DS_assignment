import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from typing import Optional, Tuple

class QueryLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super(QueryLayer, self).__init__()
        self.linear = nn.Linear(d_model, d_model * n_heads)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class KeyLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super(KeyLayer, self).__init__()
        self.linear = nn.Linear(d_model, d_model * n_heads)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class ValueLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super(ValueLayer, self).__init__()
        self.linear = nn.Linear(d_model, d_model * n_heads)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
        
class ScaledDotProductAttention(nn.Module): #어텐션스코어 계산!
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        d_k = k.size(-1)  # Key의 마지막 차원 크기
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  #Q와 K의 내적, 스케일링
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))  #마스크
        attention = F.softmax(scores, dim=-1)  #소프트맥스
        output = torch.matmul(attention, v)  #V에 웨이트 곱하기
        return output, attention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        
        self.query_layers = QueryLayer(d_model, n_heads)
        self.key_layers = KeyLayer(d_model, n_heads)
        self.value_layers = ValueLayer(d_model, n_heads)
        self.attention = ScaledDotProductAttention()
        self.fc = nn.Linear(n_heads * d_model, d_model)
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = Q.size(0)
        
        # Q, K, V를 각각 여러 헤드로 변환
        q = self.query_layers(Q).view(batch_size, -1, self.n_heads, self.d_model).transpose(1, 2)
        k = self.key_layers(K).view(batch_size, -1, self.n_heads, self.d_model).transpose(1, 2)
        v = self.value_layers(V).view(batch_size, -1, self.n_heads, self.d_model).transpose(1, 2)
        
        # Scaled Dot Product Attention 적용
        output, attention = self.attention(q, k, v, mask)
        
        # 헤드들을 합치고, 최종 선형 변환
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_model)
        output = self.fc(output)
        
        return output