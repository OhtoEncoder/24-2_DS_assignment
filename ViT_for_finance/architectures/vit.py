import torch
from torch import nn
from torch.nn import functional as F
from timm.models.layers import trunc_normal_

class EmbeddingLayer(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super(EmbeddingLayer, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = patch_size * patch_size * in_chans  # 패치 차원
        self.projection = nn.Linear(self.patch_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 입력 이미지 크기가 패치 크기의 배수인지 확인
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(f"Input image size ({H}x{W}) must be divisible by the patch size ({self.patch_size}).")
        
        # unfold 연산을 통해 패치를 추출합니다
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        _, _, num_patches_h, num_patches_w, patch_h, patch_w = x.shape
        
        # 패치 확인 및 뷰 설정
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()  # (B, num_patches_h, num_patches_w, C, patch_h, patch_w)
        x = x.view(B, num_patches_h * num_patches_w, -1)  # (B, num_patches, C * patch_h * patch_w)
        
        # 프로젝션 레이어를 통해 임베딩 차원으로 변환
        x = self.projection(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        return x
        
# Multi-Head Self Attention Layer: 입력 텐서에 대해 다중 헤드 셀프 어텐션을 수행
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# MLP (Multi-Layer Perceptron) Layer: Transformer 블록 내에 사용되는 다층 퍼셉트론
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, drop=0.):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# Block: Transformer Encoder 블록 하나로, 어텐션과 MLP가 결합된 구조
class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., drop_path=0.):
        super(Block, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, qkv_bias, attn_drop, drop)
        self.drop_path = nn.Identity() if drop_path == 0. else nn.Dropout(drop_path)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(embed_dim, mlp_hidden_dim, embed_dim, drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# Vision Transformer: 전체 ViT 모델
class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=1, in_chans=65, num_classes=1000, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = EmbeddingLayer(img_size, patch_size, in_chans, embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls_token = x[:, 0]
        x = self.head(cls_token)
        return x