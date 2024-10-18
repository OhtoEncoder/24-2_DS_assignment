import torch
import torch.nn as nn

"""
정말 간단하게 논문 구현만 해봅시다
논문에서 제시한대로 구현하면 됩니다!!
데이터셋은 Fashion MNIST를 사용하기 때문에, 이미지의 크기(=in_features)는 28 * 28 = 784입니다

Hint: nn.Sequential을 사용하면 간단하게 구현할 수 있습니다.

"""

class Original_Discriminator(nn.Module):

  def __init__(self, in_features):
    super().__init__()
    self.disc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

  def forward(self, x):
    return self.disc(x)