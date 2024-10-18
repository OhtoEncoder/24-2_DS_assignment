import torch
import torch.nn as nn

"""
GAN의 Discriminator를 자유롭게 개선해주세요!!
단순 논문 구현한 Discriminator에 이것 저것을 추가해도 좋고, 변경해도 좋습니다!

Hint:
1. Batch Normalization
2. Dropout
3. Deep Layer
4. etc...

Layer가 깊을수록 성능이 좋아질까요?? 

"""

class Discriminator(nn.Module):

    def __init__(self, in_features):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features, 512),     
            nn.LeakyReLU(0.2),                
            nn.Dropout(0.1),                  
            #층 여러겹 쌓자
            nn.Linear(512, 256),          
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),      
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),             
            nn.Sigmoid()                      
        )

    def forward(self,z):
        #One Line
        return self.disc(z)