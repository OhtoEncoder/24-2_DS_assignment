import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from model.VAE import VAE
from model.my_VAE import my_VAE
import os

# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 중인 디바이스: {device}")

# 데이터셋 준비 (Fashion MNIST)
transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

# 모델 불러오기 및 GPU로 이동
origin_dir = "original_images"
os.makedirs(origin_dir, exist_ok=True)
model = VAE().to(device)
model.load_state_dict(torch.load('vae_fashion_mnist.pth'))
model.eval()
print("모델 불러오기 완료!")
save_dir = "original_generated_images"
os.makedirs(save_dir, exist_ok=True)

model_improved = my_VAE().to(device)
model_improved.load_state_dict(torch.load('vae.pth'))
model_improved.eval()
print("개선 모델 불러오기 완료!")
improved_save_dir = "improved_generated_images"
os.makedirs(improved_save_dir, exist_ok=True)

# 테스트 (원본 이미지와 재구성 이미지 비교)
def save_images(original, reconstructed, improved_reconstructed, num_images=1000): #처음부터 1000말고 5로 체크
    original = original[:num_images]  # 원본 이미지들
    reconstructed = reconstructed[:num_images]  # 재구성된 이미지들
    
    # 원본 이미지 저장
    for i in range(num_images):
        # 원본 이미지 저장
        plt.imsave(os.path.join(origin_dir, f"original_image_{i+1}.png"), original[i].squeeze().cpu().numpy(), cmap='gray')
        # 재구성된 이미지 저장
        plt.imsave(os.path.join(save_dir, f"reconstructed_image_wK_{i+1}.png"), reconstructed[i].squeeze().cpu().numpy(), cmap='gray')
        plt.imsave(os.path.join(improved_save_dir, f"reconstructed_image_wK_{i+1}.png"), improved_reconstructed[i].squeeze().cpu().numpy(), cmap='gray')

# 테스트 데이터에서 재구성 이미지 확인
with torch.no_grad():
    for batch_idx, (data, _) in enumerate(test_loader):
        data = data.to(device)  # 데이터를 GPU로 이동
        recon_batch, _, _ = model(data)
        improved_recon_batch, _, _ = model_improved(data)
        save_images(data, recon_batch, improved_recon_batch, num_images=1000)
        break  # 첫 배치만 사용하여 원본과 재구성 이미지 저장