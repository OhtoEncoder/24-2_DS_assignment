import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from improved_discriminator import Discriminator as Improved_Discriminator
from original_discriminator import Original_Discriminator
from original_generator import Original_Generator
from improved_generator import Generator as Improved_Generator
from dataset import get_fashion_mnist_loader
import os
import argparse


"""
코드 실행 방법:

단순 구현한 GAN 훈련 시 (터미널에 입력)
python train.py --model_type original

개선한 GAN 훈련 시 (터미널에 입력)
python train.py --model_type improved

"""

def save_checkpoint(model, optimizer, directory_name, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    filename = os.path.join(directory_name, filename)
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def main(model_type):
    device = "mps:0" if torch.backends.mps.is_available() else "cpu"
    """
    Hyperparameters:
    lr: learning rate
    z_dim: latent vector z의 차원
    image_dim: 이미지의 차원
    batch_size: 배치 크기
    num_epochs: epoch 횟수
    """
    
    lr = 3e-4
    z_dim = 64  # 128, 256 (보통 이 중에서 선택)
    image_dim = 784  #  28 * 28 = 784 (Fashion MNIST 이미지의 크기)
    batch_size = 32
    num_epochs = 50
    save_model = True  

    # 인자에 따라 모델을 변경
    if model_type == "original":
        disc = Original_Discriminator(image_dim).to(device)
        gen = Original_Generator(z_dim, image_dim).to(device)
        directory_name = "original_model"
    elif model_type == "improved":
        disc = Improved_Discriminator(image_dim).to(device)
        gen = Improved_Generator(z_dim, image_dim).to(device)
        directory_name = "improved_model"
    else:
        raise ValueError("model_type should be either 'original' or 'improved'")

    fixed_noise = torch.randn((batch_size, z_dim)).to(device)

    # Dataset 불러오기
    loader = get_fashion_mnist_loader(batch_size=batch_size)

    #이 부분도 수정하셔도 됩니다!!
    opt_disc = optim.Adam(disc.parameters(), lr=lr)
    opt_gen = optim.Adam(gen.parameters(), lr=lr)
    criterion = nn.BCELoss()

    writer_fake = SummaryWriter(f"runs/GAN_FashionMNIST/fake")
    writer_real = SummaryWriter(f"runs/GAN_FashionMNIST/real")

    step = 0
    best_lossG = 9999

    for epoch in range(num_epochs):
        loop = tqdm(loader, leave=True)
        for batch_idx, (real, _) in enumerate(loop):
            real = real.view(-1, 784).to(device)
            batch_size = real.shape[0]

            ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
            noise = torch.randn(batch_size, z_dim).to(device)
            fake = gen(noise)
            disc_real = disc(real).view(-1)
            lossD_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake).view(-1)
            lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            lossD = (lossD_real + lossD_fake) / 2
            disc.zero_grad()
            lossD.backward(retain_graph=True)
            opt_disc.step()

            ### Train Generator: max log(D(G(z)))
            output = disc(fake).view(-1)
            lossG = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            lossG.backward()
            opt_gen.step()

            # tqdm 진행 바 업데이트
            loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
            loop.set_postfix(lossD=lossD.item(), lossG=lossG.item())

            if batch_idx == 0:
                print(
                    f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                        Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
                )

                with torch.no_grad():
                    fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                    data = real.reshape(-1, 1, 28, 28)
                    img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                    img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                    writer_fake.add_image(
                        "Mnist Fake Images", img_grid_fake, global_step=step
                    )
                    writer_real.add_image(
                        "Mnist Real Images", img_grid_real, global_step=step
                    )
                    step += 1

        # Generator의 loss가 가장 적은 epoch를 checkpoint로 저장
        if save_model and lossG < best_lossG:
            best_lossG = lossG  # 최저 lossG 값 갱신
            save_checkpoint(gen, opt_gen, directory_name=directory_name, filename=f"generator_best_v2.pth.tar")
            print(f"New best model saved at epoch {epoch} with lossG: {lossG}")
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True, help="Type of model: 'original' or 'improved'")
    args = parser.parse_args()
    main(args.model_type)

