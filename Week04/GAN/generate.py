import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from original_generator import Original_Generator
from improved_generator import Generator 
import os
import argparse

"""
훈련한 모델을 불러와서 이미지를 생성하는 코드입니다!
명령줄 인자를 사용하여 simple GAN과 improved GAN을 선택할 수 있습니다.

단순 구현한 GAN 모델 불러올 시 (터미널에 입력)
python generate.py --model_type original 

개선한 GAN 모델 불러올 시 (터미널에 입력)
python generate.py --model_type improved

나머지는 default 값으로 설정되어 있습니다.

checkpoint_file은 커맨드로 입력하면 너무 길어지기 때문에 직접 코드에서 수정해주세요!

여기서 생성한 이미지의 디렉토리를 IS Score를 계산하는 코드에 입력해주세요!
"""

def show_generated_image(generated_image, save_path=None):
    generated_image = (generated_image.squeeze(0) + 1) / 2
    generated_image = generated_image.detach().cpu().numpy()

    plt.imshow(generated_image, cmap="gray")
    plt.axis("off")  

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  
        plt.imsave(save_path, generated_image, cmap="gray") 
        print(f"Image saved to {save_path}")

    plt.close()

def generate_random_image(model_type, checkpoint_file, z_dim, index):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    image_dim = 784  # 28 * 28 = 784 (Fashion MNIST 이미지의 크기)

    # 모델의 종류에 따라 Generator 선택
    if model_type == "original":
        gen = Original_Generator(z_dim, image_dim).to(device)  # Original_Generator
        save_dir = "original_generated_image"
    elif model_type == "improved":
        gen = Generator(z_dim, image_dim).to(device)  # Improved_Generator
        save_dir = "improved_generated_image"
    else:
        raise ValueError("model_type should be either 'original' or 'improved'")

    checkpoint = torch.load(checkpoint_file)
    gen.load_state_dict(checkpoint['state_dict'])
    gen.eval() 

    noise = torch.randn(index, z_dim).to(device) 

    # 이미지 생성
    for idx, n in enumerate(noise):
        n = n.view(1, z_dim)
        with torch.no_grad():
            generated_image = gen(n)
            generated_image = generated_image.view(1, 28, 28)  # (batch_size, height, width)

        # 저장 경로에서 모델 타입에 맞는 디렉토리로 저장
        show_generated_image(generated_image, save_path=f"{save_dir}/generated_image_{idx}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True, help="Type of model: 'original' or 'improved'")
    parser.add_argument("--z_dim", type=int, default=64, help="Dimensionality of latent vector z")
    parser.add_argument("--num_images", type=int, default=1000, help="Number of images to generate")
    args = parser.parse_args()

    generator_checkpoint = "original_model/generator_best.pth.tar"

    generate_random_image(args.model_type, generator_checkpoint, args.z_dim, args.num_images)
