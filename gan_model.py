import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import os

# 创建保存生成图像的文件夹
os.makedirs("gan_images", exist_ok=True)

# 定义生成器网络
class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_shape=(1, 28, 28)):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),  # 输入层到隐藏层
            nn.LeakyReLU(0.2, inplace=True),  # LeakyReLU激活函数
            nn.Linear(128, 256),  # 隐藏层到隐藏层
            nn.BatchNorm1d(256),  # Batch Normalization
            nn.LeakyReLU(0.2, inplace=True),  # LeakyReLU激活函数
            nn.Linear(256, 512),  # 隐藏层到隐藏层
            nn.BatchNorm1d(512),  # Batch Normalization
            nn.LeakyReLU(0.2, inplace=True),  # LeakyReLU激活函数
            nn.Linear(512, img_shape[0] * img_shape[1] * img_shape[2]),  # 隐藏层到输出层
            nn.Tanh()  # Tanh激活函数
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img

# 定义判别器网络
class Discriminator(nn.Module):
    def __init__(self, img_shape=(1, 28, 28)):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(img_shape[0] * img_shape[1] * img_shape[2], 512),  # 输入层到隐藏层
            nn.LeakyReLU(0.2, inplace=True),  # LeakyReLU激活函数
            nn.Linear(512, 256),  # 隐藏层到隐藏层
            nn.LeakyReLU(0.2, inplace=True),  # LeakyReLU激活函数
            nn.Linear(256, 1),  # 隐藏层到输出层
            nn.Sigmoid()  # Sigmoid激活函数
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

# 参数设置
latent_dim = 100
img_shape = (1, 28, 28)
lr = 0.0002
batch_size = 64
epochs = 100

# 准备数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 初始化生成器和判别器
generator = Generator(latent_dim, img_shape)
discriminator = Discriminator(img_shape)

# 定义损失函数和优化器
adversarial_loss = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

# cpu上训练
device = torch.device('cpu')

# 训练
for epoch in range(epochs):
    for i, (imgs, _) in enumerate(train_loader):
        # 训练判别器
        optimizer_D.zero_grad()
        real_imgs = imgs.view(imgs.size(0), -1).to(device)
        real_labels = torch.ones(imgs.size(0), 1).to(device)
        fake_labels = torch.zeros(imgs.size(0), 1).to(device)

        z = torch.randn(imgs.size(0), latent_dim).to(device)
        fake_imgs = generator(z)

        real_loss = adversarial_loss(discriminator(real_imgs), real_labels)
        fake_loss = adversarial_loss(discriminator(fake_imgs.detach()), fake_labels)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        # 训练生成器
        optimizer_G.zero_grad()
        z = torch.randn(imgs.size(0), latent_dim).to(device)
        fake_imgs = generator(z)
        g_loss = adversarial_loss(discriminator(fake_imgs), real_labels)

        g_loss.backward()
        optimizer_G.step()

        # 打印训练信息
        batches_done = epoch * len(train_loader) + i
        if batches_done % 400 == 0:
            print(
                f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(train_loader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]"
            )

    # 保存生成的图片
    if epoch % 10 == 0:
        save_image(fake_imgs.data[:25], f"gan_images/{epoch}.png", nrow=5, normalize=True)