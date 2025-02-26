import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import random


# 1. 自定义数据集：读取 JPG 文件并处理
class MedicalImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, transform=None, low_res_size=32, high_res_size=64):
        self.image_dir = image_dir
        self.image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.jpg')]
        self.transform = transform
        self.low_res_size = low_res_size
        self.high_res_size = high_res_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        high_res_image = Image.open(image_path).convert('L')  # 读取并转换为灰度图

        # 随机裁剪 64x64 高分辨率图像
        crop_x = random.randint(0, high_res_image.size[0] - self.high_res_size)
        crop_y = random.randint(0, high_res_image.size[1] - self.high_res_size)
        high_res_image = high_res_image.crop((crop_x, crop_y, crop_x + self.high_res_size, crop_y + self.high_res_size))

        # 对高分辨率图像进行 bicubic 插值，生成低分辨率图像
        low_res_image = high_res_image.resize((self.low_res_size, self.low_res_size), Image.BICUBIC)

        if self.transform:
            high_res_image = self.transform(high_res_image)
            low_res_image = self.transform(low_res_image)

        return low_res_image, high_res_image


# 2. EDSR 模型
class EDSR(nn.Module):
    def __init__(self, num_channels=1, num_features=32, num_residual_blocks=8, scale_factor=2):
        super(EDSR, self).__init__()
        self.scale_factor = scale_factor
        self.input_conv = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=1)
        self.residual_blocks = nn.Sequential(
            *[self._make_residual_block(num_features) for _ in range(num_residual_blocks)]
        )
        self.upsample = nn.Sequential(
            nn.Conv2d(num_features, num_features * (scale_factor**2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.ReLU(inplace=True)
        )
        self.output_conv = nn.Conv2d(num_features, num_channels, kernel_size=3, padding=1)

    def _make_residual_block(self, num_features):
        return nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),  # Dropout 正则化
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x_in = x
        x = self.input_conv(x)
        residual = x

        x = self.residual_blocks(x)
        x += residual  # 加入残差连接

        x = self.upsample(x)
        x = self.output_conv(x)

        return x


# 3. 训练函数
def train_model(model, train_loader, num_epochs=10, learning_rate=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # L2 正则化


    train_loss = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for low_res_images, high_res_images in train_loader:
            low_res_images, high_res_images = low_res_images.to(device), high_res_images.to(device)

            optimizer.zero_grad()
            outputs = model(low_res_images)  # 使用低分辨率图像生成预测

            loss = criterion(outputs, high_res_images)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_loss.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    print("Training Finished!")

    # 绘制损失曲线
    plt.plot(range(1, num_epochs+1), train_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.show()


# 4. 测试函数
def test_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()
    with torch.no_grad():
        for low_res_images, high_res_images in test_loader:
            low_res_images = low_res_images.to(device)
            high_res_images = high_res_images.to(device)

            outputs = model(low_res_images)

            # 展示结果
            plt.figure(figsize=(10, 5))

            # 展示低分辨率图像
            plt.subplot(1, 3, 1)
            plt.imshow(low_res_images[0].cpu().numpy().squeeze(), cmap='gray')
            plt.title('Low Resolution Image')

            # 展示高分辨率图像
            plt.subplot(1, 3, 2)
            plt.imshow(high_res_images[0].cpu().numpy().squeeze(), cmap='gray')
            plt.title('High Resolution Image')

            # 展示超分辨率重建图像
            plt.subplot(1, 3, 3)
            plt.imshow(outputs[0].cpu().numpy().squeeze(), cmap='gray')
            plt.title('Super-Resolution Image')

            plt.show()


# 5. 数据加载
transform = transforms.Compose([transforms.ToTensor()])

# 更新数据集路径和图像格式
train_dataset = MedicalImageDataset(r'C:\Users\17158\PycharmProjects\FIRE\FIRE\train', transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = MedicalImageDataset(r'C:\Users\17158\PycharmProjects\FIRE\FIRE\test', transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

# 6. 初始化模型并开始训练
model = EDSR(num_channels=1, num_features=32, num_residual_blocks=8, scale_factor=2)
train_model(model, train_loader, num_epochs=10, learning_rate=1e-4)

# 7. 测试模型
test_model(model, test_loader)
