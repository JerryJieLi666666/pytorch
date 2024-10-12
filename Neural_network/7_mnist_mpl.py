import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# 1. 加载 MNIST 数据集
# 定义图像的转换：将图像转换为张量并进行归一化
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Normalize((0.5,), (0.5,))  # 将像素值归一化到 [-1, 1] 范围
])

# 加载训练集
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 加载测试集
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 2. 定义简单的多层感知器（MLP）
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        # 输入层：784 -> 隐藏层 1：128
        self.fc1 = nn.Linear(28 * 28, 128)
        # 隐藏层 1：128 -> 隐藏层 2：64
        self.fc2 = nn.Linear(128, 64)
        # 隐藏层 2：64 -> 输出层：10
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # 展平输入图像（28x28 -> 784）
        x = x.view(-1, 28 * 28)
        # 第一个全连接层 + ReLU
        x = torch.relu(self.fc1(x))
        # 第二个全连接层 + ReLU
        x = torch.relu(self.fc2(x))
        # 输出层（不需要激活函数）
        x = self.fc3(x)
        return x

# 初始化网络
model = SimpleMLP()

# 3. 定义损失函数和优化器
# 使用交叉熵损失函数
criterion = nn.CrossEntropyLoss()

# 使用随机梯度下降（SGD）作为优化器，学习率为 0.01
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 4. 实现训练过程（前向传播和反向传播）
num_epochs = 10  # 训练 5 个周期

for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        # 清除之前的梯度
        optimizer.zero_grad()

        # 前向传播：将输入传入模型
        outputs = model(inputs)

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播：计算梯度
        loss.backward()

        # 更新参数
        optimizer.step()

        # 记录损失
        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# 5. 在测试集上评估模型
correct = 0
total = 0

# 禁用梯度计算，因为在评估时不需要计算梯度
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'测试集上的准确率: {accuracy:.2f}%')


# 6. 可视化部分MNIST数据集

# 从 DataLoader 中取一个 batch 的数据
data_iter = iter(train_loader)
images, labels = next(data_iter)

# 由于图像被归一化到了 [-1, 1]，需要反归一化到 [0, 1] 以便显示
images = images / 2 + 0.5  # 反归一化

# 将 8 张图片可视化
def imshow(img):
    img = img.numpy()  # 将图像从张量转换为 NumPy 格式
    plt.imshow(img, cmap='gray')  # 使用灰度颜色映射显示图像
    plt.show()

# 展示图片和对应的标签
plt.figure(figsize=(10, 2))  # 设置图像大小
for i in range(8):
    plt.subplot(1, 8, i+1)  # 1 行 8 列
    plt.axis('off')  # 隐藏坐标轴
    imshow(images[i][0])  # 将第 i 个图片展示出来（MNIST 图像是单通道）
    plt.title(f'{labels[i].item()}')  # 显示标签