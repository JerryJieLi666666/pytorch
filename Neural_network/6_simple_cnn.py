import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 定义第一层卷积层：输入通道 1（灰度图像），输出通道 16，卷积核大小为 5x5
        self.conv1 = nn.Conv2d(1, 16, 5)
        # 定义第二层卷积层：输出通道 32，卷积核大小为 5x5
        self.conv2 = nn.Conv2d(16, 32, 5)
        # 定义全连接层：输入 32*4*4，输出 64
        self.fc1 = nn.Linear(32 * 4 * 4, 64)
        # 定义输出层：输出 10 个类别
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        # 卷积层 + ReLU 激活 + 最大池化
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        # 展平张量
        x = x.view(-1, 32 * 4 * 4)
        # 全连接层 + ReLU 激活
        x = F.relu(self.fc1(x))
        # 输出层（不加激活）
        x = self.fc2(x)
        return x

# 初始化模型
model = SimpleCNN()
print(model)


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super(SimpleCNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 16, 5)
#         self.conv2 = nn.Conv2d(16, 32, 5)
#         self.fc1 = nn.Linear(32 * 4 * 4, 64)
#         self.fc2 = nn.Linear(64, 10)
#
#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2(x), 2))
#         x = x.view(-1, 32 * 4 * 4)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
#
# model = SimpleCNN()
# print((model))
