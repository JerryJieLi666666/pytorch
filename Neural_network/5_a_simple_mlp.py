import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义神经网络
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        # 定义层：使用 nn.Linear 定义全连接层
        self.fc1 = nn.Linear(28 * 28, 128)  # 输入是 28x28 像素的图像，输出 128 维
        self.fc2 = nn.Linear(128, 64)  # 隐藏层：从 128 维到 64 维
        self.fc3 = nn.Linear(64, 10)  # 输出层：10 个类别对应 MNIST 中的数字 0-9

    # 实现前向传播
    def forward(self, x):
        # 将输入 x 展平为一维向量
        x = x.view(-1, 28 * 28)  # 输入图像为 (batch_size, 28, 28)，展平成 (batch_size, 28*28)
        x = F.relu(self.fc1(x))  # 第一个线性层加 ReLU 激活
        x = F.relu(self.fc2(x))  # 第二个线性层加 ReLU 激活
        x = self.fc3(x)  # 输出层，不加激活，因为交叉熵损失函数中会隐含 softmax
        return x

# 初始化网络
model = SimpleMLP()
print(model)

# import torch
# import torch.nn as nn
# import torch.functional as F
#
# class SimpleMLP(nn.Module):
#     def __init__(self):
#         super(SimpleMLP, self).__init__()
#
#         self.fc1 = nn.Linear(28 * 28, 128)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, 10)
#
#     def forward(self, x):
#         x = x.view(-1, 28 * 28)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#
#         return x
#
# model = SimpleMLP()
#
# print(model)

