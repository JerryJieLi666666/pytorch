# import torch
# import torch.nn as nn
#
# # 创建一个简单的线性层和激活函数
# layer = nn.Linear(2, 3)  # 输入 2 个特征，输出 3 个特征
# activation = nn.ReLU()   # 使用 ReLU 激活函数
#
# # 输入数据
# x = torch.tensor([[1.0, 2.0]])  # 输入数据 x 的形状是 (1, 2)
#
# # 前向传播
# output = activation(layer(x))  # 先通过线性层，再经过 ReLU 激活函数
# print(output)

import torch
import torch.nn as nn

# # 定义一个两层的简单神经网络
# class SimpleNN(nn.Module):
#     def __init__(self):
#         super(SimpleNN, self).__init__()
#         self.fc1 = nn.Linear(2, 3)  # 第一层：线性层，输入为2个特征，输出为3个特征
#         self.relu = nn.ReLU()       # 激活函数：ReLU
#         self.fc2 = nn.Linear(3, 1)  # 第二层：线性层，输入为3个特征，输出为1个特征
#
#     def forward(self, x):
#         out = self.fc1(x)       # 输入经过第一层
#         out = self.relu(out)    # 经过ReLU激活函数
#         out = self.fc2(out)     # 经过第二层
#         return out  # 返回最终输出
#
# # 创建网络的实例
# model = SimpleNN()
#
# # 准备输入数据
# input_data = torch.tensor([[1.0, 2.0]])
#
# # 执行前向传播，计算输出
# output = model(input_data)
#
# # 打印输出
# print("Output:", output)

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 3)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(3, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)

        return out

model = SimpleNN()

input_data = torch.tensor([[1.0, 2.0]])

output = model(input_data)

print("Output: ", output)
