import torch
import torch.nn as nn

#######################################################
# 定义损失函数
mse_loss = nn.MSELoss()

# 假设真实房价 y_true 和预测房价 y_pred
y_true = torch.tensor([[520000.0]])  # 实际房价
y_pred = torch.tensor([[500000.0]])  # 预测房价

# 计算损失
loss = mse_loss(y_pred, y_true)
print(loss.item())  # 输出 MSE 损失


########################################################
# 定义交叉熵损失函数
cross_entropy_loss = nn.CrossEntropyLoss()

# 假设真实标签 y_true 和预测概率 y_pred
# 对于多分类问题，真实标签是类别索引，预测值是每个类别的概率分布
y_true = torch.tensor([0])  # 真实类别为第 0 类
y_pred = torch.tensor([[0.7, 0.2, 0.1]])  # 预测为第 0 类的概率是 0.7

# 计算损失
loss = cross_entropy_loss(y_pred, y_true)
print(loss.item())  # 输出交叉熵损失


########################################################
# 定义二元交叉熵损失函数
bce_loss = nn.BCELoss()

# 假设真实标签 y_true 和预测概率 y_pred
y_true = torch.tensor([[1.0]])  # 真实值为 1，即垃圾邮件
y_pred = torch.tensor([[0.9]])  # 模型预测为 0.9

# 计算损失
loss = bce_loss(y_pred, y_true)
print(loss.item())  # 输出二元交叉熵损失


######################################################
# 定义负对数似然损失函数
nll_loss = nn.NLLLoss()

# 假设预测值 y_pred 是对数形式
y_pred = torch.tensor([[-1.2, -0.8, -2.0]])  # log_softmax 的输出
y_true = torch.tensor([1])  # 真实标签为类别 1

# 计算损失
loss = nll_loss(y_pred, y_true)
print(loss.item())  # 输出 NLL 损失