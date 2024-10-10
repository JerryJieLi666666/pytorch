import torch

# 创建一个简单的线性模型
w = torch.tensor([2.0], requires_grad=True)
b = torch.tensor([1.0], requires_grad=True)
x = torch.tensor([3.0])
y_true = torch.tensor([7.0])

# 前向传播：计算预测值
y_pred = w * x + b

# 计算损失
loss = (y_pred - y_true) ** 2

# 反向传播：计算梯度
loss.backward()

# 打印梯度
print(f'w 的梯度: {w.grad.item()}')  # 输出 w 的梯度
print(f'b 的梯度: {b.grad.item()}')  # 输出 b 的梯度

# 使用梯度下降更新 w 和 b
learning_rate = 0.01
with torch.no_grad():  # 禁用梯度追踪
    w -= learning_rate * w.grad
    b -= learning_rate * b.grad

# 清除梯度
w.grad.zero_()
b.grad.zero_()