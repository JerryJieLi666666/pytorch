import torch

# 创建一个张量 w，初始值为 2.0，要求跟踪梯度
w = torch.tensor([2.0], requires_grad=True)

# 打印张量 w
print("张量 w：", w)

# 打印 w 的数据部分
print("w 的数据部分：", w.data)

# 打印 w 的梯度（此时还没有计算梯度）
print("w 的梯度：", w.grad)

# 进行一次计算：假设 y = w * 3
y = w * 3

# 计算 y 的梯度
y.backward()

# 打印此时 w 的梯度
print("w 的梯度（计算后）：", w.grad)
