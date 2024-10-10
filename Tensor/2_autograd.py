import torch

x = torch.tensor([2.0], requires_grad=True)

y = x**2

y.backward()

print(x.grad)

# requires_grad=True：告诉 PyTorch 需要跟踪这个张量的所有操作，以便自动计算梯度。
# y.backward()：自动计算 y 对 x 的导数。
# x.grad：保存 x 的梯度值。