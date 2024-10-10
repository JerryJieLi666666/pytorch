import torch


w = torch.tensor([2.0], requires_grad=True)

x = torch.tensor([3.0])
y_true = torch.tensor([6.0])

y_pred = w * x

loss = (y_pred - y_true)**2

loss.backward()

print("w_grad: ", w.grad)

learning_rate = 0.01
w.data = w.data - w.grad * learning_rate

print("Updated w: ", w)

