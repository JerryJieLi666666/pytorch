import torch

# 1. 定义输入数据和真实标签
# 假设我们有一些线性关系的数据，y = 3 * x + 2
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y_real = torch.tensor([[5.0], [8.0], [11.0], [14.0]])

w = torch.tensor([[1.0]], requires_grad=True)
b = torch.tensor([[1.0]], requires_grad=True)

learning_rate = 0.01

for epoch in range(10000):

    y_pred = x.mm(w) + b

    loss = torch.mean((y_pred - y_real)**2)

    loss.backward()

    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    w.grad.zero_()
    b.grad.zero_()

    if epoch % 100 == 0:
        print(f'Epoch: {epoch}, loss: {loss.item()}')

print(f'Trained weight: {w.item()}')
print(f'Trained bias: {b.item()}')