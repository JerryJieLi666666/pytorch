import torch
import torch.nn as nn

# Create ReLU activation function
relu = nn.ReLU()
x = torch.tensor([[-1.0, 0.5], [2.0, -3]])

output = relu(x)
print(output)


# Leaky ReLU
leaky_relu = nn.LeakyReLU()

output = leaky_relu(x)
print(output)


# Sigmoid
sigmoid = nn.Sigmoid()

output = sigmoid(x)
print(output)


# Tanh
tanh = nn.Tanh()

print(tanh(x))


# Softmax
softmax = nn.Softmax()

print(softmax(x))
