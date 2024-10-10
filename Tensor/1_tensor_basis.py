import torch

# Create tensor
tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(tensor)

# tensor addition
tensor1 = torch.tensor([1, 2, 3])
tensor2 = torch.tensor([4, 5, 6])

result = tensor1 + tensor2
print(result)

# Random tensor
random_tensor = torch.rand(3, 3)
print(random_tensor)


# Tensor shape
print(tensor.shape)
# Tensor size
print(tensor.size())

# Move tensor to gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_on_gpu = torch.tensor([1, 2, 3], device=device)
print(torch_on_gpu)
