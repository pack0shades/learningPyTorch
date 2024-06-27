import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], device="cuda")
print(my_tensor)
x = torch.eye(5, 5, device=device)
y = torch.rand(5, 5, device=device)
print(f"y = {y}")

# addition
z1 = torch.empty(3, device=device)
z2 = torch.add(x, y)
torch.add(x, y, out=z2)
z2 = x + y
print(f"z2 = {z2} \nand device = {z2.device}")

# subtraction
z2 = x - y
print(f"subtraction = {z2}")

# division
z2 = torch.true_divide(x, y)  # elementwise division

# inplace operations computationally efficient
t = torch.zeros(5, 5, device=device)
t.add_(x)
print(f"t= {t}")
t += x
print(f"t = {t}")

z = x**2
print(z)

# simple comparison

z = x > 0
print(z)
z = x < 0
print(z)

# matrix multiplication
x = torch.rand(3, 4)
print(f"x = {x}")
y = torch.rand(4, 5)
print(f"y = {y}")
z = torch.mm(x, y)  # dim = 3x5
print(f"matrix multiplication = {z}\n\n")
z = x.mm(y)

# matrix exponentiation
x = torch.rand(4, 4)
print(f"x = {x}")
z = x.matrix_power(3)
print(f"cube of a matrix x = {z}\n\n")

# dot product
x = torch.rand(4)
print(f"x = {x}")
y = torch.rand(4)
print(f"y = {y}")
z = torch.dot(x, y)
print(f"dot product = {z}\n\n")

# Batch matrix multiplication
batch = 32
n = 10
m = 20
p = 30

x = torch.rand(batch, n, m)
print(f"x = {x}")
y = torch.rand(batch, m, n)
print(f"y = {y}")
z = torch.bmm(x, y)
print(f"batch matrix mult. = {z}\n\n")
print("##Tensor Indexing\n")
# Tensor indexing

batch_size = 18
features = 25
x = torch.rand(batch_size, features)

print(x[0].shape)  # eq. to (x[0,:].shape)
print("\n\n")

# Tensor reshaping

x = torch.arange(9)
print(f"x = {x}")
x_3x3 = x.view(3, 3)
print(f"x_3x3 = {x_3x3}\n")
x_3x3 = x.reshape(3, 3)  # view works on contiguous memory but efficient computation whereas reshape works everywhere
print(f"x_3x3 = {x_3x3}")
# transpose
y = x_3x3.t()
print(f"transpose = {y}")
# flattening
x = torch.rand(2, 5)
print(f"x = {x}, dim. = {x.shape}")
z = x.view(-1)
print(f"flattened = {z}, dim. = {z.shape}")
x = torch.rand(batch, 2, 5)
print(f"x = {x}, dim. = {x.shape}")
z = x.view(batch, -1)
print(f"flattened = {z}, dim. = {z.shape}")
