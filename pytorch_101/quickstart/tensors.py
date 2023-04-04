"""
Simply tensors 101
"""
import numpy as np
import torch

# From list
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)


# From Numpy
np_array = np.array(data)
x_np = torch.as_tensor(np_array)

# Ones Like - keeps the data type of x_data
x_ones = torch.ones_like(x_data)
print(f"Ones Tensor: \n {x_ones} \n")

# Randoms - overrided the data type of x_data
x_rand = torch.rand_like(x_data, dtype=torch.float)  # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")


shape = (
    2,
    3,
)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")


# Examples of last data.
data1 = torch.arange(0, 100)

data1.reshape(2, 2, 5, 5)

# print(x_data1[..., -1])

# print(x_data1[:, -1])
