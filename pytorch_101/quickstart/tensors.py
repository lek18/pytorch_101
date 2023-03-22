import numpy as np
import torch

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)


np_array = np.array(data)
x_np = torch.from_numpy(np_array)
