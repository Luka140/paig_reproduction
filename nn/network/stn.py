import tensorflow as tf
import torch.nn.functional as F
import torch

def stn(U, theta, out_size):
    U_pytorch = U#.permute(0, 3, 1, 2).clone().detach()

    num_batch = theta.size(0)
    num_transforms = theta.size(1)
    num_channels = U.size(1)
    theta = theta.view(-1, 2, 3)  # Reshape theta to have shape (num_batch*num_transforms, 2, 3)
    # Adjust the size to match the expected dimensions
    size = torch.Size((num_batch, num_channels, *out_size))
    grid = F.affine_grid(theta, size)
    output = F.grid_sample(U, grid)

    return output #output.permute(0, 2, 3, 1)  # Transpose dimensions to match expected shape

def batch_transformer(U, thetas, out_size):
    U_pytorch = U#.permute(0, 3, 1, 2).clone().detach()
    num_batch, num_transforms = thetas.shape[:2]
    input_repeated = U_pytorch.unsqueeze(1).repeat(1, num_transforms, 1, 1, 1)
    input_repeated = input_repeated.view(-1, *U_pytorch.shape[1:])
    return stn(input_repeated, thetas, out_size)
