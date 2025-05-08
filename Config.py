import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np

# Set default dtype
torch.set_default_dtype(torch.float64)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device == 'cuda':
    print(torch.cuda.get_device_name())

def data_preparation(gd):
    def to_tensor(data, requires_grad):
        return torch.tensor(data, requires_grad=requires_grad).to(device)

    dlx_up, dly_up = gd.total_point_up[:, 0], gd.total_point_up[:, 1]
    dlx_down, dly_down = gd.total_point_down[:, 0], gd.total_point_down[:, 1]
    nor_vec_up, nor_vec_down = gd.nor_vec_up, gd.nor_vec_down

    tensors = [
        to_tensor(np.array([dlx_up]).T, requires_grad=True),
        to_tensor(np.array([dly_up]).T, requires_grad=True),
        to_tensor(np.array([dlx_down]).T, requires_grad=True),
        to_tensor(np.array([dly_down]).T, requires_grad=True),
        to_tensor(np.array([nor_vec_up]).T, requires_grad=False),
        to_tensor(np.array([nor_vec_down]).T, requires_grad=False),
    ]

    return tensors

def convert_to_tensors(*numpy_arrays):
    """
    Convert multiple NumPy arrays to corresponding PyTorch tensors.

    Parameters:
    *numpy_arrays (np.ndarray): The NumPy arrays to convert.

    Returns:
    torch.Tensor: The converted PyTorch tensors, returned as individual tensor objects.
    """
    tensors = []

    for array in numpy_arrays:
        # Convert list to NumPy array if it's a list
        if isinstance(array, list):
            array = np.array(array)
        elif not isinstance(array, np.ndarray):
            raise ValueError("All inputs must be NumPy arrays or Python lists.")

        # Convert the NumPy array to a PyTorch tensor
        tensor = torch.from_numpy(array).to(device)
        tensors.append(tensor)

    # Unpack the list into separate tensors for return
    return tuple(tensors)

def get_activation_function(activ):
    if activ == 'relu':
        sigma = nn.ReLU()
    elif activ == 'tanh':
        sigma = nn.Tanh()
    elif activ == 'sigmoid':
        sigma = nn.Sigmoid()
    elif activ == 'leakyrelu':
        sigma = nn.LeakyReLU()
    elif activ == 'softplus':
        sigma = nn.Softplus()
    elif activ == 'logsigmoid':
        sigma = nn.LogSigmoid()
    elif activ == 'elu':
        sigma = nn.ELU()
    elif activ == 'gelu':
        sigma = nn.GELU()
    elif activ == 'identity':
        sigma = nn.Identity()
    elif activ == 'swish':
        sigma = nn.SiLU()  # Swish is the same as SiLU
    else:
        raise ValueError(f"Incorrect activation function: {activ}")
    return sigma
