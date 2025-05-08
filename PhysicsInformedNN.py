import torch
import torch.nn as nn
from Config import get_activation_function

# Set default dtype
torch.set_default_dtype(torch.float64)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
if device == 'cuda':
    print(torch.cuda.get_device_name())

print(torch.cuda.is_available())  # Should return True
print(torch.version.cuda)         # Should match system CUDA version (12.8)
print(torch.backends.cudnn.version())  # Should print cuDNN version
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")


class PhysicsInformedNN(nn.Module):

    def __init__(self, param_dict):
        super(PhysicsInformedNN, self).__init__()

        self.param_dict = param_dict

        self.arch = [item.strip().strip("'") if item.strip().startswith("'") and item.strip().endswith("'") else int(
            item.strip()) for item in self.param_dict.get('arch').strip('[]').split(',')]

        self.input_nodes = self.param_dict['input_nodes']
        self.output_nodes = self.param_dict['output_nodes']

        # Upper model and Lower model
        self.model_up = self.build_model()
        self.model_down = self.build_model()

        # Add trainable parameters for upper and lower parts
        self.enrich_param_up = nn.Parameter(torch.ones(1, device=device, requires_grad=True))
        self.enrich_param_down = nn.Parameter(torch.ones(1, device=device, requires_grad=True))

    def build_model(self):
        """Builds the neural network model based on the architecture defined in the param_dict."""
        model = torch.nn.Sequential()
        curr_dim = self.input_nodes
        output_dim = self.output_nodes
        layer_count = 1
        activ_count = 1

        for val in self.arch:  # Check the value (val) instead of the index (i)
            if isinstance(val, int):
                # Add a linear layer
                layer = nn.Linear(curr_dim, val, bias=True)
                # Initialize weights and biases using Xavier initialization
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)  # Initialize biases to zero
                model.add_module('layer ' + str(layer_count), layer)
                curr_dim = val
                layer_count += 1
            elif isinstance(val, str):
                # Add an activation function
                model.add_module('activ ' + str(activ_count), get_activation_function(val))
                activ_count += 1
        model.add_module('layer ' + str(layer_count), torch.nn.Linear(curr_dim, output_dim, bias=True))
        # Move model to the specified device
        model.to(device)
        return model

    def forward(self, dlx, dly,ck):
        # Combine inputs
        xy = torch.cat((dlx, dly), dim=1)

        if ck == 1:
            u_pre = self.model_up(xy)
        elif ck == 2:
            u_pre = self.model_down(xy)
        else:
            raise ValueError("Invalid part specified. Choose 'upper' or 'lower'.")

        u_pre_x = u_pre[:, [0]]
        u_pre_y = u_pre[:, [1]]

        return u_pre_x, u_pre_y

# class PhysicsInformedNN(nn.Module):
#     def __init__(self, param_dict):
#         super(PhysicsInformedNN, self).__init__()
#
#         self.param_dict = param_dict
#         self.arch = [
#             item.strip().strip("'") if item.strip().startswith("'") and item.strip().endswith("'") else int(
#                 item.strip())
#             for item in self.param_dict.get('arch').strip('[]').split(',')
#         ]
#
#         self.input_nodes = self.param_dict['input_nodes']
#         self.output_nodes = self.param_dict['output_nodes']
#
#         # Upper and Lower models
#         self.model_up = self.build_model()
#         self.model_down = self.build_model()
#
#         # Trainable parameters for enrichment
#         self.enrich_param_up = nn.Parameter(torch.ones(1, device=device, requires_grad=True))
#         self.enrich_param_down = nn.Parameter(torch.ones(1, device=device, requires_grad=True))
#
#     def build_model(self):
#         """Builds the neural network model using nn.ModuleList instead of nn.Sequential."""
#         layers = nn.ModuleList()
#         curr_dim = self.input_nodes
#         output_dim = self.output_nodes
#
#         for val in self.arch:
#             if isinstance(val, int):
#                 layer = nn.Linear(curr_dim, val, bias=True)
#                 nn.init.xavier_normal_(layer.weight)
#                 nn.init.zeros_(layer.bias)  # Initialize biases to zero
#                 layers.append(layer)
#                 curr_dim = val
#             elif isinstance(val, str):
#                 layers.append(get_activation_function(val))
#
#         layers.append(nn.Linear(curr_dim, output_dim, bias=True))
#         return layers.to(device)
#
#     def forward_model(self, model, x):
#         """Custom forward pass for handling nn.ModuleList."""
#         for layer in model:
#             x = layer(x)
#         return x
#
#     def forward(self, dlx, dly, ck):
#         xy = torch.cat((dlx, dly), dim=1)
#
#         if ck == 1:
#             u_pre = self.forward_model(self.model_up, xy)
#         elif ck == 2:
#             u_pre = self.forward_model(self.model_down, xy)
#         else:
#             raise ValueError("Invalid part specified. Choose 'upper' or 'lower'.")
#
#         u_pre_x = u_pre[:, [0]]
#         u_pre_y = u_pre[:, [1]]
#
#         return u_pre_x, u_pre_y
