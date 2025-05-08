import torch
import numpy as np
from Config import data_preparation


class ExtendFunction:

    def __init__(self, param_dict, material_params_dict, geo_data,file_manager):

        self.param_dict = param_dict
        self.material_params_dict = material_params_dict
        self.gd = geo_data
        self.file_manager = file_manager

        self.dlx_up, self.dly_up, self.dlx_down, self.dly_down, self.nor_vec_up, self.nor_vec_down = data_preparation(
            self.gd)

        # Extract parameters for convenience
        self.extract_parameters()

        self.extend_function()

    def extract_parameters(self):
        # Extract parameters for convenience
        self.a_crack = self.param_dict['a_crack']
        self.ck1 = self.param_dict['ck1']
        self.ck2 = self.param_dict['ck2']
        self.mu = self.param_dict['mu']
        self.G = self.material_params_dict['G']

    def extend_function(self):
        self.u1_add_up, self.u2_add_up, self.u1_x_add_up, self.u1_y_add_up, self.u2_x_add_up, self.u2_y_add_up = self.compute_extended_param(
            self.ck1, self.dlx_up, self.dly_up)
        self.u1_add_down, self.u2_add_down, self.u1_x_add_down, self.u1_y_add_down, self.u2_x_add_down, self.u2_y_add_down = self.compute_extended_param(
            self.ck2, self.dlx_down, self.dly_down)

        self.file_manager.extend_function_excel(self.u1_add_up, self.u2_add_up, self.u1_x_add_up, self.u1_y_add_up,
            self.u2_x_add_up, self.u2_y_add_up, self.u1_add_down, self.u2_add_down, self.u1_x_add_down,
            self.u1_y_add_down, self.u2_x_add_down, self.u2_y_add_down)

    def compute_extended_param(self, ck, dlx, dly):
        x_new, y_new = dlx - self.a_crack, dly
        y_new = y_new + (1e-23 if ck == 1 else -1e-23)

        rho, theta = self.compute_polar_coords(x_new, y_new, ck)

        u1, u2 = self.calculate_displacement_fields(rho, theta)

        # Use manual derivatives
        u1_x, u1_y, u2_x, u2_y = self.calculate_gradients(x_new, y_new, ck)

        # Use .clone().detach() directly instead of torch.tensor() to avoid the warning
        return (t.clone().detach() for t in [u1, u2, u1_x, u1_y, u2_x, u2_y])

    def compute_polar_coords(self, x, y, ck):
        # Small epsilon to prevent division by zero in the case where rho is very close to zero
        epsilon = 1e-23
        rho = torch.sqrt(x ** 2 + y ** 2) + epsilon
        theta = torch.acos(x / rho) * (1 if ck == 1 else -1)
        return rho, theta

    def calculate_displacement_fields(self, rho, theta):
        factor = torch.sqrt(rho / (2 * np.pi)) / self.G
        u1 = factor * torch.cos(theta / 2) * (1 - 2 * self.mu + torch.sin(theta / 2) ** 2)
        u2 = factor * torch.sin(theta / 2) * (2 - 2 * self.mu - torch.cos(theta / 2) ** 2)

        return u1, u2

    def calculate_gradients(self, x_new, y_new, ck):
        # helper term
        pi = torch.tensor(torch.pi)
        den_term_1 = 4 * self.G * torch.sqrt(pi) * (x_new ** 2 + y_new ** 2) ** (3 / 4)
        den_term_2 = 4 * self.G * torch.sqrt(pi) * (x_new ** 2 + y_new ** 2) ** (7 / 4) * torch.sqrt(
            y_new ** 2 / (x_new ** 2 + y_new ** 2))
        den_term_3 = 4 * self.G * torch.sqrt(pi) * (x_new ** 2 + y_new ** 2) ** (5 / 4) * torch.sqrt(
            y_new ** 2 / (x_new ** 2 + y_new ** 2))

        arccos_term = torch.acos(x_new / torch.sqrt(x_new ** 2 + y_new ** 2)) / 2

        num_term_1 = 2 * self.mu + (x_new / (2 * torch.sqrt(x_new ** 2 + y_new ** 2))) - 3 / 2
        num_term_2 = x_new + torch.sqrt(x_new ** 2 + y_new ** 2)
        num_term_3 = x_new - torch.sqrt(x_new ** 2 + y_new ** 2)

        if ck == 1:
            u1_x = -((2 ** 0.5) * x_new * torch.cos(arccos_term) * num_term_1 / den_term_1) \
                   - ((2 ** 0.5) * y_new ** 2 * torch.sin(arccos_term) * num_term_2 / den_term_2) \
                   - ((2 ** 0.5) * y_new ** 2 * torch.sin(arccos_term) * num_term_1 / den_term_3)

            u1_y = ((2 ** 0.5) * x_new * y_new * torch.sin(arccos_term) * num_term_2 / den_term_2) \
                   - ((2 ** 0.5) * y_new * torch.cos(arccos_term) * num_term_1 / den_term_1) \
                   + ((2 ** 0.5) * x_new * y_new * torch.sin(arccos_term) * num_term_1 / den_term_3)

            u2_x = ((2 ** 0.5) * y_new ** 2 * torch.cos(arccos_term) * num_term_1 / den_term_3) \
                   - ((2 ** 0.5) * x_new * torch.sin(arccos_term) * num_term_1 / den_term_1) \
                   + ((2 ** 0.5) * y_new ** 2 * torch.cos(arccos_term) * num_term_3 / den_term_2)

            u2_y = -((2 ** 0.5) * y_new * torch.sin(arccos_term) * num_term_1 / den_term_1) \
                   - ((2 ** 0.5) * x_new * y_new * torch.cos(arccos_term) * num_term_1 / den_term_3) \
                   - ((2 ** 0.5) * x_new * y_new * torch.cos(arccos_term) * num_term_3 / den_term_2)

        elif ck == 2:

            u1_x = -((2 ** 0.5) * x_new * torch.cos(arccos_term) * num_term_1 / den_term_1) \
                   - ((2 ** 0.5) * y_new ** 2 * torch.sin(arccos_term) * num_term_2 / den_term_2) \
                   - ((2 ** 0.5) * y_new ** 2 * torch.sin(arccos_term) * num_term_1 / den_term_3)

            u1_y = ((2 ** 0.5) * x_new * y_new * torch.sin(arccos_term) * num_term_2 / den_term_2) \
                   - ((2 ** 0.5) * y_new * torch.cos(arccos_term) * num_term_1 / den_term_1) \
                   + ((2 ** 0.5) * x_new * y_new * torch.sin(arccos_term) * num_term_1 / den_term_3)

            u2_x = ((2 ** 0.5) * x_new * torch.sin(arccos_term) * num_term_1 / den_term_1) \
                   - ((2 ** 0.5) * y_new ** 2 * torch.cos(arccos_term) * num_term_1 / den_term_3) \
                   - ((2 ** 0.5) * y_new ** 2 * torch.cos(arccos_term) * num_term_3 / den_term_2)

            u2_y = ((2 ** 0.5) * y_new * torch.sin(arccos_term) * num_term_1 / den_term_1) \
                   + ((2 ** 0.5) * x_new * y_new * torch.cos(arccos_term) * num_term_1 / den_term_3) \
                   + ((2 ** 0.5) * x_new * y_new * torch.cos(arccos_term) * num_term_3 / den_term_2)

        return u1_x, u1_y, u2_x, u2_y


