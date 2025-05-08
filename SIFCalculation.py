import torch
import math
import numpy as np
from Config import data_preparation


class SIFCalculator:
    def __init__(self, param_dict, material_params_dict, geo_data, boundary_condition, model, extend_function, pinn_train,
                 file_manager):
        self.param_dict = param_dict
        self.material_params_dict = material_params_dict
        self.gd = geo_data
        self.bc = boundary_condition
        self.model = model
        self.ex = extend_function
        self.pt = pinn_train
        self.fm = file_manager

        self.extract_parameters()
    def extract_parameters(self):
        # Extract parameters for convenience
        self.ck1 = self.param_dict['ck1']
        self.ck2 = self.param_dict['ck2']
        self.n_crack_SIF = self.param_dict['n_crack_SIF']
        self.a_crack = self.param_dict['a_crack']
        self.mu = self.param_dict['mu']
        self.q = self.param_dict['q']
        self.b = self.param_dict['b']
        self.G = self.material_params_dict['G']
        self.N_SIF_Start_1 = 3

        self.cp = self.gd.bp1_up
        self.crack_tip_ind = self.bc.n_crack_new


        self.dlx_up, self.dly_up, self.dlx_down, self.dly_down, self.nor_vec_up, self.nor_vec_down = (
            data_preparation(self.gd))

        self.model.eval()
        with torch.no_grad():
            u1_pred_up, u2_pred_up = self.model(self.dlx_up, self.dly_up, self.ck1)
            u1_pred_down, u2_pred_down = self.model(self.dlx_down, self.dly_down, self.ck2)


        # Enriched displacements
        self.u_y_up = u2_pred_up + self.model.enrich_param_up * self.ex.u2_add_up
        self.u_y_down = u2_pred_down + self.model.enrich_param_down * self.ex.u2_add_down

        # Initialize SIF values
        self.SIF_num, self.r_kk = self.calculate_SIF_displacement()
        self.SIF_exact = self.calculate_SIF_exact()
        self.calculate_SIF()

    def calculate_SIF_displacement(self):
        SIF_values = []
        r_kk = []

        for kk in range(self.n_crack_SIF):
            index = self.crack_tip_ind - (kk + 1)

            du_y = self.u_y_up[index] - self.u_y_down[index]

            rh = self.a_crack - self.cp[index, 0]

            if rh <= 0:
                raise ValueError(f"Invalid rh value: {rh}. Check crack tip coordinates and geometry data.")

            pi = torch.tensor(torch.pi)

            # Calculate SIF for current kk
            SIF = (self.G / (4 * (1 - self.mu))) * torch.sqrt(2 * pi / rh) * du_y

            # Clamp SIF to ensure only positive values
            SIF = torch.clamp(SIF, min=0)  # Ensures SIF values are non-negative

            SIF_values.append(SIF)
            r_kk.append(rh)

        # Convert lists to tensors
        SIF_values = torch.tensor(SIF_values, dtype=torch.float64)
        r_kk = torch.tensor(r_kk, dtype=torch.float64)
        print(SIF_values)
        return SIF_values, r_kk

    def calculate_SIF_exact(self):
        # Analytical SIF calculation
        a_b_ratio = self.a_crack / self.b
        correction_factor = 1 - 0.025 * a_b_ratio ** 2 + 0.06 * a_b_ratio ** 4

        if a_b_ratio >= 1 or a_b_ratio <= 0:
            raise ValueError(f"Invalid a/b ratio: {a_b_ratio}. Check crack and geometry parameters.")

        SIF = self.q * math.sqrt(math.pi * self.a_crack) * correction_factor * math.sqrt(
            1 / math.cos(math.pi * self.a_crack / (2 * self.b)))

        return SIF

    def calculate_SIF(self):
        # Define torch linspace for extrapolation
        x_i = torch.linspace(0, self.r_kk[-1], 200, dtype=torch.float64)

        # Perform weighted linear regression for robustness
        r_h_subset = self.r_kk[self.N_SIF_Start_1:self.n_crack_SIF]
        SIF_subset = self.SIF_num[self.N_SIF_Start_1:self.n_crack_SIF]

        if len(r_h_subset) < 2:
            raise ValueError("Insufficient data points for regression. Increase `N_SIF_Start_1` or `n_crack_SIF`.")

        # Weighted least squares regression (optional)
        p_1 = np.polyfit(r_h_subset.cpu().numpy(), SIF_subset.cpu().numpy(), 1)
        p_1 = torch.tensor(p_1, dtype=torch.float64)

        # Calculate extrapolated SIF
        SIF_1_new = x_i * p_1[0] + p_1[1]
        SIF = SIF_1_new[0]

        # Error calculation
        self.SIF_exact = torch.tensor(self.SIF_exact, dtype=torch.float64)
        SIF_error = torch.abs(SIF - self.SIF_exact) / torch.abs(self.SIF_exact)

        # Plotting
        self.fm.plot_sif(self.r_kk, self.SIF_num, x_i, SIF_1_new)
        self.fm.log_sif_result(self.SIF_exact.item(), SIF.item(), SIF_error.item())

        return SIF_1_new, SIF, SIF_error
