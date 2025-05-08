import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LossFunctions():

    def __init__(self, model, extend_function, material_params):

        self.model = model
        self.extend_function = extend_function
        self.material_params = material_params

        self.loss_fn = nn.MSELoss()

        # Define body-force functions
        self.fx = self.fy = lambda x, y: torch.zeros_like(x).to(device)

    def autograd_calculation(self, u1, u2, dlx, dly):
        # Calculate first-order and second-order derivatives in a single pass
        u1_x, u1_y, u2_x, u2_y = self.compute_gradients(u1, u2, dlx, dly)
        u1_xx, u1_yy, u1_xy = self.compute_second_order(u1_x, u1_y, dlx, dly)
        u2_xx, u2_yy, u2_xy = self.compute_second_order(u2_x, u2_y, dlx, dly)
        return u1_x, u1_y, u2_x, u2_y, u1_xx, u1_yy, u1_xy, u2_xx, u2_yy, u2_xy

    def compute_gradients(self, u1, u2, dlx, dly):
        u1_x = torch.autograd.grad(u1, dlx, grad_outputs=torch.ones_like(u1), create_graph=True)[0]
        u1_y = torch.autograd.grad(u1, dly, grad_outputs=torch.ones_like(u1), create_graph=True)[0]
        u2_x = torch.autograd.grad(u2, dlx, grad_outputs=torch.ones_like(u2), create_graph=True)[0]
        u2_y = torch.autograd.grad(u2, dly, grad_outputs=torch.ones_like(u2), create_graph=True)[0]
        return u1_x, u1_y, u2_x, u2_y

    def compute_second_order(self, u_x, u_y, dlx, dly):
        u_xx = torch.autograd.grad(u_x, dlx, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, dly, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
        u_xy = torch.autograd.grad(u_x, dly, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        return u_xx, u_yy, u_xy

    def function_loss_PDE(self, u1_xx, u1_yy, u1_xy, u2_xx, u2_yy, u2_xy, nr_inp, dlx, dly):
        c1, c2, c3 = self.material_params['c1'], self.material_params['c2'], self.material_params['c3']
        pre_pde_1 = c1 * u1_xx[nr_inp] + c2 * u1_yy[nr_inp] + c3 * u2_xy[nr_inp]
        pre_pde_2 = c1 * u2_yy[nr_inp] + c2 * u2_xx[nr_inp] + c3 * u1_xy[nr_inp]

        act_pde_1 = self.fx(dlx[nr_inp], dly[nr_inp])
        act_pde_2 = self.fy(dlx[nr_inp], dly[nr_inp])

        loss_pde_1 = self.loss_fn(pre_pde_1, act_pde_1)
        loss_pde_2 = self.loss_fn(pre_pde_2, act_pde_2)

        return loss_pde_1, loss_pde_2

    def function_loss_dirichlet(self, ck, u1, u2, dirichlet_bc_x, dirichlet_bc_y, bc_x, bc_y, interface):
        if ck == 1:
            enrich_up = self.model.enrich_param_up
            u1 = u1 + enrich_up * self.extend_function.u1_add_up
            u2 = u2 + enrich_up * self.extend_function.u2_add_up

        elif ck == 2:
            enrich_down =self.model.enrich_param_down
            u1 = u1 + enrich_down * self.extend_function.u1_add_down
            u2 = u2 + enrich_down * self.extend_function.u2_add_down

        indices = torch.clamp(dirichlet_bc_x.long(), 0, len(bc_x) - 1)

        u1_dirichlet = u1[indices]
        bc_x_dirichlet = bc_x[indices].unsqueeze(-1)
        u2_dirichlet = u2[indices]
        bc_y_dirichlet = bc_y[indices].unsqueeze(-1)

        # Calculate loss_dirichlet_1
        if dirichlet_bc_x.numel() > 0:
            loss_dirichlet_1 = self.loss_fn(u1_dirichlet, bc_x_dirichlet)
        else:
            loss_dirichlet_1 = torch.tensor(0.0, device=u1.device)  # Ensure it is a tensor with the correct device

        # Calculate loss_dirichlet_2
        if dirichlet_bc_y.numel() > 0:
            loss_dirichlet_2 = self.loss_fn(u2_dirichlet, bc_y_dirichlet)
        else:
            loss_dirichlet_2 = torch.tensor(0.0, device=u2.device)  # Ensure it is a tensor with the correct device

        u1_dirichlet_interface = u1[interface]
        u2_dirichlet_interface = u2[interface]

        return loss_dirichlet_1, loss_dirichlet_2, u1_dirichlet_interface, u2_dirichlet_interface

    def function_loss_neumann(self, ck, u1_x, u1_y, u2_x, u2_y, nr_bp, nor_vec, neumann_bc_x, neumann_bc_y,
                              bc_x, bc_y, interface):
        a1, a2, a3 = self.material_params['a1'], self.material_params['a2'], self.material_params['a3']
        if ck == 1:
            enrich_up = self.model.enrich_param_up
            u1_x = u1_x + enrich_up * self.extend_function.u1_x_add_up
            u1_y = u1_y + enrich_up * self.extend_function.u1_y_add_up
            u2_x = u2_x + enrich_up * self.extend_function.u2_x_add_up
            u2_y = u2_y + enrich_up * self.extend_function.u2_y_add_up
        elif ck == 2:
            enrich_down = self.model.enrich_param_down
            u1_x = u1_x + enrich_down * self.extend_function.u1_x_add_down
            u1_y = u1_y + enrich_down * self.extend_function.u1_y_add_down
            u2_x = u2_x + enrich_down * self.extend_function.u2_x_add_down
            u2_y = u2_y + enrich_down * self.extend_function.u2_y_add_down

        f_neumann_x = a1 * nor_vec[0] * u1_x[nr_bp] + a3 * nor_vec[1] * u1_y[nr_bp] + a2 * nor_vec[0] * \
                      u2_y[nr_bp] + a3 * nor_vec[1] * u2_x[nr_bp]
        f_neumann_y = a2 * nor_vec[1] * u1_x[nr_bp] + a3 * nor_vec[0] * u1_y[nr_bp] + a3 * nor_vec[0] * \
                      u2_x[nr_bp] + a1 * nor_vec[1] * u2_y[nr_bp]

        neumann_bc_x = torch.clamp(neumann_bc_x.long(), 0, len(bc_x) - 1)
        neumann_bc_y = torch.clamp(neumann_bc_y.long(), 0, len(bc_y) - 1)

        f_x = f_neumann_x[neumann_bc_x]
        f_y = f_neumann_y[neumann_bc_y]

        bc_x_neumann = bc_x[neumann_bc_x].reshape(f_x.shape)
        bc_y_neumann = bc_y[neumann_bc_y].reshape(f_y.shape)

        loss_neumann_1 = self.loss_fn(f_x, bc_x_neumann)
        loss_neumann_2 = self.loss_fn(f_y, bc_y_neumann)

        n1_neumann_interface = f_neumann_x[interface]
        n2_neumann_interface = f_neumann_y[interface]

        return loss_neumann_1, loss_neumann_2, n1_neumann_interface, n2_neumann_interface

    def function_loss_interface(self, u1_dir_int_up, u1_dir_int_down,
                                u2_dir_int_up, u2_dir_int_down,
                                n1_neu_int_up, n1_neu_int_down,
                                n2_neu_int_up, n2_neu_int_down):

        l_int_dir_x = self.loss_fn(u1_dir_int_up, u1_dir_int_down)
        l_int_dir_y = self.loss_fn(u2_dir_int_up, u2_dir_int_down)

        # Loss calculation for Neumann interfaces
        l_int_neu_x = self.loss_fn(n2_neu_int_up, -n1_neu_int_down)
        l_int_neu_y = self.loss_fn(n1_neu_int_up, -n2_neu_int_down)

        return l_int_dir_x, l_int_dir_y, l_int_neu_x, l_int_neu_y
