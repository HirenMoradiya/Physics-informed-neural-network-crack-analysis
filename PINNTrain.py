import torch
import time
from Config import data_preparation, convert_to_tensors

torch.autograd.set_detect_anomaly(True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PINNTrain:
    def __init__(self, param_dict, geo_data, boundary_condition, model, loss_function, file_manager):

        self.param_dict = param_dict
        self.geo_data = geo_data
        self.bc = boundary_condition
        self.model = model
        self.loss_function = loss_function
        self.file_manager = file_manager

        self.loss_values = {key: [] for key in ["pde", "dir", "neumann", "interface", "total"]}

        self.extract_parameters()
        # Data Preparation
        self.prepare_data()
        # Model Optimizers
        self.prepare_optimizers()
        # Save the model structure
        self.file_manager.log_model_structure(self.model)
        # train the model
        self.train()

    def extract_parameters(self):
        # Extract parameters for convenience
        self.ck1 = self.param_dict['ck1']
        self.ck2 = self.param_dict['ck2']
        self.epochs = self.param_dict['epochs']
        self.learning_rate = self.param_dict['learning_rate']
        self.optimizer_switch_epoch = self.param_dict['optimizer_switch_epoch']

    def prepare_data(self):
        # Prepares the data needed for training, including boundary conditions and grid points.
        self.dlx_up, self.dly_up, self.dlx_down, self.dly_down, self.nor_vec_up, self.nor_vec_down = data_preparation(
            self.geo_data)

        # Convert boundary conditions and interface data to tensors
        self.dirichlet_bc_x_up, self.dirichlet_bc_y_up, self.dirichlet_bc_x_down, self.dirichlet_bc_y_down, \
            self.neumann_bc_x_up, self.neumann_bc_y_up, self.neumann_bc_x_down, self.neumann_bc_y_down, \
            self.bc_x_up, self.bc_y_up, self.bc_x_down, self.bc_y_down, self.interface = convert_to_tensors(
            self.bc.dirichlet_bc_x_up, self.bc.dirichlet_bc_y_up, self.bc.dirichlet_bc_x_down,
            self.bc.dirichlet_bc_y_down, self.bc.neumann_bc_x_up, self.bc.neumann_bc_y_up,
            self.bc.neumann_bc_x_down, self.bc.neumann_bc_y_down, self.bc.bc_x_up, self.bc.bc_y_up,
            self.bc.bc_x_down, self.bc.bc_y_down, self.bc.interface)

    def prepare_optimizers(self):
        # Prepares the optimizers (LBFGS and Adam) for training the model.
        model_parameters = list(self.model.model_up.parameters()) + list(self.model.model_down.parameters()) + \
                           [self.model.enrich_param_up, self.model.enrich_param_down]

        self.optimizer_adam = torch.optim.Adam(model_parameters, lr=self.learning_rate, weight_decay=1e-5)

        self.optimizer_lbfgs = torch.optim.LBFGS(model_parameters, history_size=50, tolerance_grad=1e-10,
                                                 tolerance_change=1e-20,
                                                 line_search_fn='strong_wolfe')
        # Learning rate scheduler for the main Adam optimizer
        self.scheduler_adam = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_adam, mode='min',
                                                                   factor=0.3, patience=30, min_lr=1e-6)
    def train(self):
        # Record the total training start time
        total_start_time = time.time()

        for epoch in range(self.epochs):
            # Record the start time for the epoch
            start_time = time.time()
            # Switch optimizer based on epoch
            optimizer_main = self.optimizer_adam if epoch < self.optimizer_switch_epoch else self.optimizer_lbfgs

            self.optimize_step(optimizer_main)
            self.scheduler_adam.step(self.total_loss)

            # Calculate the epoch duration
            self.log_epoch_info(epoch, time.time() - start_time)

            # Write a Enrich parameter in Excel file
            self.file_manager.enrich_param_to_excel(epoch, self.model.enrich_param_up, self.model.enrich_param_down,
                                                    self.model.enrich_param_up.grad, self.model.enrich_param_down.grad)

            # Write final results and plot losses at the end of training
            if epoch + 1 == self.epochs:
                self.save_final_results()

        # Add the total training time in output file
        self.file_manager.log_total_time(time.time() - total_start_time)

        # Plot inputs and results after training
        self.plot_results_and_losses()

    def optimize_step(self, optimizer):
        # Step for main model parameters
        if optimizer == self.optimizer_lbfgs:
            optimizer.step(self.closure)
        else:
            optimizer.zero_grad()
            self.forward_pass_and_loss_computation()
            self.total_loss.backward()
            optimizer.step()

    def closure(self):
        # Closure function required for L-BFGS optimizer. It recalculates the loss and clears gradients.
        self.optimizer_lbfgs.zero_grad()  # Zero the gradients
        self.forward_pass_and_loss_computation()  # Compute forward pass and loss
        self.total_loss.backward()  # Compute gradients
        return self.total_loss

    def forward_pass_and_loss_computation(self):

        # Prediction
        self.u1_up, self.u2_up = self.model.forward(self.dlx_up, self.dly_up, self.ck1)
        self.u1_down, self.u2_down = self.model.forward(self.dlx_down, self.dly_down, self.ck2)

        # gradient calculation for upper part
        u1_x_up = torch.autograd.grad(self.u1_up, self.dlx_up, grad_outputs=torch.ones_like(self.u1_up), create_graph=True)[0]
        u1_y_up = torch.autograd.grad(self.u1_up, self.dly_up, grad_outputs=torch.ones_like(self.u1_up), create_graph=True)[0]
        u2_x_up = torch.autograd.grad(self.u2_up, self.dlx_up, grad_outputs=torch.ones_like(self.u2_up), create_graph=True)[0]
        u2_y_up = torch.autograd.grad(self.u2_up, self.dly_up, grad_outputs=torch.ones_like(self.u2_up), create_graph=True)[0]
        u1_xx_up = torch.autograd.grad(u1_x_up, self.dlx_up, grad_outputs=torch.ones_like(u1_x_up), create_graph=True)[0]
        u1_yy_up = torch.autograd.grad(u1_y_up, self.dly_up, grad_outputs=torch.ones_like(u1_y_up), create_graph=True)[0]
        u1_xy_up = torch.autograd.grad(u1_x_up, self.dly_up, grad_outputs=torch.ones_like(u1_x_up), create_graph=True)[0]
        u2_xx_up = torch.autograd.grad(u2_x_up, self.dlx_up, grad_outputs=torch.ones_like(u2_x_up), create_graph=True)[0]
        u2_yy_up = torch.autograd.grad(u2_y_up, self.dly_up, grad_outputs=torch.ones_like(u2_y_up), create_graph=True)[0]
        u2_xy_up = torch.autograd.grad(u2_x_up, self.dly_up, grad_outputs=torch.ones_like(u2_x_up), create_graph=True)[0]

        # gradient calculation for lower part
        u1_x_down = torch.autograd.grad(self.u1_down, self.dlx_down, grad_outputs=torch.ones_like(self.u1_down), create_graph=True)[0]
        u1_y_down = torch.autograd.grad(self.u1_down, self.dly_down, grad_outputs=torch.ones_like(self.u1_down), create_graph=True)[0]
        u2_x_down = torch.autograd.grad(self.u2_down, self.dlx_down, grad_outputs=torch.ones_like(self.u2_down), create_graph=True)[0]
        u2_y_down = torch.autograd.grad(self.u2_down, self.dly_down, grad_outputs=torch.ones_like(self.u2_down), create_graph=True)[0]
        u1_xx_down = torch.autograd.grad(u1_x_down, self.dlx_down, grad_outputs=torch.ones_like(u1_x_down), create_graph=True)[0]
        u1_yy_down = torch.autograd.grad(u1_y_down, self.dly_down, grad_outputs=torch.ones_like(u1_y_down), create_graph=True)[0]
        u1_xy_down = torch.autograd.grad(u1_x_down, self.dly_down, grad_outputs=torch.ones_like(u1_x_down), create_graph=True)[0]
        u2_xx_down = torch.autograd.grad(u2_x_down, self.dlx_down, grad_outputs=torch.ones_like(u2_x_down), create_graph=True)[0]
        u2_yy_down = torch.autograd.grad(u2_y_down, self.dly_down, grad_outputs=torch.ones_like(u2_y_down), create_graph=True)[0]
        u2_xy_down = torch.autograd.grad(u2_x_down, self.dly_down, grad_outputs=torch.ones_like(u2_x_down), create_graph=True)[0]

        # # gradient calculation for upper part
        # (u1_x_up, u1_y_up, u2_x_up, u2_y_up, u1_xx_up, u1_yy_up, u1_xy_up, u2_xx_up, u2_yy_up, u2_xy_up) = \
        #     (self.loss_function.autograd_calculation(self.u1_up, self.u2_up, self.dlx_up, self.dly_up))

        # # gradient calculation for lower part
        # (u1_x_down, u1_y_down, u2_x_down, u2_y_down, u1_xx_down, u1_yy_down, u1_xy_down, u2_xx_down, u2_yy_down,
        #  u2_xy_down) = (self.loss_function.autograd_calculation(self.u1_down, self.u2_down, self.dlx_down,
        #                                                         self.dly_down))

        # Number of Boundary and interior point for upper half
        nr_bp_up = torch.arange(0, len(self.nor_vec_up[0])).to(device)
        nr_inp_up = torch.arange(len(self.nor_vec_up[0]), len(self.dlx_up)).to(device)
        # Number of Boundary and interior point for lower half
        nr_bp_down = torch.arange(0, len(self.nor_vec_down[0])).to(device)
        nr_inp_down = torch.arange(len(self.nor_vec_down[0]), len(self.dlx_down)).to(device)

        # PDE loss for upper part
        self.l_pde_1_up, self.l_pde_2_up = self.loss_function.function_loss_PDE(u1_xx_up, u1_yy_up, u1_xy_up, u2_xx_up,
            u2_yy_up, u2_xy_up, nr_inp_up, self.dlx_up, self.dly_up)

        # PDE loss for lower part
        self.l_pde_1_down, self.l_pde_2_down = self.loss_function.function_loss_PDE(u1_xx_down, u1_yy_down, u1_xy_down,
            u2_xx_down, u2_yy_down, u2_xy_down, nr_inp_down, self.dlx_down, self.dly_down)

        # Dirichlet_loss for upper part
        (self.l_dir_1_up, self.l_dir_2_up, self.u1_dir_int_up, self.u2_dir_int_up) = (
            self.loss_function.function_loss_dirichlet(self.ck1, self.u1_up, self.u2_up, self.dirichlet_bc_x_up,
            self.dirichlet_bc_y_up, self.bc_x_up, self.bc_y_up, self.interface))

        # Dirichlet_loss for lower part
        (self.l_dir_1_down, self.l_dir_2_down, self.u1_dir_int_down, self.u2_dir_int_down) = (
            self.loss_function.function_loss_dirichlet(self.ck2, self.u1_down, self.u2_down, self.dirichlet_bc_x_down,
            self.dirichlet_bc_y_down, self.bc_x_down, self.bc_y_down, self.interface))

        # Neumann_loss for upper part
        self.l_neu_1_up, self.l_neu_2_up, self.n1_neu_int_up, self.n2_neu_int_up = self.loss_function.function_loss_neumann(
            self.ck1, u1_x_up, u1_y_up, u2_x_up, u2_y_up, nr_bp_up, self.nor_vec_up, self.neumann_bc_x_up,
            self.neumann_bc_y_up, self.bc_x_up, self.bc_y_up, self.interface)

        # Neumann_loss for lower part
        self.l_neu_1_down, self.l_neu_2_down, self.n1_neu_int_down, self.n2_neu_int_down = self.loss_function.function_loss_neumann(
            self.ck2, u1_x_down, u1_y_down, u2_x_down, u2_y_down, nr_bp_down, self.nor_vec_down, self.neumann_bc_x_down,
            self.neumann_bc_y_down, self.bc_x_down, self.bc_y_down, self.interface)

        # Loss calculation for Dirichlet and Neumann interfaces
        self.l_int_dir_x, self.l_int_dir_y, self.l_int_neu_x, self.l_int_neu_y = self.loss_function.function_loss_interface(
            self.u1_dir_int_up, self.u1_dir_int_down, self.u2_dir_int_up, self.u2_dir_int_down, self.n1_neu_int_up,
            self.n1_neu_int_down, self.n2_neu_int_up, self.n2_neu_int_down)

        # PDE loss
        self.pde_loss = self.l_pde_1_up + self.l_pde_2_up + self.l_pde_1_down + self.l_pde_2_down
        # Dirichlet loss
        self.dirichlet_loss = self.l_dir_1_up + self.l_dir_2_up + self.l_dir_1_down + self.l_dir_2_down
        # Neumann loss
        self.neumann_loss = self.l_neu_1_up + self.l_neu_2_up + self.l_neu_1_down + self.l_neu_2_down
        # Interface loss
        self.interface_loss = self.l_int_dir_x + self.l_int_dir_y + self.l_int_neu_x + self.l_int_neu_y

        # Total Loss
        self.total_loss = self.pde_loss + self.dirichlet_loss + self.neumann_loss + self.interface_loss

    def log_epoch_info(self, epoch, duration):
        """Logs training loss and duration per epoch."""
        self.loss_values["pde"].append(self.pde_loss.item())
        self.loss_values["dir"].append(self.dirichlet_loss.item())
        self.loss_values["neumann"].append(self.neumann_loss.item())
        self.loss_values["interface"].append(self.interface_loss.item())
        self.loss_values["total"].append(self.total_loss.item())

        self.file_manager.print_and_save_losses(epoch, duration, self.pde_loss, self.dirichlet_loss,
                                                self.neumann_loss, self.interface_loss, self.total_loss)
        print(f"Epoch [{epoch + 1}/{self.epochs}] completed in {duration:.2f} seconds.")

    def save_final_results(self):
        self.file_manager.log_tensor_dimensions({
            "dlx_up": self.dlx_up,
            "dly_up": self.dly_up,
            "u_pre_1_up": self.u1_up,
            "u_pre_2_up": self.u2_up,
            "dlx_down": self.dlx_down,
            "dly_down": self.dly_down,
            "u_pre_1_down": self.u1_down,
            "u_pre_2_down": self.u2_down,
        })
        # Save final results in Excel and log tensor dimensions
        self.file_manager.write_to_excel(self.epochs, self.dlx_up, self.dly_up, self.dlx_down, self.dly_down,
                                         self.u1_up, self.u2_up, self.u1_down, self.u2_down)

    def plot_results_and_losses(self):
        # Plot results after training
        # self.fm.plot_input(self.dlx_up, self.dly_up, self.dlx_down, self.dly_down)
        self.file_manager.plot_results(self.dlx_up, self.dly_up, self.dlx_down, self.dly_down,
                                       self.u1_up, self.u2_up, self.u1_down, self.u2_down)
        self.file_manager.plot_losses(self.loss_values)
