import numpy as np


class BoundaryConditions:
    def __init__(self, param_dict, geo_data):
        self.param_dict = param_dict
        self.geo_data = geo_data

        # Extract parameters and generate boundary condition data
        self.extract_parameters()
        self.generate_boundary_conditions()

    def extract_parameters(self):
        # Extract parameters for convenience.
        self.ck1 = self.param_dict['ck1']
        self.ck2 = self.param_dict['ck2']
        self.a_crack = self.param_dict['a_crack']
        self.q = self.param_dict['q']

    def generate_boundary_conditions(self):
        # Generate boundary conditions for the model.
        self.n_interface_half_new = np.sum(self.geo_data.bp1_up[:, 0] > self.a_crack)
        self.n_crack_new = np.sum(self.geo_data.bp1_up[:, 0] < self.a_crack)

        # Calculate boundary conditions for upper and lower boundaries
        ''' bc_x_up,bc_y_up = BC's in x and y direction,
        dirichlet_bc_x_up, dirichlet_bc_y_up, neumann_bc_x_up, neumann_bc_y_up is the index, where the both BC's apply 
        same as a lowre part 
        '''
        (self.bc_x_up, self.bc_y_up, self.dirichlet_bc_x_up, self.dirichlet_bc_y_up, self.neumann_bc_x_up,
         self.neumann_bc_y_up) = self.calculate_boundary_conditions(self.ck1, self.geo_data.bp_up, self.geo_data.bp1_up,
            self.geo_data.bp2_up, self.geo_data.bp3_up, self.geo_data.bp4_up)

        (self.bc_x_down, self.bc_y_down, self.dirichlet_bc_x_down, self.dirichlet_bc_y_down, self.neumann_bc_x_down,
         self.neumann_bc_y_down) = self.calculate_boundary_conditions(self.ck2, self.geo_data.bp_down,
            self.geo_data.bp1_down, self.geo_data.bp2_down, self.geo_data.bp3_down, self.geo_data.bp4_down)

        # Interface indices
        self.interface = list(range(self.n_crack_new, self.n_crack_new + self.n_interface_half_new))

    def calculate_boundary_conditions(self, ck, bp, bp1, bp2, bp3, bp4):
        n_crack_indices = list(range(0, self.n_crack_new))
        n_interface_indices = list(range(self.n_crack_new, self.n_crack_new + self.n_interface_half_new))

        n2_indices, n3_indices, n4_indices = self.get_boundary_indices(n_interface_indices, len(bp2), len(bp3), len(bp4))
        bc_x, bc_y = self.initialize_boundary_conditions(len(bp))

        # Apply boundary conditions
        dir_bc_x, dir_bc_y = self.apply_dirichlet_conditions(n4_indices)
        neu_bc_x, neu_bc_y = self.apply_neumann_conditions(n_crack_indices, n2_indices, n3_indices, n4_indices)

        # Apply loading on bc_y for boundary n3
        bc_y[n3_indices] = self.apply_loading_on_boundary(ck)

        return bc_x, bc_y, dir_bc_x, dir_bc_y, neu_bc_x, neu_bc_y

    @staticmethod
    def get_boundary_indices(n_interface, n2, n3, n4):
        # Get indices for the boundary points.
        n2_indices = list(range(n_interface[-1] + 1, n_interface[-1] + n2 + 1))
        n3_indices = list(range(n2_indices[-1] + 1, n2_indices[-1] + n3 + 1))
        n4_indices = list(range(n3_indices[-1] + 1, n3_indices[-1] + n4 + 1))

        return n2_indices, n3_indices, n4_indices

    def initialize_boundary_conditions(self, num_bp):
        # Initialize boundary condition arrays.
        return np.zeros(num_bp), np.zeros(num_bp)

    def apply_dirichlet_conditions(self, n4_indices):
        # Apply Dirichlet boundary conditions.
        return [n4_indices], []  # Assuming dirichlet_bc_y is not used, can be adjusted as necessary.

    def apply_neumann_conditions(self, n_crack, n2_indices, n3_indices, n4_indices):
        # Apply Neumann boundary conditions.
        neu_bc_x = n_crack + n2_indices + n3_indices
        neu_bc_y = n_crack + n2_indices + n3_indices + n4_indices
        return neu_bc_x, neu_bc_y

    def apply_loading_on_boundary(self, ck):
        # Apply loading on the boundary depending on ck.
        return self.q if ck == 1 else -self.q
