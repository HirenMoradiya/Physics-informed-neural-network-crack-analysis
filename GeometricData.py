'''
bp_interface
n1, n2, n3, n4
bp1_up, bp2_up, bp3_up, bp4_up
inp_up, nor_vec_up, bp_up, s_up

bp1_down, bp2_down, bp3_down, bp4_down
inp_down, nor_vec_down, bp_down, s_down
'''
import numpy as np

class GeometricData:
    def __init__(self, param_dict, file_manager):
        self.param_dict = param_dict
        self.file_manager = file_manager

        # Extract parameters for convenience
        self.extract_parameters()

        # Generate geometric data
        self.generate_geometric_data()

    def extract_parameters(self):
        # Extract parameters for convenience
        self.ck1 = self.param_dict['ck1']
        self.ck2 = self.param_dict['ck2']
        self.a_crack = self.param_dict['a_crack']
        self.n_crack = self.param_dict['n_crack']
        self.refine_ratio = self.param_dict['refine_ratio']
        self.refine_time = self.param_dict['refine_time']
        self.x_up = self.param_dict['x_up']
        self.y_up = self.param_dict['y_up']
        self.x_down = self.param_dict['x_down']
        self.y_down = self.param_dict['y_down']
        self.refine_length_x = self.param_dict['refine_length_x']
        self.refine_length_y = self.param_dict['refine_length_y']

    def generate_geometric_data(self):

        self.dx_crack, self.bp_interface = self.calculate_elements()
        self.ny_up = self.calculate_ny(self.y_up)
        self.ny_down = self.calculate_ny(self.y_down)

        # Process for ck = 1 (upper part)
        self.bp1_up, self.bp2_up, self.bp3_up, self.bp4_up, self.inp_up, self.nor_vec_up, self.bp_up, self.total_point_up = (
            self.create_rectangle_data(self.refine_length_x, self.refine_length_y, self.x_up, self.y_up, self.ny_up, self.bp_interface, self.ck1))

        # Process for ck = 2 (lower part)
        (self.bp1_down, self.bp2_down, self.bp3_down, self.bp4_down, self.inp_down, self.nor_vec_down, self.bp_down,
         self.total_point_down) = self.create_rectangle_data(self.refine_length_x, self.refine_length_y, self.x_down,
            self.y_down, self.ny_down, self.bp_interface, self.ck2)

        self.file_manager.plot_geometric_data(self.bp1_up, self.bp2_up, self.bp3_up, self.bp4_up, self.inp_up,
                             self.bp1_down, self.bp2_down, self.bp3_down, self.bp4_down, self.inp_down)

    def calculate_elements(self):

        # characteristic length scale of crack
        dx_crack = self.a_crack / self.n_crack

        # Calculates the remaining length on one side of the crack after accounting for the crack length
        l_remaining_half = (self.x_up[1] - self.x_up[0]) - self.a_crack

        # number of elements needed to represent the remaining length based on the characteristic length scale
        n_remaining_half = int(l_remaining_half / dx_crack)

        # Generates x-coordinates for the crack surface elements
        x_crack = []

        for i in range(self.n_crack + 1):
            x_crack.append(i * self.a_crack / self.n_crack)

        x_half_right = []
        for i in range(n_remaining_half + 1):
            x_half_right.append(self.a_crack + i * l_remaining_half / n_remaining_half)

        # boundary point at interface of upper and lower part
        bp_interface = []
        for i in range(len(x_crack) - 1):
            bp_interface.append([x_crack[i], self.y_up[0]])

        for i in range(len(x_half_right)):
            bp_interface.append([x_half_right[i], self.y_up[0]])

        bp_interface = np.array(bp_interface)

        return dx_crack, bp_interface

    def calculate_ny(self, y):
        # Calculate the number of elements in y-direction
        return int((y[1] - y[0]) / self.dx_crack) + 1

    def create_rectangle_data(self, refine_length_x, refine_length_y, x, y, ny, bp_interface, ck):
        error = 1e-8

        # Initialize temporary variables
        x_temp = bp_interface[:, 0]
        y_temp = np.linspace(y[0], y[1], ny)
        nx = len(x_temp)

        # Generate grid and initial input points
        Allx, Ally = np.meshgrid(x_temp, y_temp)
        inp = np.column_stack((Allx.ravel(), Ally.ravel()))

        # Prepare boundary points and normals
        bp2, bp3, n2, n3 = self.initialize_boundary_points(nx, ny, x, y, x_temp, y_temp, ck)

        # Set refinement step sizes
        dx = self.a_crack / self.n_crack
        dy = (y[1] - y[0]) / (ny - 1)

        # Refine the grid
        for _ in range(self.refine_time):
            inp, x_temp, dx, dy, y_temp = self.node_refine(ck, inp, x_temp, dx, dy, refine_length_x,
                                                           refine_length_y,y_temp)

            refine_length_x = self.a_crack + (refine_length_x - self.a_crack) * self.refine_ratio
            refine_length_y *= self.refine_ratio

        # Create additional boundary points and normals
        bp1, bp4, n1, n4 = self.create_additional_boundaries(ck, x_temp, y_temp, x, y)

        # Filter input points
        inp = self.filter_input_points(inp, x, y, y_large=y[1], y_small=y[0], error=error, ck=ck)

        # Remove crack tip points
        bp1, n1 = self.remove_crack_tip_points(bp1, n1)

        # Combine all boundary points and normals
        n = np.vstack((n1, n2, n3, n4))
        bp = np.vstack((bp1, bp2, bp3, bp4))
        total_point = np.vstack((bp, inp))

        return bp1, bp2, bp3, bp4, inp, n, bp, total_point

    @staticmethod
    def node_refine(ck, inp, x_temp, dx, dy, refine_length_x, refine_length_y, y_temp):
        error = 1e-8

        # Filter indices based on the refine length
        inp_index = np.where((np.abs(inp[:, 0]) < refine_length_x - error) &
                             (np.abs(inp[:, 1]) < refine_length_y - error))[0]
        inp_1 = inp[inp_index]

        # Adjust dx and dy based on ck value
        if ck == 2:
            dy = -dy

        # Generate new points by combining dx and dy shifts
        shifts = [(dx / 2, dy / 2), (0, dy / 2), (dx / 2, 0)]
        inp_add = np.vstack([inp_1 + shift for shift in shifts])

        # Update input points
        inp = np.vstack((inp, inp_add))

        # Update and sort x_temp
        x_temp_add = x_temp[np.abs(x_temp) < refine_length_x - error] + dx / 2
        x_temp = np.sort(np.concatenate((x_temp, x_temp_add)))

        # Update and sort y_temp
        y_temp_add = y_temp[np.abs(y_temp) < refine_length_y - error] + dy / 2
        y_temp = np.sort(np.concatenate((y_temp, y_temp_add)))

        # Halve dx and dy for the next refinement
        dx_new, dy_new = dx / 2, dy / 2

        return inp, x_temp, dx_new, dy_new, y_temp

    @staticmethod
    def initialize_boundary_points(nx, ny, x, y, x_temp, y_temp, ck):
        # Initialize boundary points and normals based on ck value.
        N2, N3 = ny - 2, nx - 2
        bp2, bp3 = np.zeros((N2, 2)), np.zeros((N3, 2))
        n2, n3 = np.zeros((N2, 2)), np.zeros((N3, 2))

        if ck == 1:
            bp2[:, 0], bp2[:, 1] = x[1], y_temp[1:ny - 1]
            bp3[:, 0], bp3[:, 1] = x_temp[1:nx - 1], y[1]
            n2[:, 0], n3[:, 1] = 1, 1
        elif ck == 2:
            bp2[:, 0], bp2[:, 1] = x[1], y_temp[1:ny - 1]
            bp3[:, 0], bp3[:, 1] = x_temp[1:nx - 1], y[0]
            n2[:, 0], n3[:, 1] = 1, -1

        return bp2, bp3, n2, n3

    @staticmethod
    def create_additional_boundaries(ck, x_temp, y_temp, x, y):
        # Create boundary points and normals for bp1 and bp4.
        bp1 = np.zeros((len(x_temp), 2))
        bp4 = np.zeros((len(y_temp) - 2, 2))
        n1 = np.zeros((len(bp1), 2))
        n4 = np.zeros((len(bp4), 2))

        if ck == 1:
            bp1[:, 0], bp1[:, 1] = x_temp, y[0]
            bp4[:, 0], bp4[:, 1] = x[0], y_temp[1:-1]
            n1[:, 1], n4[:, 0] = -1, -1
        elif ck == 2:
            bp1[:, 0], bp1[:, 1] = x_temp, y[1]
            bp4[:, 0], bp4[:, 1] = x[0], y_temp[1:-1]
            n1[:, 1], n4[:, 0] = 1, -1

        return bp1, bp4, n1, n4

    @staticmethod
    def filter_input_points(inp, x, y, y_large, y_small, error, ck):
        # Filter input points based on the refinement boundaries.
        if ck == 2:
            y_large, y_small = -y[0], y[1]

        inp_index = np.where(
            (inp[:, 0] < x[1] - error) &
            (inp[:, 0] > x[0] + error) &
            (np.abs(inp[:, 1]) < y_large - error) &
            (np.abs(inp[:, 1]) > y_small + error)
        )[0]

        return inp[inp_index, :]

    def remove_crack_tip_points(self, bp1, n1):
        # Remove points near the crack tip.
        crack_tip_index = np.where(
            (np.abs(bp1[:, 0]) >= self.a_crack - 1e-8) &
            (np.abs(bp1[:, 0]) <= self.a_crack + 1e-8)
        )[0]

        bp1 = np.delete(bp1, crack_tip_index, axis=0)
        n1 = np.delete(n1, crack_tip_index, axis=0)

        return bp1, n1