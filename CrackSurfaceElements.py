import numpy as np

class CrackSurfaceElements:
    def __init__(self, param_dict):
        self.param_dict = param_dict
        self.extract_parameters()
        self.calculate_elements()

    def extract_parameters(self):
        # Extract and convert parameters from the dictionary.
        self.a_crack = self.param_dict['a_crack']
        self.n_crack = self.param_dict['n_crack']
        self.x_up = self.param_dict['x_up']
        self.y_up = self.param_dict['y_up']
        self.x_up_start, self.x_up_end = self.x_up[0], self.x_up[1]

    def calculate_elements(self):

        # characteristic length scale of crack
        self.dx_crack = self.a_crack / self.n_crack

        # Calculates the remaining length on one side of the crack after accounting for the crack length
        l_remaining_half = (self.x_up_end - self.x_up_start) - self.a_crack

        # number of elements needed to represent the remaining length based on the characteristic length scale
        n_remaining_half = int(l_remaining_half / self.dx_crack)

        # Generates x-coordinates for the crack surface elements
        x_crack = []

        for i in range(self.n_crack + 1):
            x_crack.append(i * self.a_crack / self.n_crack)

        x_half_right = []
        for i in range(n_remaining_half + 1):
            x_half_right.append(self.a_crack + i * l_remaining_half / n_remaining_half)

        # boundary point at interface of upper and lower part
        self.bp_interface = []
        for i in range(len(x_crack) - 1):
            self.bp_interface.append([x_crack[i], self.y_up[0]])

        for i in range(len(x_half_right)):
            self.bp_interface.append([x_half_right[i], self.y_up[0]])

        self.bp_interface = np.array(self.bp_interface)