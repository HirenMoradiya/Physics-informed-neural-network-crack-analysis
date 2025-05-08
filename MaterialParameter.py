class MaterialParameter:
    def __init__(self, param_dict):
        self.param_dict = param_dict

    def get_parameters(self):
        E = self.param_dict['E']
        mu = self.param_dict['mu']
        # shear modulus
        G = E / (2 * (1 + mu))
        # elasticity constants
        c1 = E * (1 - mu) / ((1 - 2 * mu) * (1 + mu))
        c2 = E / (2 * (1 + mu))
        c3 = E / ((1 + mu) * (1 - 2 * mu) * 2)
        # additional parameters
        a1 = E * (1 - mu) / ((1 + mu) * (1 - 2 * mu))
        a2 = E * mu / ((1 + mu) * (1 - 2 * mu))
        a3 = E / (2 * (1 + mu))
        #  Lam ́e parameter
        lame_para = 3 - 4 * mu

        return {
            "G": G,
            "c1": c1,
            "c2": c2,
            "c3": c3,
            "a1": a1,
            "a2": a2,
            "a3": a3,
            "lame_para": lame_para,
        }
