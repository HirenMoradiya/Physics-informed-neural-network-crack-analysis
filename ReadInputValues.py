class ReadInputValues:
    def __init__(self):
        self.param_dict = {}
        self.load_parameters()
        self.validate_and_set_parameters()
        self.calculate_and_update_parameters()

    def load_parameters(self):
        """Reads the input file and populates param_dict with raw values (as strings)."""
        with open("InputData.inp", 'r') as file:
            for line in file:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().split('#')[0].strip()  # Remove comments after #
                    self.param_dict[key] = value  # Keep as string initially

    def validate_and_set_parameters(self):
        """Converts parameters to appropriate types and validates them."""
        param_types = {
            'ck1': int, 'ck2': int, 'q': float, 'E': float, 'mu': float,
            'b': float, 'h': float, 'a_crack': float, 'n_crack': int,
            'refine_ratio': float, 'refine_time': int, 'mesh_density_x': float, 'mesh_density_y': float,
            'input_nodes': int, 'output_nodes': int, 'learning_rate': float, 'epochs': int, 'arch': str,
            'n_crack_SIF': int, 'optimizer_switch_epoch': int, 'name': str
        }

        errors = {'type_errors': [], 'negative_params': []}

        for param, param_type in param_types.items():
            value = self.param_dict.get(param, '0')  # Default to '0' if not found
            try:
                if param == 'use_auto_derivatives':  # Special handling for boolean conversion
                    converted_value = value.lower() in ("true", "1", "yes")  # Convert to True/False
                else:
                    converted_value = param_type(value)  # Convert to correct type
                if not self.is_valid_value(converted_value, param_type):
                    errors['negative_params'].append(param)
                self.param_dict[param] = converted_value  # Store converted value
            except ValueError:
                errors['type_errors'].append(param)

        if errors['type_errors']:
            raise TypeError(f"Parameters with incorrect types: {', '.join(errors['type_errors'])}")
        if errors['negative_params']:
            raise ValueError(f"Parameters should be non-zero or non-negative: {', '.join(errors['negative_params'])}")

    @staticmethod
    def is_valid_value(value, value_type):
        """Checks if the value is valid based on its type."""
        if value_type == int:
            return value > 0
        elif value_type == float:
            return value >= 0
        return True  # Strings don't need validation

    def calculate_and_update_parameters(self):
        """Calculates derived parameters and updates param_dict."""
        x_up = (0, self.param_dict['b'])
        y_up = (0, self.param_dict['h'])
        x_down = (0, self.param_dict['b'])
        y_down = (-self.param_dict['h'], 0)

        refine_length = self.param_dict['a_crack'] * 0.8
        refine_length_x = self.param_dict['a_crack'] + refine_length
        refine_length_y = refine_length

        self.param_dict.update({
            'x_up': x_up,
            'y_up': y_up,
            'x_down': x_down,
            'y_down': y_down,
            'refine_length': refine_length,
            'refine_length_x': refine_length_x,
            'refine_length_y': refine_length_y
        })

    def get_param_dict(self):
        """Returns the parameter dictionary with correctly formatted values."""
        return self.param_dict
