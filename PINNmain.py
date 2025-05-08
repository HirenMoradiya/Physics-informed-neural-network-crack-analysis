from ReadInputValues import ReadInputValues
from MaterialParameter import MaterialParameter
from GeometricData import GeometricData
from BoundaryConditions import BoundaryConditions
from PhysicsInformedNN import PhysicsInformedNN
from ExtendFunction import ExtendFunction
from LossFunction import LossFunctions
from PINNTrain import PINNTrain
from SIFCalculation import SIFCalculator
from FileManager import FileManager


def main():
    # Initialize the input values
    read_input = ReadInputValues()
    param_dict = read_input.get_param_dict()

    # Initialize the material parameters
    material_parameter = MaterialParameter(param_dict)
    material_params = material_parameter.get_parameters()

    # Create an instance of File Manager
    file_manager = FileManager(param_dict)

    # # Create an instance of GeometricData
    geo_data = GeometricData(param_dict, file_manager)

    # Create an instance of BoundaryConditions
    boundary_condition = BoundaryConditions(param_dict, geo_data)

    # Create an instance of Expended function
    extend_function = ExtendFunction(param_dict, material_params, geo_data, file_manager)

    # Create an instance of PhysicsInformedNN
    model = PhysicsInformedNN(param_dict)

    # Create an instance of Loss Function
    loss_function = LossFunctions(model, extend_function, material_params)

    # Create an instance of PINN Train
    pinn_train = PINNTrain(param_dict, geo_data, boundary_condition, model, loss_function, file_manager)

    # Create an instance of SIF Calculator
    sif = SIFCalculator(param_dict, material_params, geo_data, boundary_condition, model, extend_function,
                        pinn_train, file_manager)


if __name__ == "__main__":
    main()
