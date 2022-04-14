import numpy as np
import sympy as sp

from pathlib import Path
#import pickle


resource_path = Path(__file__).parent / "data"

# The files were compatible with sympy 1.7.1 They are no-longer compatible with sympy 1.9

#matrixfile = resource_path.joinpath("6T_matrices.npy")
#matrices = np.load(matrixfile, allow_pickle=True)
#matrices_6T = [sp.Matrix(matrices[i].reshape((21,6))) for i in range(len(matrices))]


#matrixfile = resource_path.joinpath("5T_matrices.npy")
#matrices = np.load(matrixfile, allow_pickle=True)
#matrices_5T = [sp.Matrix(matrices[i].reshape((13,5))) for i in range(len(matrices))]

#matrixfile = resource_path.joinpath("4T_matrix.npy")
#matrices_4T = [sp.Matrix(np.load(matrixfile, allow_pickle=True))]

#matrixfile = resource_path.joinpath("3T_matrix.npy")
#matrices_3T = [sp.Matrix(np.load(matrixfile, allow_pickle=True))]

######################

#matrixfile = resource_path.joinpath("6T_matrices_p4.pickle")
#with open(matrixfile, "rb") as f:
#    matrices_6T = pickle.load(f)
#matrixfile = resource_path.joinpath("5T_matrices_p4.pickle")
#with open(matrixfile, "rb") as f:
#    matrices_5T = pickle.load(f)
#matrixfile = resource_path.joinpath("4T_matrices_p4.pickle")
#with open(matrixfile, "rb") as f:
#    matrices_4T = pickle.load(f)
#matrixfile = resource_path.joinpath("3T_matrices_p4.pickle")
#with open(matrixfile, "rb") as f:
#    matrices_3T = pickle.load(f)

    
matrixfile = resource_path.joinpath("6T_matrices.txt")
with open(matrixfile, "r") as f:
    matrices_6T = sp.sympify(f.read())
matrixfile = resource_path.joinpath("5T_matrices.txt")
with open(matrixfile, "r") as f:
    matrices_5T = sp.sympify(f.read())
matrixfile = resource_path.joinpath("4T_matrices.txt")
with open(matrixfile, "r") as f:
    matrices_4T = sp.sympify(f.read())
matrixfile = resource_path.joinpath("3T_matrices.txt")
with open(matrixfile, "r") as f:
    matrices_3T = sp.sympify(f.read())