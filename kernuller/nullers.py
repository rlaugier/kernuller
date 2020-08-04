import numpy as np
import sympy as sp

from pathlib import Path


resource_path = Path(__file__).parent / "data"

matrixfile = resource_path.joinpath("6T_matrices.npy")
matrices = np.load(matrixfile, allow_pickle=True)
matrices_6T = [sp.Matrix(matrices[i].reshape((21,6))) for i in range(len(matrices))]


matrixfile = resource_path.joinpath("5T_matrices.npy")
matrices = np.load(matrixfile, allow_pickle=True)
matrices_5T = [sp.Matrix(matrices[i].reshape((13,5))) for i in range(len(matrices))]

matrixfile = resource_path.joinpath("4T_matrix.npy")
matrices_4T = [sp.Matrix(np.load(matrixfile, allow_pickle=True))]

matrixfile = resource_path.joinpath("3T_matrix.npy")
matrices_3T = [sp.Matrix(np.load(matrixfile, allow_pickle=True))]