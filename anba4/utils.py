import numpy as np

def ComputeShearCenter(stiff_matrix):
    K1 = np.array([[stiff_matrix[i, j] for j in range(3)] for i in range(3)])
    K3 = np.array([[stiff_matrix[i, j+3] for j in range(3)] for i in range(3)])
    Y = np.linalg.solve(K1, -K3)
    return [-Y[1,2], Y[0,2]]

def ComputeTensionCenter(stiff_matrix):
    K1 = np.array([[stiff_matrix[i, j] for j in range(3)] for i in range(3)])
    K3 = np.array([[stiff_matrix[i, j+3] for j in range(3)] for i in range(3)])
    Y = np.linalg.solve(K1, -K3)
    return [Y[2,1], -Y[2,0]]

def ComputeMassCenter(mass_matrix):
    M1 = np.array([[mass_matrix[i, j] for j in range(3)] for i in range(3)])
    M3 = np.array([[mass_matrix[i, j+3] for j in range(3)] for i in range(3)])
    Y = np.linalg.solve(M1, -M3)
    return [Y[2,1], -Y[2,0]]
