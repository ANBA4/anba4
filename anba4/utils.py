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

def DecoupleStiffness(stiff_matrix):
    K = np.array([[stiff_matrix[i, j] for j in range(6)] for i in range(6)])
    K1 = np.array([[stiff_matrix[i, j] for j in range(3)] for i in range(3)])
    K3 = np.array([[stiff_matrix[i, j+3] for j in range(3)] for i in range(3)])
    Y = np.linalg.solve(K1, -K3)
    I3 = np.eye(3)
    Z3 = np.zeros((3,3))
    TL = np.block([[I3, Z3], [Y.T, I3]])
    TR = np.block([[I3, Y], [Z3, I3]])
    return TL @ K @ TR

def PrincipalAxesRotationAngle(decoupled_stiff_matrix):
    K1 = np.array([[decoupled_stiff_matrix[i, j] for j in range(3)] for i in range(3)])
    K3 = np.array([[decoupled_stiff_matrix[i+3, j+3] for j in range(3)] for i in range(3)])
    (w1, v1) = np.linalg.eig(K1)
    (w3, v3) = np.linalg.eig(K3)
    
    if np.abs(v3[0,0]) < np.abs(v3[0,1]):
        angle = np.arccos(v3[0,0])
    else:
        angle = -np.arcsin(v3[0,1])
    return angle
