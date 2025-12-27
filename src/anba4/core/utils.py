"""Utility functions for beam analysis."""

import numpy as np
import dolfin

from .voight_notation import (
    stressVectorToStressTensor,
    strainVectorToStrainTensor,
    strainTensorToStrainVector,
)
from ..data.data_model import AnbaData


def ComputeShearCenter(stiff_matrix):
    """Compute shear center."""
    K1 = np.array([[stiff_matrix[i, j] for j in range(3)] for i in range(3)])
    K3 = np.array([[stiff_matrix[i, j + 3] for j in range(3)] for i in range(3)])
    Y = np.linalg.solve(K1, -K3)
    return [-Y[1, 2], Y[0, 2]]


def ComputeTensionCenter(stiff_matrix):
    """Compute tension center."""
    K1 = np.array([[stiff_matrix[i, j] for j in range(3)] for i in range(3)])
    K3 = np.array([[stiff_matrix[i, j + 3] for j in range(3)] for i in range(3)])
    Y = np.linalg.solve(K1, -K3)
    return [Y[2, 1], -Y[2, 0]]


def ComputeMassCenter(mass_matrix):
    """Compute mass center."""
    M1 = np.array([[mass_matrix[i, j] for j in range(3)] for i in range(3)])
    M3 = np.array([[mass_matrix[i, j + 3] for j in range(3)] for i in range(3)])
    Y = np.linalg.solve(M1, -M3)
    return [Y[2, 1], -Y[2, 0]]


def DecoupleStiffness(stiff_matrix):
    """Decouple stiffness matrix."""
    K = np.array([[stiff_matrix[i, j] for j in range(6)] for i in range(6)])
    K1 = np.array([[stiff_matrix[i, j] for j in range(3)] for i in range(3)])
    K3 = np.array([[stiff_matrix[i, j + 3] for j in range(3)] for i in range(3)])
    Y = np.linalg.solve(K1, -K3)
    I3 = np.eye(3)
    Z3 = np.zeros((3, 3))
    TL = np.block([[I3, Z3], [Y.T, I3]])
    TR = np.block([[I3, Y], [Z3, I3]])
    return TL @ K @ TR


def PrincipalAxesRotationAngle(decoupled_stiff_matrix):
    """Compute principal axes rotation angle."""
    K1 = np.array([[decoupled_stiff_matrix[i, j] for j in range(3)] for i in range(3)])
    K3 = np.array(
        [[decoupled_stiff_matrix[i + 3, j + 3] for j in range(3)] for i in range(3)]
    )
    (w1, v1) = np.linalg.eig(K1)
    (w3, v3) = np.linalg.eig(K3)

    if np.abs(v3[0, 0]) < np.abs(v3[0, 1]):
        angle = np.arccos(v3[0, 0])
    else:
        angle = -np.arcsin(v3[0, 1])
    return angle


def pos3d(POS):
    """Return node coordinates Vector."""
    return dolfin.as_vector([POS[0], POS[1], 0.0])


def grad3d(u, up):
    """Return 3d gradient."""
    g = dolfin.grad(u)
    return dolfin.as_tensor(
        [
            [g[0, 0], g[0, 1], up[0]],
            [g[1, 0], g[1, 1], up[1]],
            [g[2, 0], g[2, 1], up[2]],
        ]
    )


def epsilon(u, up):
    """Return symmetric 3D infinitesimal strain tensor."""
    g3 = grad3d(u, up)
    return 0.5 * (g3.T + g3)


def rotated_epsilon(data: AnbaData, u, up):
    """Return symmetric 3D infinitesimal strain tensor rotated into material reference."""
    eps = epsilon(u, up)
    rot = data.material_data.MaterialRotation_matrix
    rotMatrix = dolfin.as_matrix(
        (
            (rot[0], rot[1], rot[2], rot[3], rot[4], rot[5]),
            (rot[6], rot[7], rot[8], rot[9], rot[10], rot[11]),
            (rot[12], rot[13], rot[14], rot[15], rot[16], rot[17]),
            (rot[18], rot[19], rot[20], rot[21], rot[22], rot[23]),
            (rot[24], rot[25], rot[26], rot[27], rot[28], rot[29]),
            (rot[30], rot[31], rot[32], rot[33], rot[34], rot[35]),
        )
    )
    roteps = strainVectorToStrainTensor(rotMatrix.T * strainTensorToStrainVector(eps))
    return roteps


def sigma_helper(mod, u, up):
    """Return second Piola-Kirchhoff stress tensor."""
    et = epsilon(u, up)
    ev = strainTensorToStrainVector(et)
    elasticMatrix = dolfin.as_matrix(
        (
            (mod[0], mod[1], mod[2], mod[3], mod[4], mod[5]),
            (mod[6], mod[7], mod[8], mod[9], mod[10], mod[11]),
            (mod[12], mod[13], mod[14], mod[15], mod[16], mod[17]),
            (mod[18], mod[19], mod[20], mod[21], mod[22], mod[23]),
            (mod[24], mod[25], mod[26], mod[27], mod[28], mod[29]),
            (mod[30], mod[31], mod[32], mod[33], mod[34], mod[35]),
        )
    )
    sv = elasticMatrix * ev
    st = stressVectorToStressTensor(sv)
    return st


def Sigma(data: AnbaData, u, up):
    """Return second Piola-Kirchhoff stress tensor."""
    return sigma_helper(data.material_data.modulus, u, up)


def RotatedSigma(data: AnbaData, u, up):
    """Return second Piola-Kirchhoff stress tensor."""
    return sigma_helper(data.material_data.RotatedStress_modulus, u, up)


def local_project(v, V, u=None):
    """Element-wise projection using LocalSolver."""
    dv = dolfin.TrialFunction(V)
    v_ = dolfin.TestFunction(V)
    a_proj = dolfin.inner(dv, v_) * dolfin.dx
    b_proj = dolfin.inner(v, v_) * dolfin.dx
    solver = dolfin.LocalSolver(a_proj, b_proj)
    solver.factorize()
    if u is None:
        u = dolfin.Function(V)
        solver.solve_local_rhs(u)
        return u
    else:
        solver.solve_local_rhs(u)
        return
