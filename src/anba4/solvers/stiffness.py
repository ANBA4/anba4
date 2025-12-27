#
# Copyright (C) 2018 Marco Morandini
#
# ----------------------------------------------------------------------
#
#    This file is part of Anba.
#
#    Anba is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Anba is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Anba.  If not, see <https://www.gnu.org/licenses/>.
#
# ----------------------------------------------------------------------
#

from dolfin import (
    derivative,
    inner,
    dx,
    as_tensor,
    assemble,
    grad,
    Constant,
    dot,
    cross,
    sqrt,
    solve,
    as_backend_type,
    PETScKrylovSolver,
)
from petsc4py import PETSc

import numpy as np

from typing import Any

from ..utils import Sigma, pos3d
from ..data.data_model import AnbaData


def compute_stiffness(data: AnbaData) -> Any:
    singular = data.input_data.singular
    if singular:
        U = data.fe_functions.U
        UP = data.fe_functions.UP
        UV = data.fe_functions.UV
        UT = data.fe_functions.UT
    else:
        U = data.fe_functions.U
        UP = data.fe_functions.UP
        UV = data.fe_functions.UV
        UT = data.fe_functions.UT
        LV = data.fe_functions.LV
        LT = data.fe_functions.LT

    stress = Sigma(data, U, UP)
    stress_n = stress[:, 2]
    stress_1 = stress[:, 0]
    stress_2 = stress[:, 1]
    stress_s = as_tensor(
        [
            [stress_1[0], stress_2[0]],
            [stress_1[1], stress_2[1]],
            [stress_1[2], stress_2[2]],
        ]
    )

    derivative(stress, U, UT)
    ES_t = derivative(stress_s, U, UT)
    ES_n = derivative(stress_s, UP, UT)
    En_t = derivative(stress_n, U, UT)
    En_n = derivative(stress_n, UP, UT)

    Mf = inner(UV, En_n) * dx
    M = assemble(Mf)
    data.output_data.M = M

    Cf = inner(grad(UV), ES_n) * dx
    C = assemble(Cf)
    Hf = (inner(grad(UV), ES_n) - inner(UV, En_t)) * dx
    H = assemble(Hf)
    data.output_data.H = H

    if singular:
        Ef = inner(grad(UV), ES_t) * dx
        E = assemble(Ef)
        data.output_data.E = E

        solver = PETScKrylovSolver("cg")
        solver.parameters["relative_tolerance"] = 1.0e-10
        solver.parameters["absolute_tolerance"] = 1.0e-16
        solver.parameters["convergence_norm_type"] = "natural"
        solver.set_operator(E)
        as_backend_type(E).set_nullspace(data.output_data.null_space)
    else:
        Escal = Constant(data.input_data.scaling_constraint)
        Ef = inner(grad(UV), ES_t) * dx
        Ef += (LV[0] * UT[0] + LV[1] * UT[1] + LV[2] * UT[2]) * Escal * dx
        Ef += LV[3] * dot(UT, data.chains.chains_d[1][0]) * Escal * dx
        Ef += (UV[0] * LT[0] + UV[1] * LT[1] + UV[2] * LT[2]) * Escal * dx
        Ef += LT[3] * dot(UV, data.chains.chains_d[1][0]) * Escal * dx
        E = assemble(Ef)
        data.output_data.E = E

    S = (
        dot(stress_n, data.fe_functions.RV3F) * dx
        + dot(cross(pos3d(data.fe_functions.POS), stress_n), data.fe_functions.RV3M)
        * dx
    )
    L_res_f = derivative(S, UP, UT)
    data.output_data.L_res = assemble(L_res_f)
    R_res_f = derivative(S, U, UT)
    data.output_data.R_res = assemble(R_res_f)

    maxres = 0.0
    for i in range(4):
        tmp = E * data.chains.chains[i][0].vector()
        maxres = max(maxres, sqrt(tmp.inner(tmp)))
    for i in [2, 3]:
        tmp = -(H * data.chains.chains[i][0].vector()) - (
            E * data.chains.chains[i][1].vector()
        )
        maxres = max(maxres, sqrt(tmp.inner(tmp)))

    # solve E d1 = -H d0
    for i in range(2):
        rhs = -(H * data.chains.chains[i][0].vector())
        data.output_data.null_space.orthogonalize(rhs)
        if singular:
            solver.solve(data.chains.chains[i][1].vector(), rhs)
        else:
            solve(E, data.chains.chains[i][1].vector(), rhs)
        data.output_data.null_space.orthogonalize(data.chains.chains[i][1].vector())

    # solve E d2 = M d0 - H d1
    for i in [2, 3]:
        rhs = -(H * data.chains.chains[i][1].vector()) + (
            M * data.chains.chains[i][0].vector()
        )
        data.output_data.null_space.orthogonalize(rhs)
        if singular:
            solver.solve(data.chains.chains[i][2].vector(), rhs)
        else:
            solve(E, data.chains.chains[i][2].vector(), rhs)
        data.output_data.null_space.orthogonalize(data.chains.chains[i][2].vector())

    a = np.zeros((2, 2))
    b = np.zeros((2, 1))
    for i in [2, 3]:
        res = -(H * data.chains.chains[i][2].vector()) + (
            M * data.chains.chains[i][1].vector()
        )
        for k in range(2):
            b[k] = res.inner(data.chains.chains[k][0].vector())
            for ii in range(2):
                a[k, ii] = (
                    -(H * data.chains.chains[ii][1].vector())
                    + (M * data.chains.chains[ii][0].vector())
                ).inner(data.chains.chains[k][0].vector())
        x = np.linalg.solve(a, b)
        for ii in range(2):
            data.chains.chains[i][2].vector()[:] -= (
                x[ii] * data.chains.chains[ii][1].vector()
            )
            data.chains.chains[i][1].vector()[:] -= (
                x[ii] * data.chains.chains[ii][0].vector()
            )

    for i in [2, 3]:
        rhs = -(H * data.chains.chains[i][2].vector()) + (
            M * data.chains.chains[i][1].vector()
        )
        data.output_data.null_space.orthogonalize(rhs)
        if singular:
            solver.solve(data.chains.chains[i][3].vector(), rhs)
        else:
            solve(E, data.chains.chains[i][3].vector(), rhs)
        data.output_data.null_space.orthogonalize(data.chains.chains[i][3].vector())

    # solve E d3 = M d1 - H d2
    for i in range(4):
        print("\nChain " + str(i) + ":")
        for k in range(len(data.chains.chains[i]) // 2, len(data.chains.chains[i])):
            print(
                "(d" + str(k) + ", d" + str(k) + ") = ",
                assemble(
                    inner(data.chains.chains_d[i][k], data.chains.chains_d[i][k]) * dx
                ),
            )
            if not singular:
                print(
                    "(l" + str(k) + ", l" + str(k) + ") = ",
                    assemble(
                        inner(data.chains.chains_l[i][k], data.chains.chains_l[i][k])
                        * dx
                    ),
                )
    for i in range(4):
        ll = len(data.chains.chains[i])
        for k in range(ll // 2, 0, -1):
            res = (
                E * data.chains.chains[i][ll - k].vector()
                + H * data.chains.chains[i][ll - 1 - k].vector()
            )
            if ll - 1 - k > 0:
                res -= M * data.chains.chains[i][ll - 2 - k].vector()
            res = as_backend_type(res).vec()
            print("residual chain", i, "order", ll, res.dot(res))
    print("")

    row1_col = []
    row2_col = []
    for i in range(6):
        row1_col.append(as_backend_type(data.chains.chains[0][0].vector().copy()).vec())
        row2_col.append(as_backend_type(data.chains.chains[0][0].vector().copy()).vec())

    M_p = as_backend_type(M).mat()
    C_p = as_backend_type(C).mat()
    E_p = as_backend_type(E).mat()
    S = PETSc.Mat().createDense([6, 6])
    S.setUp()

    B = PETSc.Mat().createDense([6, 6])
    B.setUp()

    G = PETSc.Mat().createDense([6, 6])
    G.setUp()

    g = PETSc.Vec().createMPI(6)

    Stiff = PETSc.Mat().createDense([6, 6])
    Stiff.setUp()

    col = -1
    for i in range(4):
        ll = len(data.chains.chains[i])
        for k in range(ll // 2, 0, -1):
            col = col + 1
            M_p.mult(
                as_backend_type(data.chains.chains[i][ll - 1 - k].vector()).vec(),
                row1_col[col],
            )
            C_p.multTransposeAdd(
                as_backend_type(data.chains.chains[i][ll - k].vector()).vec(),
                row1_col[col],
                row1_col[col],
            )
            C_p.mult(
                as_backend_type(data.chains.chains[i][ll - 1 - k].vector()).vec(),
                row2_col[col],
            )
            E_p.multAdd(
                as_backend_type(data.chains.chains[i][ll - k].vector()).vec(),
                row2_col[col],
                row2_col[col],
            )

    row = -1
    for i in range(4):
        ll = len(data.chains.chains[i])
        for k in range(ll // 2, 0, -1):
            row = row + 1
            for c in range(6):
                S.setValues(
                    row,
                    c,
                    as_backend_type(data.chains.chains[i][ll - 1 - k].vector())
                    .vec()
                    .dot(row1_col[c])
                    + as_backend_type(data.chains.chains[i][ll - k].vector())
                    .vec()
                    .dot(row2_col[c]),
                )
            B.setValues(
                row,
                range(6),
                as_backend_type(
                    data.output_data.L_res * data.chains.chains[i][ll - 1 - k].vector()
                    + data.output_data.R_res * data.chains.chains[i][ll - k].vector()
                ).vec(),
            )

    S.assemble()
    B.assemble()

    ksp = PETSc.KSP()
    ksp.create()
    ksp.setOperators(S)
    ksp.setType(ksp.Type.PREONLY)  # Just use the preconditioner without a Krylov method
    pc = ksp.getPC()  # Preconditioner
    pc.setType(pc.Type.LU)  # Use a direct solve

    for i in range(6):
        ksp.solve(B.getColumnVector(i), g)
        G.setValues(range(6), i, g)

    G.assemble()

    G.transposeMatMult(S, B)
    B.matMult(G, Stiff)

    data.output_data.B = B
    data.output_data.G = G
    data.output_data.Stiff = Stiff
    return Stiff
