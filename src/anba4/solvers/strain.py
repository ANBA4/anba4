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

from dolfin import split, Function
from petsc4py import PETSc

from ..utils import epsilon, rotated_epsilon, local_project
from ..core.voight_notation import (
    strainTensorToStrainVector,
    strainTensorToParaviewStrainVector,
)
from ..data.data_model import AnbaData


def strain_field(
    data: AnbaData, force, moment, reference="local", voigt_convention="anba"
):
    singular = data.input_data.singular
    if reference == "local":

        def strain_comp(u, up):
            return rotated_epsilon(data, u, up)
    elif reference == "global":

        def strain_comp(u, up):
            return epsilon(u, up)
    else:
        raise ValueError(
            'reference argument should be equal to either to"local" or to "global", got "'
            + reference
            + '" instead'
        )
    if voigt_convention == "anba":
        vector_conversion = strainTensorToStrainVector
    elif voigt_convention == "paraview":
        vector_conversion = strainTensorToParaviewStrainVector
    else:
        raise ValueError(
            'voigt_convention argument should be equal to either to"anba" or to "paraview", got "'
            + voigt_convention
            + '" instead'
        )

    eigensol_magnitudes = PETSc.Vec().createMPI(6)

    AzInt = PETSc.Vec().createMPI(6)

    AzInt.setValues(range(3), force)
    AzInt.setValues(range(3, 6), moment)
    AzInt.assemblyBegin()
    AzInt.assemblyEnd()

    ksp = PETSc.KSP()
    ksp.create()
    ksp.setOperators(data.output_data.B)
    ksp.setType(ksp.Type.PREONLY)  # Just use the preconditioner without a Krylov method
    pc = ksp.getPC()  # Preconditioner
    pc.setType(pc.Type.LU)  # Use a direct solve

    ksp.solve(AzInt, eigensol_magnitudes)

    if singular:
        UL = data.fe_functions.U
        ULP = data.fe_functions.UP
    else:
        UL = Function(data.fe_functions.UF3R4)
        ULP = Function(data.fe_functions.UF3R4)
    UL.vector()[:] = 0.0
    ULP.vector()[:] = 0.0
    row = -1
    for i in range(4):
        ll = len(data.chains.chains[i])
        for k in range(ll // 2, 0, -1):
            row = row + 1
            UL.vector()[:] += (
                data.chains.chains[i][ll - k].vector() * eigensol_magnitudes[row]
            )
            ULP.vector()[:] += (
                data.chains.chains[i][ll - 1 - k].vector() * eigensol_magnitudes[row]
            )
    (U, L) = split(UL) if not singular else (UL, None)
    (UP, LP) = split(ULP) if not singular else (ULP, None)
    strain = local_project(
        vector_conversion(strain_comp(U, UP)), data.fe_functions.STRESS_FS
    )
    strain.rename("strain tensor", "")
    return strain
