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

# from dolfin import *
import dolfin
from ..data.data_model import AnbaData


def initialize_fe_functions(data: AnbaData) -> AnbaData:
    mesh = data.input_data.mesh
    degree = data.input_data.degree
    singular = data.input_data.singular

    # Define function on space.
    UF3_ELEMENT = dolfin.VectorElement("CG", mesh.ufl_cell(), degree, 3)
    data.fe_functions.UF3 = dolfin.FunctionSpace(mesh, UF3_ELEMENT)

    # Lagrange multipliers needed to compute the stress resultants and moment resultants
    R3_ELEMENT = dolfin.VectorElement("R", mesh.ufl_cell(), 0, 3)
    data.fe_functions.R3 = dolfin.FunctionSpace(mesh, R3_ELEMENT)
    sp = dolfin.parameters["reorder_dofs_serial"]
    dolfin.parameters["reorder_dofs_serial"] = False
    R3R3_ELEMENT = dolfin.MixedElement(R3_ELEMENT, R3_ELEMENT)
    data.fe_functions.R3R3 = dolfin.FunctionSpace(mesh, R3R3_ELEMENT)
    dolfin.parameters["reorder_dofs_serial"] = sp
    (data.fe_functions.RV3F, data.fe_functions.RV3M) = dolfin.TestFunctions(
        data.fe_functions.R3R3
    )
    (data.fe_functions.RT3F, data.fe_functions.RT3M) = dolfin.TrialFunctions(
        data.fe_functions.R3R3
    )

    STRESS_ELEMENT = dolfin.VectorElement("DG", mesh.ufl_cell(), 0, 6)
    data.fe_functions.STRESS_FS = dolfin.FunctionSpace(mesh, STRESS_ELEMENT)

    if not singular:
        # Lagrange multipliers needed to impose the BCs
        R4_ELEMENT = dolfin.VectorElement("R", mesh.ufl_cell(), 0, 4)
        data.fe_functions.R4 = dolfin.FunctionSpace(mesh, R4_ELEMENT)
        sp = dolfin.parameters["reorder_dofs_serial"]
        dolfin.parameters["reorder_dofs_serial"] = False
        UF3R4_ELEMENT = dolfin.MixedElement(UF3_ELEMENT, R4_ELEMENT)
        data.fe_functions.UF3R4 = dolfin.FunctionSpace(mesh, UF3R4_ELEMENT)
        dolfin.parameters["reorder_dofs_serial"] = sp

        data.fe_functions.UL = dolfin.Function(data.fe_functions.UF3R4)
        (data.fe_functions.U, data.fe_functions.L) = dolfin.split(data.fe_functions.UL)
        data.fe_functions.ULP = dolfin.Function(data.fe_functions.UF3R4)
        (data.fe_functions.UP, data.fe_functions.LP) = dolfin.split(
            data.fe_functions.ULP
        )
        data.fe_functions.ULV = dolfin.TestFunction(data.fe_functions.UF3R4)
        (data.fe_functions.UV, data.fe_functions.LV) = dolfin.TestFunctions(
            data.fe_functions.UF3R4
        )
        data.fe_functions.ULT = dolfin.TrialFunction(data.fe_functions.UF3R4)
        (data.fe_functions.UT, data.fe_functions.LT) = dolfin.TrialFunctions(
            data.fe_functions.UF3R4
        )
    else:
        data.fe_functions.U = dolfin.Function(data.fe_functions.UF3)
        data.fe_functions.UP = dolfin.Function(data.fe_functions.UF3)
        data.fe_functions.UV = dolfin.TestFunction(data.fe_functions.UF3)
        data.fe_functions.UT = dolfin.TrialFunction(data.fe_functions.UF3)
    return data
