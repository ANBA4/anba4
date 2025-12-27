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

from dolfin import dot, cross, dx, assemble, as_backend_type
from petsc4py import PETSc

from typing import Any

from ..utils import pos3d
from ..data.data_model import AnbaData


def compute_inertia(data: AnbaData) -> Any:
    Mf = (
        dot(data.fe_functions.RV3F, data.fe_functions.RT3F)
        * data.material_data.density[0]
        * dx
    )
    Mf -= (
        dot(
            data.fe_functions.RV3F,
            cross(pos3d(data.fe_functions.POS), data.fe_functions.RT3M),
        )
        * data.material_data.density[0]
        * dx
    )
    Mf -= (
        dot(
            cross(pos3d(data.fe_functions.POS), data.fe_functions.RV3M),
            data.fe_functions.RT3F,
        )
        * data.material_data.density[0]
        * dx
    )
    Mf += (
        dot(
            cross(pos3d(data.fe_functions.POS), data.fe_functions.RV3M),
            cross(pos3d(data.fe_functions.POS), data.fe_functions.RT3M),
        )
        * data.material_data.density[0]
        * dx
    )
    MM = assemble(Mf)
    M = as_backend_type(MM).mat()
    Mass = PETSc.Mat()
    M.convert("dense", Mass)
    return Mass
