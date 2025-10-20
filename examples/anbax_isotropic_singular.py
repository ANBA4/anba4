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

from dolfin import *

# from dolfin import compile_extension_module
import numpy as np
from petsc4py import PETSc

from anba4 import *

parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 2

# Basic material parameters. 9 is needed for orthotropic materials.

E = 1.0
nu = 0.33
# Assmble into material mechanical property Matrix.
matMechanicProp = [E, nu]
# Meshing domain.

mesh = UnitSquareMesh(10, 10)
ALE.move(mesh, Constant([-0.5, -0.5]))

# CompiledSubDomain
materials = MeshFunction("size_t", mesh, mesh.topology().dim())
fiber_orientations = MeshFunction("double", mesh, mesh.topology().dim())
plane_orientations = MeshFunction("double", mesh, mesh.topology().dim())

materials.set_all(0)
fiber_orientations.set_all(0.0)
plane_orientations.set_all(90.0)

# Build material property library.
mat1 = material.IsotropicMaterial(matMechanicProp, 1.0)

matLibrary = []
matLibrary.append(mat1)


anba = anbax(
    mesh,
    2,
    matLibrary,
    materials,
    plane_orientations,
    fiber_orientations,
    singular=True,
)
stiff = anba.compute()
stiff.view()

mass = anba.inertia()
mass.view()

stress_result_file = XDMFFile("Stress.xdmf")
stress_result_file.parameters["functions_share_mesh"] = True
stress_result_file.parameters["rewrite_function_mesh"] = False
stress_result_file.parameters["flush_output"] = True

# anba.stress_field([1., 0., 0.,], [0., 0., 0.], "global", "paraview")
# stress_result_file.write(anba.STRESS, t = 0.)
