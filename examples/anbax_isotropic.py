#
# Copyright (C) 2018 Marco Morandini
#
#----------------------------------------------------------------------
#
#    This file is part of Anba.
#
#    Anba is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Hanba is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Anba.  If not, see <https://www.gnu.org/licenses/>.
#
#----------------------------------------------------------------------
#

from dolfin import *
# from dolfin import compile_extension_module
import numpy as np
from petsc4py import PETSc

from voight_notation import stressVectorToStressTensor, stressTensorToStressVector, strainVectorToStrainTensor, strainTensorToStrainVector
from material import material
from anbax import anbax

parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 2

# Basic material parameters. 9 is needed for orthotropic materials.

E = 1.
nu = 0.33
#Assmble into material mechanical property Matrix.
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
plane_orientations.set_all(0.0)

# Build material property library.
mat1 = material.IsotropicMaterial(matMechanicProp)

matLibrary = []
matLibrary.append(mat1)


anba = anbax(mesh, 2, matLibrary, materials, plane_orientations, fiber_orientations)
stiff = anba.compute()
stiff.view()