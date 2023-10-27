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
#    Anba is distributed in the hope that it will be useful,
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

from anba4 import *
import mshr

parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 2

# Basic material parameters. 9 is needed for orthotropic materials.

E = 1.
nu = 0.33
#Assmble into material mechanical property Matrix.
matMechanicProp = [E, nu]
# Meshing domain.

thickness = 0.1
Square1 = mshr.Rectangle(Point(0., -1., 0.), Point(1., 1., 0.))
Square2 = mshr.Rectangle(Point(thickness, -1+thickness, 0), Point(2., 1.-thickness, 0))
C_shape = Square1 - Square2
mesh = mshr.generate_mesh(C_shape, 64)
rot_angle = 30. / 180. * np.pi
cr = cos(rot_angle)
sr = sin(rot_angle)
rot_tensor = np.array([[cr, -sr], [sr, cr]])
mesh.coordinates()[:] = (rot_tensor @ mesh.coordinates().T).T
mesh.coordinates()[:] += np.array([3, 1])

# plot(mesh)

# import matplotlib.pyplot as plt
# plt.show()


# CompiledSubDomain
materials = MeshFunction("size_t", mesh, mesh.topology().dim())
fiber_orientations = MeshFunction("double", mesh, mesh.topology().dim())
plane_orientations = MeshFunction("double", mesh, mesh.topology().dim())

materials.set_all(0)
fiber_orientations.set_all(0.0)
plane_orientations.set_all(90.0)

# Build material property library.
mat1 = material.IsotropicMaterial(matMechanicProp, 1.)

matLibrary = []
matLibrary.append(mat1)

anba = anbax(mesh, 2, matLibrary, materials, plane_orientations, fiber_orientations)
stiff = anba.compute()
print('Stiff:')
stiff.view()

mass = anba.inertia()
print('Mass:')
mass.view()


decoupled_stiff = DecoupleStiffness(stiff)
print('Decoupled Stiff:')
print(decoupled_stiff)
angle = PrincipalAxesRotationAngle(decoupled_stiff)
print('Rotation angle:', angle / np.pi * 180.)
