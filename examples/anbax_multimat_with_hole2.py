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
import time
import math
import numpy as np
from petsc4py import PETSc
import os
import matplotlib.pyplot as plt

from anba4 import *
import mshr

parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 2

# Basic material parameters. 9 is needed for orthotropic materials.
matMechanicProp1 = [ 80000, 0.3]
matMechanicProp2 = [ 80000 * 0.5, 0.3]
matMechanicProp3 = [ 80000 * 0.001, 0.3]

# Meshing domain.
sectionWidth = 20
sectionHeight = 20


#mesh = RectangleMesh.create([Point(0., 0.), Point(sectionWidth, sectionHeight)], [30, 32], CellType.Type.quadrilateral)
Square = mshr.Rectangle(Point(-10., -10.), Point(10., 10.))
Rectangle = mshr.Rectangle(Point(-2., -6.), Point(2., 6.))
Domain = Square - Rectangle
mesh = mshr.generate_mesh(Domain, 64)

# CompiledSubDomain
materials = MeshFunction("size_t", mesh, mesh.topology().dim())
fiber_orientations = MeshFunction("double", mesh, mesh.topology().dim())
plane_orientations = MeshFunction("double", mesh, mesh.topology().dim())
#isActive = MeshFunction("bool", mesh, mesh.topology().dim())
tol = 1e-14

lower_portion = CompiledSubDomain("x[1] <= 0 + tol", tol=tol)

# Rotate mesh.
rotation_angle = 0.
materials.set_all(0)
fiber_orientations.set_all(0.0)
plane_orientations.set_all(rotation_angle)

lower_portion.mark(materials, 1)
plot(materials, "Subdomains")
import matplotlib.pyplot as plt
#plt.show()

# rotate mesh.
mat1 = material.IsotropicMaterial(matMechanicProp1)
mat2 = material.IsotropicMaterial(matMechanicProp2)
mat3 = material.IsotropicMaterial(matMechanicProp3)
matLibrary = []
matLibrary.append(mat1)
matLibrary.append(mat2)
matLibrary.append(mat3)


anba = anbax(mesh, 1, matLibrary, materials, plane_orientations, fiber_orientations, 1)
stiff = anba.compute()
stiff.view()

JordanChains = XDMFFile('jordan_chains_with_real_hole2.xdmf')
JordanChains.parameters['functions_share_mesh'] = True
JordanChains.parameters['rewrite_function_mesh'] = False
JordanChains.parameters["flush_output"] = True
for i in range(len(anba.chains_d)):
    for j in range(len(anba.chains_d[i])):
        #print('chain_'+str(i)+'_'+str(j))
        chain = Function(anba.UF3, name='chain_'+str(i)+'_'+str(j))
        chain.vector()[:] = project(anba.chains_d[i][j], anba.UF3).vector()
        JordanChains.write(chain, t = 0.)
    

