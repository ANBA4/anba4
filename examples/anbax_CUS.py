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
import time
import math
import numpy as np
from petsc4py import PETSc
import os
import matplotlib.pyplot as plt
from mshr import Polygon, generate_mesh

from anba4 import *

parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 2

# Basic material parameters. 9 is needed for orthotropic materials.
# psi2pa = 6894.7572931783
# e_xx = 1.42e6*psi2pa
# e_yy = 1.42e6*psi2pa
# e_zz = 20.59e6*psi2pa
# g_xy = 0.6960e6*psi2pa
# g_xz = 0.8700e6*psi2pa
# g_yz = 0.8700e6*psi2pa
# nu_xy = 0.34
# nu_zx = 0.3
# nu_zy = 0.3
e_xx = 9.8e9
e_yy = 9.8e9
e_zz = 1.42e11
g_xy = 4.8e9
g_xz = 6.0e9
g_yz = 6.0e9
nu_xy = 0.34
nu_zx = 0.3
nu_zy = 0.3
# msi2pa = 6894.7572931783
# e_xx = 0.75e6*msi2pa
# e_yy = 0.75e6*msi2pa
# e_zz = 30.0e6*msi2pa
# g_xy = 0.45e6*msi2pa
# g_xz = 0.3700e6*msi2pa
# g_yz = 0.37e6*msi2pa
# nu_xy = 0.25
# nu_zx = 0.25
# nu_zy = 0.25
# Assmble into material mechanical property Matrix.
matMechanicProp = np.zeros((3, 3))
matMechanicProp[0, 0] = e_xx
matMechanicProp[0, 1] = e_yy
matMechanicProp[0, 2] = e_zz
matMechanicProp[1, 0] = g_yz
matMechanicProp[1, 1] = g_xz
matMechanicProp[1, 2] = g_xy
matMechanicProp[2, 0] = nu_zy
matMechanicProp[2, 1] = nu_zx
matMechanicProp[2, 2] = nu_xy

# Meshing domain.
# sectionWidth = 3.0023e-2
# sectionHeight = 1.9215e-3
# nPly = 16 # t = 0.2452mm per ply.

in2m = 0.0254
# tp = 2.54e-4
t = 0.0300 * in2m / 6
b0 = 0.9530 * in2m
h0 = 0.5300 * in2m

b1 = b0 - 2.0 * t
h1 = h0 - 2.0 * t

b2 = b1 - 2.0 * t
h2 = h1 - 2.0 * t

b3 = b2 - 2.0 * t
h3 = h2 - 2.0 * t

b4 = b3 - 2.0 * t
h4 = h3 - 2.0 * t

b5 = b4 - 2.0 * t
h5 = h4 - 2.0 * t

b6 = b5 - 2.0 * t
h6 = h5 - 2.0 * t

# t   = 0.400*in2m
# b0  = 10.000*in2m
# h0  = 2.000*in2m
# b1  = b0 - 2.0*t
# h1  = h0 - 2.0*t
# tt = 0.03
top_plies_0 = Polygon(
    [
        Point(-0.5 * b1, 0.5 * h1),
        Point(0.5 * b1, 0.5 * h1),
        Point(0.5 * b0, 0.5 * h0),
        Point(-0.5 * b0, 0.5 * h0),
    ]
)
top_plies_1 = Polygon(
    [
        Point(-0.5 * b2, 0.5 * h2),
        Point(0.5 * b2, 0.5 * h2),
        Point(0.5 * b1, 0.5 * h1),
        Point(-0.5 * b1, 0.5 * h1),
    ]
)
top_plies_2 = Polygon(
    [
        Point(-0.5 * b3, 0.5 * h3),
        Point(0.5 * b3, 0.5 * h3),
        Point(0.5 * b2, 0.5 * h2),
        Point(-0.5 * b2, 0.5 * h2),
    ]
)
top_plies_3 = Polygon(
    [
        Point(-0.5 * b4, 0.5 * h4),
        Point(0.5 * b4, 0.5 * h4),
        Point(0.5 * b3, 0.5 * h3),
        Point(-0.5 * b3, 0.5 * h3),
    ]
)
top_plies_4 = Polygon(
    [
        Point(-0.5 * b5, 0.5 * h5),
        Point(0.5 * b5, 0.5 * h5),
        Point(0.5 * b4, 0.5 * h4),
        Point(-0.5 * b4, 0.5 * h4),
    ]
)
top_plies_5 = Polygon(
    [
        Point(-0.5 * b6, 0.5 * h6),
        Point(0.5 * b6, 0.5 * h6),
        Point(0.5 * b5, 0.5 * h5),
        Point(-0.5 * b5, 0.5 * h5),
    ]
)

down_plies_0 = Polygon(
    [
        Point(-0.5 * b0, -0.5 * h0),
        Point(0.5 * b0, -0.5 * h0),
        Point(0.5 * b1, -0.5 * h1),
        Point(-0.5 * b1, -0.5 * h1),
    ]
)
down_plies_1 = Polygon(
    [
        Point(-0.5 * b1, -0.5 * h1),
        Point(0.5 * b1, -0.5 * h1),
        Point(0.5 * b2, -0.5 * h2),
        Point(-0.5 * b2, -0.5 * h2),
    ]
)
down_plies_2 = Polygon(
    [
        Point(-0.5 * b2, -0.5 * h2),
        Point(0.5 * b2, -0.5 * h2),
        Point(0.5 * b3, -0.5 * h3),
        Point(-0.5 * b3, -0.5 * h3),
    ]
)
down_plies_3 = Polygon(
    [
        Point(-0.5 * b3, -0.5 * h3),
        Point(0.5 * b3, -0.5 * h3),
        Point(0.5 * b4, -0.5 * h4),
        Point(-0.5 * b4, -0.5 * h4),
    ]
)
down_plies_4 = Polygon(
    [
        Point(-0.5 * b4, -0.5 * h4),
        Point(0.5 * b4, -0.5 * h4),
        Point(0.5 * b5, -0.5 * h5),
        Point(-0.5 * b5, -0.5 * h5),
    ]
)
down_plies_5 = Polygon(
    [
        Point(-0.5 * b5, -0.5 * h5),
        Point(0.5 * b5, -0.5 * h5),
        Point(0.5 * b6, -0.5 * h6),
        Point(-0.5 * b6, -0.5 * h6),
    ]
)

left_plies_0 = Polygon(
    [
        Point(-0.5 * b0, -0.5 * h0),
        Point(-0.5 * b1, -0.5 * h1),
        Point(-0.5 * b1, 0.5 * h1),
        Point(-0.5 * b0, 0.5 * h0),
    ]
)
left_plies_1 = Polygon(
    [
        Point(-0.5 * b1, -0.5 * h1),
        Point(-0.5 * b2, -0.5 * h2),
        Point(-0.5 * b2, 0.5 * h2),
        Point(-0.5 * b1, 0.5 * h1),
    ]
)
left_plies_2 = Polygon(
    [
        Point(-0.5 * b2, -0.5 * h2),
        Point(-0.5 * b3, -0.5 * h3),
        Point(-0.5 * b3, 0.5 * h3),
        Point(-0.5 * b2, 0.5 * h2),
    ]
)
left_plies_3 = Polygon(
    [
        Point(-0.5 * b3, -0.5 * h3),
        Point(-0.5 * b4, -0.5 * h4),
        Point(-0.5 * b4, 0.5 * h4),
        Point(-0.5 * b3, 0.5 * h3),
    ]
)
left_plies_4 = Polygon(
    [
        Point(-0.5 * b4, -0.5 * h4),
        Point(-0.5 * b5, -0.5 * h5),
        Point(-0.5 * b5, 0.5 * h5),
        Point(-0.5 * b4, 0.5 * h4),
    ]
)
left_plies_5 = Polygon(
    [
        Point(-0.5 * b5, -0.5 * h5),
        Point(-0.5 * b6, -0.5 * h6),
        Point(-0.5 * b6, 0.5 * h6),
        Point(-0.5 * b5, 0.5 * h5),
    ]
)

right_plies_0 = Polygon(
    [
        Point(0.5 * b1, -0.5 * h1),
        Point(0.5 * b0, -0.5 * h0),
        Point(0.5 * b0, 0.5 * h0),
        Point(0.5 * b1, 0.5 * h1),
    ]
)
right_plies_1 = Polygon(
    [
        Point(0.5 * b2, -0.5 * h2),
        Point(0.5 * b1, -0.5 * h1),
        Point(0.5 * b1, 0.5 * h1),
        Point(0.5 * b2, 0.5 * h2),
    ]
)
right_plies_2 = Polygon(
    [
        Point(0.5 * b3, -0.5 * h3),
        Point(0.5 * b2, -0.5 * h2),
        Point(0.5 * b2, 0.5 * h2),
        Point(0.5 * b3, 0.5 * h3),
    ]
)
right_plies_3 = Polygon(
    [
        Point(0.5 * b4, -0.5 * h4),
        Point(0.5 * b3, -0.5 * h3),
        Point(0.5 * b3, 0.5 * h3),
        Point(0.5 * b4, 0.5 * h4),
    ]
)
right_plies_4 = Polygon(
    [
        Point(0.5 * b5, -0.5 * h5),
        Point(0.5 * b4, -0.5 * h4),
        Point(0.5 * b4, 0.5 * h4),
        Point(0.5 * b5, 0.5 * h4),
    ]
)
right_plies_5 = Polygon(
    [
        Point(0.5 * b6, -0.5 * h6),
        Point(0.5 * b5, -0.5 * h5),
        Point(0.5 * b5, 0.5 * h5),
        Point(0.5 * b6, 0.5 * h6),
    ]
)

domain_0 = (
    top_plies_0
    + down_plies_0
    + left_plies_0
    + right_plies_0
    + top_plies_1
    + down_plies_1
    + left_plies_1
    + right_plies_1
    + top_plies_2
    + down_plies_2
    + left_plies_2
    + right_plies_2
    + top_plies_3
    + down_plies_3
    + left_plies_3
    + right_plies_3
    + top_plies_4
    + down_plies_4
    + left_plies_4
    + right_plies_4
    + top_plies_5
    + down_plies_5
    + left_plies_5
    + right_plies_5
)


domain_0.set_subdomain(1, top_plies_0)
domain_0.set_subdomain(2, top_plies_1)
domain_0.set_subdomain(3, top_plies_2)
domain_0.set_subdomain(4, top_plies_3)
domain_0.set_subdomain(5, top_plies_4)
domain_0.set_subdomain(6, top_plies_5)

domain_0.set_subdomain(7, down_plies_0)
domain_0.set_subdomain(8, down_plies_1)
domain_0.set_subdomain(9, down_plies_2)
domain_0.set_subdomain(10, down_plies_3)
domain_0.set_subdomain(11, down_plies_4)
domain_0.set_subdomain(12, down_plies_5)

domain_0.set_subdomain(13, left_plies_0)
domain_0.set_subdomain(14, left_plies_1)
domain_0.set_subdomain(15, left_plies_2)
domain_0.set_subdomain(16, left_plies_3)
domain_0.set_subdomain(17, left_plies_4)
domain_0.set_subdomain(18, left_plies_5)

domain_0.set_subdomain(19, right_plies_0)
domain_0.set_subdomain(20, right_plies_1)
domain_0.set_subdomain(21, right_plies_2)
domain_0.set_subdomain(22, right_plies_3)
domain_0.set_subdomain(23, right_plies_4)
domain_0.set_subdomain(24, right_plies_5)

mesh = generate_mesh(domain_0, 20000)
mf = MeshFunction("size_t", mesh, 2, mesh.domains())

materials = MeshFunction("size_t", mesh, 2, mesh.domains())
# isPiezoActive = MeshFunction("bool", mesh, 2, mesh.domains())
# isMagneticActive = MeshFunction("bool", mesh, 2, mesh.domains())
fiber_orientations = MeshFunction("double", mesh, 2, mesh.domains())
plane_orientations = MeshFunction("double", mesh, 2, mesh.domains())

materials.set_all(0)
fiber_orientations.set_all(15.0)
plane_orientations.set_all(0.0)

for ele in cells(mesh):
    i = ele.index()
    flag = mf[i]
    if flag in range(1, 7):
        # Base
        materials.set_value(i, 0)
        plane_orientations.set_value(i, 0.0)
        fiber_orientations.set_value(i, 15.0)
    elif flag in range(7, 13):
        # Left patch
        materials.set_value(i, 0)
        plane_orientations.set_value(i, 180.0)
        fiber_orientations.set_value(i, 15.0)
    elif flag in range(13, 19):
        # Left patch
        materials.set_value(i, 0)
        plane_orientations.set_value(i, 90.0)
        fiber_orientations.set_value(i, 15.0)
    elif flag in range(19, 25):
        # Left patch
        materials.set_value(i, 0)
        plane_orientations.set_value(i, 270.0)
        fiber_orientations.set_value(i, 15.0)
    else:
        break


# #mesh = RectangleMesh.create([Point(0., 0.), Point(sectionWidth, sectionHeight)], [30, 32], CellType.Type.quadrilateral)
# mesh = RectangleMesh(Point(0., 0.), Point(sectionWidth, sectionHeight), 30, 32, 'crossed')
# ALE.move(mesh, Constant([-sectionWidth/2.0, -sectionHeight/2.0]))
# mesh_points=mesh.coordinates()
# #print(mesh_points)
#
# # CompiledSubDomain
# materials = MeshFunction("size_t", mesh, mesh.topology().dim())
# fiber_orientations = MeshFunction("double", mesh, mesh.topology().dim())
# plane_orientations = MeshFunction("double", mesh, mesh.topology().dim())
# #isActive = MeshFunction("bool", mesh, mesh.topology().dim())
# tol = 1e-14

"""subdomain_0 = CompiledSubDomain(["x[1] >= -5.0*thickness + tol && x[1] <= -4.0*thickness + tol",\
                                 "x[1] >= -3.0*thickness + tol && x[1] <= -2.0*thickness + tol",\
                                 "x[1] >= -1.0*thickness + tol && x[1] <= 0.0*thickness + tol"\
                                 "x[1] >= 0.0*thickness + tol && x[1] <= 1.0*thickness + tol",\
                                 "x[1] >= 2.0*thickness + tol && x[1] <= 3.0*thickness + tol",\
                                 "x[1] >= 4.0*thickness + tol && x[1] <= 5.0*thickness + tol"], \
                                 thickness = sectionHeight / nPly, tol=tol)
subdomain_1 = CompiledSubDomain(["x[1] >= -6.0*thickness + tol && x[1] <= -5.0*thickness + tol",\
                                 "x[1] >= -4.0*thickness + tol && x[1] <= -3.0*thickness + tol",\
                                 "x[1] >= -2.0*thickness + tol && x[1] <= -1.0*thickness + tol",\
                                 "x[1] >= 1.0*thickness + tol && x[1] <= 2.0*thickness + tol",\
                                 "x[1] >= 3.0*thickness + tol && x[1] <= 4.0*thickness + tol",\
                                 "x[1] >= 5.0*thickness + tol && x[1] <= 6.0*thickness + tol"],\
                                  thickness = sectionHeight / nPly, tol=tol)
                                  """
# subdomain_0_p20 = CompiledSubDomain("x[1] >= -8.0*thickness - tol && x[1] <= -7.0*thickness + tol",thickness = sectionHeight / nPly, tol=tol)
# subdomain_1_m70 = CompiledSubDomain("x[1] >= -7.0*thickness - tol && x[1] <= -6.0*thickness + tol",thickness = sectionHeight / nPly, tol=tol)
# subdomain_2_m70 = CompiledSubDomain("x[1] >= -6.0*thickness - tol && x[1] <= -5.0*thickness + tol",thickness = sectionHeight / nPly, tol=tol)
# subdomain_3_p20 = CompiledSubDomain("x[1] >= -5.0*thickness - tol && x[1] <= -4.0*thickness + tol",thickness = sectionHeight / nPly, tol=tol)
#
# subdomain_4_p20 = CompiledSubDomain("x[1] >= -4.0*thickness - tol && x[1] <= -3.0*thickness + tol",thickness = sectionHeight / nPly, tol=tol)
# subdomain_5_m70 = CompiledSubDomain("x[1] >= -3.0*thickness - tol && x[1] <= -2.0*thickness + tol",thickness = sectionHeight / nPly, tol=tol)
# subdomain_6_m70 = CompiledSubDomain("x[1] >= -2.0*thickness - tol && x[1] <= -1.0*thickness + tol",thickness = sectionHeight / nPly, tol=tol)
# subdomain_7_p20 = CompiledSubDomain("x[1] >= -1.0*thickness - tol && x[1] <= -0.0*thickness + tol",thickness = sectionHeight / nPly, tol=tol)
#
# subdomain_8_m20 = CompiledSubDomain("x[1] >= 0.0*thickness - tol && x[1] <= 1.0*thickness + tol",thickness = sectionHeight / nPly, tol=tol)
# subdomain_9_p70 = CompiledSubDomain("x[1] >= 1.0*thickness - tol && x[1] <= 2.0*thickness + tol",thickness = sectionHeight / nPly, tol=tol)
# subdomain_10_p70 = CompiledSubDomain("x[1] >= 2.0*thickness - tol && x[1] <= 3.0*thickness + tol",thickness = sectionHeight / nPly, tol=tol)
# subdomain_11_m20 = CompiledSubDomain("x[1] >= 3.0*thickness - tol && x[1] <= 4.0*thickness + tol",thickness = sectionHeight / nPly, tol=tol)
#
# subdomain_12_m20 = CompiledSubDomain("x[1] >= 4.0*thickness - tol && x[1] <= 5.0*thickness + tol",thickness = sectionHeight / nPly, tol=tol)
# subdomain_13_p70 = CompiledSubDomain("x[1] >= 5.0*thickness - tol && x[1] <= 6.0*thickness + tol",thickness = sectionHeight / nPly, tol=tol)
# subdomain_14_p70 = CompiledSubDomain("x[1] >= 6.0*thickness - tol && x[1] <= 7.0*thickness + tol",thickness = sectionHeight / nPly, tol=tol)
# subdomain_15_m20 = CompiledSubDomain("x[1] >= 7.0*thickness - tol && x[1] <= 8.0*thickness + tol",thickness = sectionHeight / nPly, tol=tol)
#
# # Rotate mesh.
# rotation_angle = 23.0
# materials.set_all(0)
# fiber_orientations.set_all(0.0)
# plane_orientations.set_all(rotation_angle)
#
# subdomain_0_p20.mark(materials, 1)
#
# subdomain_0_p20.mark(fiber_orientations, 20.0)
# subdomain_1_m70.mark(fiber_orientations, -70.0)
# subdomain_2_m70.mark(fiber_orientations, -70.0)
# subdomain_3_p20.mark(fiber_orientations, 20.0)
#
# subdomain_4_p20.mark(fiber_orientations, 20.0)
# subdomain_5_m70.mark(fiber_orientations, -70.0)
# subdomain_6_m70.mark(fiber_orientations, -70.0)
# subdomain_7_p20.mark(fiber_orientations, 20.0)
#
# subdomain_8_m20.mark(fiber_orientations, -20.0)
# subdomain_9_p70.mark(fiber_orientations, 70.0)
# subdomain_10_p70.mark(fiber_orientations, 70.0)
# subdomain_11_m20.mark(fiber_orientations, -20.0)
#
# subdomain_12_m20.mark(fiber_orientations, -20.0)
# subdomain_13_p70.mark(fiber_orientations, 70.0)
# subdomain_14_p70.mark(fiber_orientations, 70.0)
# subdomain_15_m20.mark(fiber_orientations, -20.0)
#
# # rotate mesh.
# rotate = Expression(("x[0] * (cos(rotation_angle)-1.0) - x[1] * sin(rotation_angle)",
#     "x[0] * sin(rotation_angle) + x[1] * (cos(rotation_angle)-1.0)"), rotation_angle = rotation_angle * np.pi / 180.0,
#     degree = 1)
#
# ALE.move(mesh, rotate)

# Build material property library.
mat1 = material.OrthotropicMaterial(matMechanicProp, 7800)

# matMechanicProp1 = matMechanicProp
# matMechanicProp1[0] = matMechanicProp1[0] / 100.
# matMechanicProp1[1] = matMechanicProp1[1] / 100.

# mat2 = material.IsotropicMaterial(matMechanicProp1)
matLibrary = []
matLibrary.append(mat1)
# matLibrary.append(mat2)


anba = anbax(
    mesh,
    2,
    matLibrary,
    materials,
    plane_orientations,
    fiber_orientations,
    1.0e0,
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

# anba.stress_field([1., 0., 0.,], [0., 0., 0.], "local", "paraview")
# stress_result_file.write(anba.STRESS, t = 0.)
