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
import json
from anba4.io import load_model_from_json, model_to_dolfin, Output, Matrix

parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 2

# Basic material parameters. 9 is needed for orthotropic materials.

e_xx = 9.8e9
e_yy = 9.8e9
e_zz = 1.42e11
g_xy = 4.8e9
g_xz = 6.0e9
g_yz = 6.0e9
nu_xy = 0.34
nu_zx = 0.3
nu_zy = 0.3
#Assmble into material mechanical property Matrix.
matMechanicProp = np.zeros((3,3))
matMechanicProp[0,0] = e_xx
matMechanicProp[0,1] = e_yy
matMechanicProp[0,2] = e_zz
matMechanicProp[1,0] = g_yz
matMechanicProp[1,1] = g_xz
matMechanicProp[1,2] = g_xy
matMechanicProp[2,0] = nu_zy
matMechanicProp[2,1] = nu_zx
matMechanicProp[2,2] = nu_xy

matMechanicProp1 = [ 9.8e9 / 100., 0.3]
# Meshing domain.
sectionWidth = 3.0023e-2
sectionHeight = 1.9215e-3
nPly = 16 # t = 0.2452mm per ply.
#mesh = RectangleMesh.create([Point(0., 0.), Point(sectionWidth, sectionHeight)], [30, 32], CellType.Type.quadrilateral)
mesh = RectangleMesh(Point(0., 0.), Point(sectionWidth, sectionHeight), 30, 32, 'crossed')
ALE.move(mesh, Constant([-sectionWidth/2.0, -sectionHeight/2.0]))
mesh_points=mesh.coordinates()
#print(mesh_points)

# CompiledSubDomain
materials = MeshFunction("size_t", mesh, mesh.topology().dim())
fiber_orientations = MeshFunction("double", mesh, mesh.topology().dim())
plane_orientations = MeshFunction("double", mesh, mesh.topology().dim())
#isActive = MeshFunction("bool", mesh, mesh.topology().dim())
tol = 1e-14

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
subdomain_0_p20 = CompiledSubDomain("x[1] >= -8.0*thickness - tol && x[1] <= -7.0*thickness + tol",thickness = sectionHeight / nPly, tol=tol)
subdomain_1_m70 = CompiledSubDomain("x[1] >= -7.0*thickness - tol && x[1] <= -6.0*thickness + tol",thickness = sectionHeight / nPly, tol=tol)
subdomain_2_m70 = CompiledSubDomain("x[1] >= -6.0*thickness - tol && x[1] <= -5.0*thickness + tol",thickness = sectionHeight / nPly, tol=tol)
subdomain_3_p20 = CompiledSubDomain("x[1] >= -5.0*thickness - tol && x[1] <= -4.0*thickness + tol",thickness = sectionHeight / nPly, tol=tol)

subdomain_4_p20 = CompiledSubDomain("x[1] >= -4.0*thickness - tol && x[1] <= -3.0*thickness + tol",thickness = sectionHeight / nPly, tol=tol)
subdomain_5_m70 = CompiledSubDomain("x[1] >= -3.0*thickness - tol && x[1] <= -2.0*thickness + tol",thickness = sectionHeight / nPly, tol=tol)
subdomain_6_m70 = CompiledSubDomain("x[1] >= -2.0*thickness - tol && x[1] <= -1.0*thickness + tol",thickness = sectionHeight / nPly, tol=tol)
subdomain_7_p20 = CompiledSubDomain("x[1] >= -1.0*thickness - tol && x[1] <= -0.0*thickness + tol",thickness = sectionHeight / nPly, tol=tol)

subdomain_8_m20 = CompiledSubDomain("x[1] >= 0.0*thickness - tol && x[1] <= 1.0*thickness + tol",thickness = sectionHeight / nPly, tol=tol)
subdomain_9_p70 = CompiledSubDomain("x[1] >= 1.0*thickness - tol && x[1] <= 2.0*thickness + tol",thickness = sectionHeight / nPly, tol=tol)
subdomain_10_p70 = CompiledSubDomain("x[1] >= 2.0*thickness - tol && x[1] <= 3.0*thickness + tol",thickness = sectionHeight / nPly, tol=tol)
subdomain_11_m20 = CompiledSubDomain("x[1] >= 3.0*thickness - tol && x[1] <= 4.0*thickness + tol",thickness = sectionHeight / nPly, tol=tol)

subdomain_12_m20 = CompiledSubDomain("x[1] >= 4.0*thickness - tol && x[1] <= 5.0*thickness + tol",thickness = sectionHeight / nPly, tol=tol)
subdomain_13_p70 = CompiledSubDomain("x[1] >= 5.0*thickness - tol && x[1] <= 6.0*thickness + tol",thickness = sectionHeight / nPly, tol=tol)
subdomain_14_p70 = CompiledSubDomain("x[1] >= 6.0*thickness - tol && x[1] <= 7.0*thickness + tol",thickness = sectionHeight / nPly, tol=tol)
subdomain_15_m20 = CompiledSubDomain("x[1] >= 7.0*thickness - tol && x[1] <= 8.0*thickness + tol",thickness = sectionHeight / nPly, tol=tol)

# Rotate mesh.
rotation_angle = 23.0
materials.set_all(0)
fiber_orientations.set_all(0.0)
plane_orientations.set_all(rotation_angle)

subdomain_0_p20.mark(materials, 1)

subdomain_0_p20.mark(fiber_orientations, 20.0)
subdomain_1_m70.mark(fiber_orientations, -70.0)
subdomain_2_m70.mark(fiber_orientations, -70.0)
subdomain_3_p20.mark(fiber_orientations, 20.0)

subdomain_4_p20.mark(fiber_orientations, 20.0)
subdomain_5_m70.mark(fiber_orientations, -70.0)
subdomain_6_m70.mark(fiber_orientations, -70.0)
subdomain_7_p20.mark(fiber_orientations, 20.0)

subdomain_8_m20.mark(fiber_orientations, -20.0)
subdomain_9_p70.mark(fiber_orientations, 70.0)
subdomain_10_p70.mark(fiber_orientations, 70.0)
subdomain_11_m20.mark(fiber_orientations, -20.0)

subdomain_12_m20.mark(fiber_orientations, -20.0)
subdomain_13_p70.mark(fiber_orientations, 70.0)
subdomain_14_p70.mark(fiber_orientations, 70.0)
subdomain_15_m20.mark(fiber_orientations, -20.0)

# rotate mesh.
rotate = Expression(("x[0] * (cos(rotation_angle)-1.0) - x[1] * sin(rotation_angle)",
    "x[0] * sin(rotation_angle) + x[1] * (cos(rotation_angle)-1.0)"), rotation_angle = rotation_angle * np.pi / 180.0,
    degree = 1)

ALE.move(mesh, rotate)

# Build material property library.
mat1 = material.OrthotropicMaterial(matMechanicProp)

#matMechanicProp1 = matMechanicProp
#matMechanicProp1[0] = matMechanicProp1[0] / 100.
#matMechanicProp1[1] = matMechanicProp1[1] / 100.

mat2 = material.IsotropicMaterial(matMechanicProp1)
matLibrary = []
matLibrary.append(mat1)
matLibrary.append(mat2)

# Now, prepare data for IO model
points = mesh.coordinates().tolist()
cells = mesh.cells().tolist()

matlibrary_data = [
    {
        "type": "orthotropic",
        "e_xx": matMechanicProp[0,0],
        "e_yy": matMechanicProp[0,1],
        "e_zz": matMechanicProp[0,2],
        "g_xy": matMechanicProp[1,2],
        "g_xz": matMechanicProp[1,1],
        "g_yz": matMechanicProp[1,0],
        "nu_xy": matMechanicProp[2,2],
        "nu_zx": matMechanicProp[2,1],
        "nu_zy": matMechanicProp[2,0],
        "rho": 1.0
    },
    {
        "type": "isotropic",
        "e": matMechanicProp1[0],
        "nu": matMechanicProp1[1],
        "rho": 1.0
    }
]

materials_list = materials.array().tolist()
plane_orientations_list = plane_orientations.array().tolist()
fiber_orientations_list = fiber_orientations.array().tolist()

model_data = {
    "mesh": {
        "points": points,
        "cells": cells
    },
    "degree": 1,
    "matlibrary": matlibrary_data,
    "materials": materials_list,
    "plane_orientations": plane_orientations_list,
    "fiber_orientations": fiber_orientations_list,
    "scaling_constraint": 1.0e9,
    "singular": False
}

# Write to JSON
with open('model.json', 'w') as f:
    json.dump(model_data, f)

# Load from JSON
loaded_model = load_model_from_json('model.json')

# Use convenience function to translate to Dolfin
mesh2, matLibrary2, materials2, plane_orientations2, fiber_orientations2 = model_to_dolfin(loaded_model)

anba = anbax(mesh2, loaded_model.degree, matLibrary2, materials2, plane_orientations2, fiber_orientations2, scaling_constraint=loaded_model.scaling_constraint, singular=loaded_model.singular)
stiff = anba.compute()
mass = anba.inertia()

# Compute additional properties
stiff_mat = stiff.getValues(range(6), range(6))
mass_mat = mass.getValues(range(6), range(6))
decoupled_stiff = utils.DecoupleStiffness(stiff_mat)
angle = utils.PrincipalAxesRotationAngle(decoupled_stiff)
shear_center = utils.ComputeShearCenter(stiff_mat)
mass_center = utils.ComputeMassCenter(mass_mat)
tension_center = utils.ComputeTensionCenter(stiff_mat)

# Create output object
output = Output(
    stiffness_matrix=Matrix(data=stiff_mat.tolist()),
    mass_matrix=Matrix(data=mass_mat.tolist()),
    decoupled_stiffness_matrix=Matrix(data=decoupled_stiff.tolist()),
    principle_axis_orientation=angle,
    shear_center_location=shear_center,
    mass_center_location=mass_center,
    tension_elastic_center_location=tension_center
)

# Write output to JSON
with open('output.json', 'w') as f:
    json.dump(output.model_dump(), f)

stiff.view()

if __name__=='__test__':
    np.testing.assert_almost_equal(stiff.getValues(range(6),range(6)), 
                    np.array([[ 4.77322779e+05,  1.87291500e+05, -1.00040147e+05, -2.65592457e+02, -1.11914084e+02, -5.56374255e+01],
                              [ 1.87291500e+05,  1.15592180e+05, -4.24645230e+04, -1.11914078e+02, -4.94441224e+01, -2.36166857e+01],
                              [-1.00040147e+05, -4.24645230e+04,  3.00795578e+06, 3.10776505e+02,  1.31916800e+02,  7.41818375e+02],
                              [-2.65592457e+02, -1.11914078e+02,  3.10776505e+02, 3.38577775e+01, -7.79421592e+01,  1.59495464e-01],
                              [-1.11914084e+02, -4.94441224e+01,  1.31916800e+02, -7.79421592e+01,  1.84393514e+02,  6.77018038e-02],
                              [-5.56374255e+01, -2.36166857e+01,  7.41818375e+02, 1.59495464e-01,  6.77018038e-02,  8.30649734e-01]]),3)
