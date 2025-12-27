#
# Copyright (C) 2018 Marco Morandini
#
# ----------------------------------------------------------------------
#
#    This file is part of Anba.
#
#    Anba is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
# ----------------------------------------------------------------------
#

import dolfin
from anba4 import (
    material,
    InputData,
    initialize_anba_model,
    initialize_fe_functions,
    initialize_chains,
    compute_stiffness,
    compute_inertia,
)

# Basic material parameters
dolfin.parameters["form_compiler"]["optimize"] = True
dolfin.parameters["form_compiler"]["quadrature_degree"] = 2
E = 1.0
nu = 0.33

# Meshing domain
mesh = dolfin.UnitSquareMesh(10, 10)
dolfin.ALE.move(mesh, dolfin.Constant([-0.5, -0.5]))

# CompiledSubDomain
materials = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim())
fiber_orientations = dolfin.MeshFunction("double", mesh, mesh.topology().dim())
plane_orientations = dolfin.MeshFunction("double", mesh, mesh.topology().dim())

materials.set_all(0)
fiber_orientations.set_all(0.0)
plane_orientations.set_all(90.0)

# Build material property library
mat1 = material.IsotropicMaterial(E, nu, 1.0)
matLibrary = [mat1]

# Create input data
input_data = InputData(
    mesh=mesh,
    degree=2,
    matLibrary=matLibrary,
    materials=materials,
    fiber_orientations=fiber_orientations,
    plane_orientations=plane_orientations,
    singular=True,
)

# Initialize model
anbax_data = initialize_anba_model(input_data)
initialize_fe_functions(anbax_data)
initialize_chains(anbax_data)

# Compute stiffness and mass
stiff = compute_stiffness(anbax_data)
mass = compute_inertia(anbax_data)

# Print results
print("Stiffness matrix:")
print(stiff.getValues(range(6), range(6)))
print("Mass matrix:")
print(mass.getValues(range(6), range(6)))
