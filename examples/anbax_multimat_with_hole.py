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
matMechanicProp1 = [80000, 0.3]
matMechanicProp2 = [80000 * 0.5, 0.3]
matMechanicProp3 = [80000 * 0.00001, 0.3]
sectionWidth = 20
sectionHeight = 20

# Meshing domain
mesh = dolfin.RectangleMesh(
    dolfin.Point(0.0, 0.0),
    dolfin.Point(sectionWidth, sectionHeight),
    50,
    50,
    "crossed",
)
dolfin.ALE.move(mesh, dolfin.Constant([-sectionWidth / 2.0, -sectionHeight / 2.0]))

# CompiledSubDomain
materials = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim())
fiber_orientations = dolfin.MeshFunction("double", mesh, mesh.topology().dim())
plane_orientations = dolfin.MeshFunction("double", mesh, mesh.topology().dim())
tol = 1e-14
lower_portion = dolfin.CompiledSubDomain("x[1] <= 0 + tol", tol=tol)
hole = dolfin.CompiledSubDomain(
    "(x[1] >= -6 + tol && x[1] <= 6. + tol)&&(x[0] >= -2 + tol && x[0] <= 2. + tol)",
    tol=tol,
)
rotation_angle = 0.0
materials.set_all(0)
fiber_orientations.set_all(0.0)
plane_orientations.set_all(rotation_angle)
lower_portion.mark(materials, 1)
hole.mark(materials, 2)

# Build material property library
mat1 = material.IsotropicMaterial(matMechanicProp1[0], matMechanicProp1[1])
mat2 = material.IsotropicMaterial(matMechanicProp2[0], matMechanicProp2[1])
mat3 = material.IsotropicMaterial(matMechanicProp3[0], matMechanicProp3[1])
matLibrary = [mat1, mat2, mat3]

# Create input data
input_data = InputData(
    mesh=mesh,
    degree=1,
    matLibrary=matLibrary,
    materials=materials,
    fiber_orientations=fiber_orientations,
    plane_orientations=plane_orientations,
    scaling_constraint=1,
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

# Output Jordan chains
output_file = "anbax_multimat_with_hole.xdmf"
JordanChains = dolfin.XDMFFile(output_file)
JordanChains.parameters["functions_share_mesh"] = True
JordanChains.parameters["rewrite_function_mesh"] = False
JordanChains.parameters["flush_output"] = True
for i in range(len(anbax_data.chains.chains_d)):
    for j in range(len(anbax_data.chains.chains_d[i])):
        print("chain_" + str(i) + "_" + str(j))
        chain = dolfin.Function(
            anbax_data.fe_functions.UF3, name="chain_" + str(i) + "_" + str(j)
        )
        chain.vector()[:] = dolfin.project(
            anbax_data.chains.chains_d[i][j], anbax_data.fe_functions.UF3
        ).vector()
        JordanChains.write(chain, t=0.0)

print(f"Jordan chains written to file: {output_file}")
