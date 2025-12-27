import pytest
import numpy as np
import dolfin
from anba4 import material
from anba4.data.anba_model import initialize_anba_model
from anba4.data.data_model import InputData
from anba4.solvers.stiffness import compute_stiffness
from anba4.solvers.inertia import compute_inertia
from anba4.fea.chains import initialize_chains
from anba4.fea.fe_functions import initialize_fe_functions


def parse_matrix(ref_str):
    lines = ref_str.strip().split("\n")
    mat = np.array([list(map(float, line.split())) for line in lines if line.strip()])
    return mat


@pytest.fixture(scope="module")
def multimat_with_hole_data():
    dolfin.parameters["form_compiler"]["optimize"] = True
    dolfin.parameters["form_compiler"]["quadrature_degree"] = 2
    matMechanicProp1 = [80000, 0.3]
    matMechanicProp2 = [80000 * 0.5, 0.3]
    matMechanicProp3 = [80000 * 0.00001, 0.3]
    sectionWidth = 20
    sectionHeight = 20
    mesh = dolfin.RectangleMesh(
        dolfin.Point(0.0, 0.0),
        dolfin.Point(sectionWidth, sectionHeight),
        50,
        50,
        "crossed",
    )
    dolfin.ALE.move(mesh, dolfin.Constant([-sectionWidth / 2.0, -sectionHeight / 2.0]))
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
    mat1 = material.IsotropicMaterial(matMechanicProp1[0], matMechanicProp1[1])
    mat2 = material.IsotropicMaterial(matMechanicProp2[0], matMechanicProp2[1])
    mat3 = material.IsotropicMaterial(matMechanicProp3[0], matMechanicProp3[1])
    matLibrary = [mat1, mat2, mat3]

    # Regular
    input_data_reg = InputData(
        mesh=mesh,
        degree=1,
        matLibrary=matLibrary,
        materials=materials,
        plane_orientations=plane_orientations,
        fiber_orientations=fiber_orientations,
        singular=False,
        scaling_constraint=1,
    )
    anbax_data_reg = initialize_anba_model(input_data_reg)
    initialize_fe_functions(anbax_data_reg)
    initialize_chains(anbax_data_reg)
    stiff_reg = compute_stiffness(anbax_data_reg)
    mass_reg = compute_inertia(anbax_data_reg)
    stiff_mat_reg = stiff_reg.getValues(range(6), range(6))
    mass_mat_reg = mass_reg.getValues(range(6), range(6))

    # Singular
    input_data_sing = InputData(
        mesh=mesh,
        degree=1,
        matLibrary=matLibrary,
        materials=materials,
        plane_orientations=plane_orientations,
        fiber_orientations=fiber_orientations,
        singular=True,
        scaling_constraint=1,
    )
    anbax_data_sing = initialize_anba_model(input_data_sing)
    initialize_fe_functions(anbax_data_sing)
    initialize_chains(anbax_data_sing)
    stiff_sing = compute_stiffness(anbax_data_sing)
    mass_sing = compute_inertia(anbax_data_sing)
    stiff_mat_sing = stiff_sing.getValues(range(6), range(6))
    mass_mat_sing = mass_sing.getValues(range(6), range(6))

    return {
        "stiff_reg": stiff_mat_reg,
        "stiff_sing": stiff_mat_sing,
        "mass_reg": mass_mat_reg,
        "mass_sing": mass_mat_sing,
    }


def test_multimat_with_hole_stiffness_regular_vs_singular(multimat_with_hole_data):
    np.testing.assert_allclose(
        multimat_with_hole_data["stiff_reg"],
        multimat_with_hole_data["stiff_sing"],
        atol=1e-5,
    )


def test_multimat_with_hole_mass_regular_vs_singular(multimat_with_hole_data):
    np.testing.assert_allclose(
        multimat_with_hole_data["mass_reg"],
        multimat_with_hole_data["mass_sing"],
        atol=1e-5,
    )


def test_multimat_with_hole_stiffness_reference(multimat_with_hole_data):
    ref_stiff = parse_matrix(reference_stiffness)
    np.testing.assert_allclose(
        multimat_with_hole_data["stiff_reg"], ref_stiff, atol=1e-5
    )


reference_stiffness = """3.6562114259374999e+06 -1.5183600001400947e+02 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 -1.0100430033504009e+07 
-1.5183599999479526e+02 5.8959962322773710e+06 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 -1.6802264333052363e+05 
0.0000000000000000e+00 0.0000000000000000e+00 2.1380834623998445e+07 3.7074245235198796e+07 3.9242147413326200e+05 0.0000000000000000e+00 
0.0000000000000000e+00 0.0000000000000000e+00 3.7074245235198833e+07 7.6962684297902262e+08 4.3342801843087474e+05 0.0000000000000000e+00 
0.0000000000000000e+00 0.0000000000000000e+00 3.9242147413333762e+05 4.3342801843092334e+05 7.9701404452929175e+08 0.0000000000000000e+00 
-1.0100430033504205e+07 -1.6802264333054138e+05 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 4.7752291234080762e+08 """
