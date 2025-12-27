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
def isotropic_data():
    dolfin.parameters["form_compiler"]["optimize"] = True
    dolfin.parameters["form_compiler"]["quadrature_degree"] = 2
    E = 1.0
    nu = 0.33
    matMechanicProp = [E, nu]
    mesh = dolfin.UnitSquareMesh(10, 10)
    dolfin.ALE.move(mesh, dolfin.Constant([-0.5, -0.5]))
    materials = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim())
    fiber_orientations = dolfin.MeshFunction("double", mesh, mesh.topology().dim())
    plane_orientations = dolfin.MeshFunction("double", mesh, mesh.topology().dim())
    materials.set_all(0)
    fiber_orientations.set_all(0.0)
    plane_orientations.set_all(90.0)
    mat1 = material.IsotropicMaterial(matMechanicProp[0], matMechanicProp[1], 1.0)
    matLibrary = [mat1]

    # Regular
    input_data_reg = InputData(
        mesh=mesh,
        degree=2,
        matLibrary=matLibrary,
        materials=materials,
        plane_orientations=plane_orientations,
        fiber_orientations=fiber_orientations,
        singular=False,
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
        degree=2,
        matLibrary=matLibrary,
        materials=materials,
        plane_orientations=plane_orientations,
        fiber_orientations=fiber_orientations,
        singular=True,
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


def test_isotropic_stiffness_regular_vs_singular(isotropic_data):
    np.testing.assert_allclose(
        isotropic_data["stiff_reg"], isotropic_data["stiff_sing"], atol=1e-6
    )


def test_isotropic_mass_regular_vs_singular(isotropic_data):
    np.testing.assert_allclose(
        isotropic_data["mass_reg"], isotropic_data["mass_sing"], atol=1e-6
    )


def test_isotropic_stiffness_reference(isotropic_data):
    ref_stiff = parse_matrix(reference_stiffness)
    np.testing.assert_allclose(isotropic_data["stiff_reg"], ref_stiff, atol=1e-6)


def test_isotropic_mass_reference(isotropic_data):
    ref_mass = parse_matrix(reference_mass)
    np.testing.assert_allclose(isotropic_data["mass_reg"], ref_mass, atol=1e-6)


reference_stiffness = """3.1106440126718432e-01 -5.7626764670407046e-07 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 1.8332325847382240e-16 
-5.7626764651431983e-07 3.1106440126718449e-01 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 -2.3565142889073543e-16 
0.0000000000000000e+00 0.0000000000000000e+00 9.9999999999998768e-01 -3.9407074330915428e-16 7.1891020955786236e-17 0.0000000000000000e+00 
0.0000000000000000e+00 0.0000000000000000e+00 -4.2103095762601342e-16 8.3333333333332371e-02 -3.4516100351236347e-17 0.0000000000000000e+00 
0.0000000000000000e+00 0.0000000000000000e+00 8.2056361062066854e-17 -3.6214433702089572e-17 8.3333333333332565e-02 0.0000000000000000e+00 
1.8889437218922982e-16 -2.1179912487034412e-16 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 5.2855945355920649e-02 """

reference_mass = """1.0000000000000007e+00 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 3.6429192995512949e-17 
0.0000000000000000e+00 1.0000000000000007e+00 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 -1.3010426069826053e-17 
0.0000000000000000e+00 0.0000000000000000e+00 1.0000000000000007e+00 -3.6429192995512949e-17 1.3010426069826053e-17 0.0000000000000000e+00 
0.0000000000000000e+00 0.0000000000000000e+00 -3.6429192995512949e-17 8.3333333333333245e-02 -2.6020852139652106e-18 0.0000000000000000e+00 
0.0000000000000000e+00 0.0000000000000000e+00 1.3010426069826053e-17 -2.6020852139652106e-18 8.3333333333333259e-02 0.0000000000000000e+00 
3.6429192995512949e-17 -1.3010426069826053e-17 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 1.6666666666666671e-01"""
