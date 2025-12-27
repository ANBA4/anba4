import pytest
import numpy as np
import dolfin
from anba4 import material, utils
from anba4.data.anba_model import initialize_anba_model
from anba4.data.data_model import InputData
from anba4.solvers.stiffness import compute_stiffness
from anba4.solvers.inertia import compute_inertia
import mshr
from anba4.fea.chains import initialize_chains
from anba4.fea.fe_functions import initialize_fe_functions


def parse_matrix(ref_str):
    lines = ref_str.strip().split("\n")
    mat = np.array([list(map(float, line.split())) for line in lines if line.strip()])
    return mat


@pytest.fixture(scope="module")
def principal_axes_data():
    dolfin.parameters["form_compiler"]["optimize"] = True
    dolfin.parameters["form_compiler"]["quadrature_degree"] = 2
    E = 1.0
    nu = 0.33
    matMechanicProp = [E, nu]
    thickness = 0.1
    Square1 = mshr.Rectangle(dolfin.Point(0.0, -1.0, 0.0), dolfin.Point(1.0, 1.0, 0.0))
    Square2 = mshr.Rectangle(
        dolfin.Point(thickness, -1 + thickness, 0),
        dolfin.Point(2.0, 1.0 - thickness, 0),
    )
    C_shape = Square1 - Square2
    mesh = mshr.generate_mesh(C_shape, 64)
    rot_angle = 30.0 / 180.0 * np.pi
    cr = np.cos(rot_angle)
    sr = np.sin(rot_angle)
    rot_tensor = np.array([[cr, -sr], [sr, cr]])
    mesh.coordinates()[:] = (rot_tensor @ mesh.coordinates().T).T
    mesh.coordinates()[:] += np.array([3, 1])
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
        fiber_orientations=fiber_orientations,
        plane_orientations=plane_orientations,
        singular=False,
    )
    anbax_data_reg = initialize_anba_model(input_data_reg)
    initialize_fe_functions(anbax_data_reg)
    initialize_chains(anbax_data_reg)
    stiff_reg = compute_stiffness(anbax_data_reg)
    mass_reg = compute_inertia(anbax_data_reg)
    stiff_mat_reg = stiff_reg.getValues(range(6), range(6))
    decoupled_stiff_reg = utils.DecoupleStiffness(stiff_mat_reg)
    mass_mat_reg = mass_reg.getValues(range(6), range(6))
    angle_reg = utils.PrincipalAxesRotationAngle(decoupled_stiff_reg)

    # Singular
    input_data_sing = InputData(
        mesh=mesh,
        degree=2,
        matLibrary=matLibrary,
        materials=materials,
        fiber_orientations=fiber_orientations,
        plane_orientations=plane_orientations,
        singular=True,
    )
    anbax_data_sing = initialize_anba_model(input_data_sing)
    initialize_fe_functions(anbax_data_sing)
    initialize_chains(anbax_data_sing)
    stiff_sing = compute_stiffness(anbax_data_sing)
    mass_sing = compute_inertia(anbax_data_sing)
    stiff_mat_sing = stiff_sing.getValues(range(6), range(6))
    decoupled_stiff_sing = utils.DecoupleStiffness(stiff_mat_sing)
    mass_mat_sing = mass_sing.getValues(range(6), range(6))
    angle_sing = utils.PrincipalAxesRotationAngle(decoupled_stiff_sing)

    return {
        "stiff_reg": stiff_mat_reg,
        "stiff_sing": stiff_mat_sing,
        "mass_reg": mass_mat_reg,
        "mass_sing": mass_mat_sing,
        "dec_stiff_reg": decoupled_stiff_reg,
        "dec_stiff_sing": decoupled_stiff_sing,
        "angle_reg": angle_reg,
        "angle_sing": angle_sing,
    }


def test_principal_axes_stiffness_regular_vs_singular(principal_axes_data):
    np.testing.assert_allclose(
        principal_axes_data["stiff_reg"], principal_axes_data["stiff_sing"], atol=1e-6
    )


def test_principal_axes_mass_regular_vs_singular(principal_axes_data):
    np.testing.assert_allclose(
        principal_axes_data["mass_reg"], principal_axes_data["mass_sing"], atol=1e-6
    )


def test_principal_axes_dec_stiff_regular_vs_singular(principal_axes_data):
    np.testing.assert_allclose(
        principal_axes_data["dec_stiff_reg"],
        principal_axes_data["dec_stiff_sing"],
        atol=1e-6,
    )


def test_principal_axes_angle_regular_vs_singular(principal_axes_data):
    np.testing.assert_allclose(
        principal_axes_data["angle_reg"], principal_axes_data["angle_sing"], atol=1e-6
    )


def test_principal_axes_stiffness_reference(principal_axes_data):
    ref_stiff = parse_matrix(reference_stiffness)
    np.testing.assert_allclose(principal_axes_data["stiff_reg"], ref_stiff, atol=1e-6)


def test_principal_axes_mass_reference(principal_axes_data):
    ref_mass = parse_matrix(reference_mass)
    np.testing.assert_allclose(principal_axes_data["mass_reg"], ref_mass, atol=1e-6)


def test_principal_axes_dec_stiff_reference(principal_axes_data):
    ref_dec = parse_matrix(decoupled_stiffness)
    np.testing.assert_allclose(principal_axes_data["dec_stiff_reg"], ref_dec, atol=1e-6)


def test_principal_axes_angle_reference(principal_axes_data):
    np.testing.assert_allclose(
        principal_axes_data["angle_reg"] * 180 / np.pi, rotation_angle, atol=1e-3
    )


def test_principal_axes_dec_stiff_zeros(principal_axes_data):
    np.testing.assert_allclose(
        principal_axes_data["dec_stiff_reg"][0:3, 3:6], np.zeros((3, 3)), atol=1e-6
    )
    np.testing.assert_allclose(
        principal_axes_data["dec_stiff_reg"][3:6, 0:3], np.zeros((3, 3)), atol=1e-6
    )


reference_stiffness = """4.9983692051719798e-02 -6.4216083302349259e-03 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 -6.0014130758270327e-02 
-6.4216083302310063e-03 5.7386051916386377e-02 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 1.6258740045203857e-01 
0.0000000000000000e+00 0.0000000000000000e+00 3.7999999999938078e-01 4.3449999999926625e-01 -1.2343967690104505e+00 0.0000000000000000e+00 
0.0000000000000000e+00 0.0000000000000000e+00 4.3449999999926836e-01 6.7776666666571550e-01 -1.3277487113035373e+00 0.0000000000000000e+00 
0.0000000000000000e+00 0.0000000000000000e+00 -1.2343967690104500e+00 -1.3277487113035293e+00 4.0941472807347932e+00 0.0000000000000000e+00 
-6.0014130758255915e-02 1.6258740045203884e-01 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 4.9662070024812510e-01 """
reference_mass = """3.7999999999999967e-01 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 -4.3450000000000011e-01 
0.0000000000000000e+00 3.7999999999999967e-01 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 1.2343967690125053e+00 
0.0000000000000000e+00 0.0000000000000000e+00 3.7999999999999967e-01 4.3450000000000011e-01 -1.2343967690125053e+00 0.0000000000000000e+00 
0.0000000000000000e+00 0.0000000000000000e+00 4.3450000000000011e-01 6.7776666666666541e-01 -1.3277487113059632e+00 0.0000000000000000e+00 
0.0000000000000000e+00 0.0000000000000000e+00 -1.2343967690125053e+00 -1.3277487113059632e+00 4.0941472807416801e+00 0.0000000000000000e+00 
-4.3450000000000011e-01 1.2343967690125053e+00 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 4.7719139474083487e+00 """

decoupled_stiffness = """ 4.99836921e-02 -6.42160833e-03  0.00000000e+00  0.00000000e+00   0.00000000e+00  6.93889390e-18
   -6.42160833e-03  5.73860519e-02  0.00000000e+00  0.00000000e+00   0.00000000e+00 -2.77555756e-17
    0.00000000e+00  0.00000000e+00  3.80000000e-01 -5.55111512e-17   0.00000000e+00  0.00000000e+00
    0.00000000e+00  0.00000000e+00  2.05391260e-15  1.80950219e-01   8.36865417e-02  0.00000000e+00
    0.00000000e+00  0.00000000e+00  4.44089210e-16  8.36865417e-02   8.43173246e-02  0.00000000e+00
    3.68455266e-15 -3.08086889e-15  0.00000000e+00  0.00000000e+00   0.00000000e+00  4.74048372e-04"""
rotation_angle = 29.99999999999973
