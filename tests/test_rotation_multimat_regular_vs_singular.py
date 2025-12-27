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
def rotation_multimat_data():
    dolfin.parameters["form_compiler"]["optimize"] = True
    dolfin.parameters["form_compiler"]["quadrature_degree"] = 2
    e_xx = 9.8e9
    e_yy = 9.8e9
    e_zz = 1.42e11
    g_xy = 4.8e9
    g_xz = 6.0e9
    g_yz = 6.0e9
    nu_xy = 0.34
    nu_zx = 0.3
    nu_zy = 0.3
    matMechanicProp1 = [9.8e9 / 100.0, 0.3]
    sectionWidth = 3.0023e-2
    sectionHeight = 1.9215e-3
    nPly = 16
    mesh = dolfin.RectangleMesh(
        dolfin.Point(0.0, 0.0),
        dolfin.Point(sectionWidth, sectionHeight),
        30,
        32,
        "crossed",
    )
    dolfin.ALE.move(mesh, dolfin.Constant([-sectionWidth / 2.0, -sectionHeight / 2.0]))
    materials = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim())
    fiber_orientations = dolfin.MeshFunction("double", mesh, mesh.topology().dim())
    plane_orientations = dolfin.MeshFunction("double", mesh, mesh.topology().dim())
    tol = 1e-14
    subdomain_0_p20 = dolfin.CompiledSubDomain(
        "x[1] >= -8.0*thickness - tol && x[1] <= -7.0*thickness + tol",
        thickness=sectionHeight / nPly,
        tol=tol,
    )
    subdomain_1_m70 = dolfin.CompiledSubDomain(
        "x[1] >= -7.0*thickness - tol && x[1] <= -6.0*thickness + tol",
        thickness=sectionHeight / nPly,
        tol=tol,
    )
    subdomain_2_m70 = dolfin.CompiledSubDomain(
        "x[1] >= -6.0*thickness - tol && x[1] <= -5.0*thickness + tol",
        thickness=sectionHeight / nPly,
        tol=tol,
    )
    subdomain_3_p20 = dolfin.CompiledSubDomain(
        "x[1] >= -5.0*thickness - tol && x[1] <= -4.0*thickness + tol",
        thickness=sectionHeight / nPly,
        tol=tol,
    )
    subdomain_4_p20 = dolfin.CompiledSubDomain(
        "x[1] >= -4.0*thickness - tol && x[1] <= -3.0*thickness + tol",
        thickness=sectionHeight / nPly,
        tol=tol,
    )
    subdomain_5_m70 = dolfin.CompiledSubDomain(
        "x[1] >= -3.0*thickness - tol && x[1] <= -2.0*thickness + tol",
        thickness=sectionHeight / nPly,
        tol=tol,
    )
    subdomain_6_m70 = dolfin.CompiledSubDomain(
        "x[1] >= -2.0*thickness - tol && x[1] <= -1.0*thickness + tol",
        thickness=sectionHeight / nPly,
        tol=tol,
    )
    subdomain_7_p20 = dolfin.CompiledSubDomain(
        "x[1] >= -1.0*thickness - tol && x[1] <= -0.0*thickness + tol",
        thickness=sectionHeight / nPly,
        tol=tol,
    )
    subdomain_8_m20 = dolfin.CompiledSubDomain(
        "x[1] >= 0.0*thickness - tol && x[1] <= 1.0*thickness + tol",
        thickness=sectionHeight / nPly,
        tol=tol,
    )
    subdomain_9_p70 = dolfin.CompiledSubDomain(
        "x[1] >= 1.0*thickness - tol && x[1] <= 2.0*thickness + tol",
        thickness=sectionHeight / nPly,
        tol=tol,
    )
    subdomain_10_p70 = dolfin.CompiledSubDomain(
        "x[1] >= 2.0*thickness - tol && x[1] <= 3.0*thickness + tol",
        thickness=sectionHeight / nPly,
        tol=tol,
    )
    subdomain_11_m20 = dolfin.CompiledSubDomain(
        "x[1] >= 3.0*thickness - tol && x[1] <= 4.0*thickness + tol",
        thickness=sectionHeight / nPly,
        tol=tol,
    )
    subdomain_12_m20 = dolfin.CompiledSubDomain(
        "x[1] >= 4.0*thickness - tol && x[1] <= 5.0*thickness + tol",
        thickness=sectionHeight / nPly,
        tol=tol,
    )
    subdomain_13_p70 = dolfin.CompiledSubDomain(
        "x[1] >= 5.0*thickness - tol && x[1] <= 6.0*thickness + tol",
        thickness=sectionHeight / nPly,
        tol=tol,
    )
    subdomain_14_p70 = dolfin.CompiledSubDomain(
        "x[1] >= 6.0*thickness - tol && x[1] <= 7.0*thickness + tol",
        thickness=sectionHeight / nPly,
        tol=tol,
    )
    subdomain_15_m20 = dolfin.CompiledSubDomain(
        "x[1] >= 7.0*thickness - tol && x[1] <= 8.0*thickness + tol",
        thickness=sectionHeight / nPly,
        tol=tol,
    )
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
    rotate = dolfin.Expression(
        (
            "x[0] * (cos(rotation_angle)-1.0) - x[1] * sin(rotation_angle)",
            "x[0] * sin(rotation_angle) + x[1] * (cos(rotation_angle)-1.0)",
        ),
        rotation_angle=rotation_angle * np.pi / 180.0,
        degree=1,
    )
    dolfin.ALE.move(mesh, rotate)
    mat1 = material.OrthotropicMaterial(
        e_xx, e_yy, e_zz, g_xy, g_xz, g_yz, nu_xy, nu_zx, nu_zy
    )
    mat2 = material.IsotropicMaterial(matMechanicProp1[0], matMechanicProp1[1])
    matLibrary = [mat1, mat2]

    # Regular
    input_data_reg = InputData(
        mesh=mesh,
        degree=1,
        matLibrary=matLibrary,
        materials=materials,
        plane_orientations=plane_orientations,
        fiber_orientations=fiber_orientations,
        singular=False,
        scaling_constraint=1.0e9,
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
        scaling_constraint=1.0e9,
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


def test_rotation_multimat_stiffness_regular_vs_singular(rotation_multimat_data):
    np.testing.assert_allclose(
        rotation_multimat_data["stiff_reg"],
        rotation_multimat_data["stiff_sing"],
        atol=1e-6,
    )


def test_rotation_multimat_mass_regular_vs_singular(rotation_multimat_data):
    np.testing.assert_allclose(
        rotation_multimat_data["mass_reg"],
        rotation_multimat_data["mass_sing"],
        atol=1e-6,
    )


def test_rotation_multimat_stiffness_reference(rotation_multimat_data):
    ref_stiff = parse_matrix(reference_stiffness_multimat)
    np.testing.assert_allclose(
        rotation_multimat_data["stiff_reg"], ref_stiff, atol=1e-6
    )


reference_stiffness_multimat = """4.7732277895589388e+05 1.8729150027703770e+05 -1.0004014718593998e+05 -2.6559245932415263e+02 -1.1191407847951135e+02 -5.5637423163899818e+01 
1.8729150027544342e+05 1.1559218008656151e+05 -4.2464523090341572e+04 -1.1191407847985531e+02 -4.9444120662803115e+01 -2.3616684972044165e+01 
-1.0004014718606474e+05 -4.2464523090388793e+04 3.0079557796313870e+06 3.1077650475276408e+02 1.3191679973707500e+02 7.4181837461959606e+02 
-2.6559245932250809e+02 -1.1191407848226883e+02 3.1077650475272691e+02 3.3857777528704339e+01 -7.7942159205097212e+01 1.5949546221770167e-01 
-1.1191407848220877e+02 -4.9444120655858107e+01 1.3191679973717240e+02 -7.7942159205096999e+01 1.8439351398588053e+02 6.7701807011101192e-02 
-5.5637423163929071e+01 -2.3616684971936071e+01 7.4181837461959753e+02 1.5949546221760513e-01 6.7701807011059351e-02 8.3064973690319510e-01 """
