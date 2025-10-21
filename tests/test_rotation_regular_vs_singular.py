import pytest
import numpy as np
import dolfin
from anba4 import *


def parse_matrix(ref_str):
    lines = ref_str.strip().split("\n")
    mat = np.array([list(map(float, line.split())) for line in lines if line.strip()])
    return mat


@pytest.fixture(scope="module")
def rotation_data():
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
    rotation_angle = 0.0
    materials.set_all(0)
    fiber_orientations.set_all(0.0)
    plane_orientations.set_all(rotation_angle)
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
    mat1 = material.OrthotropicMaterial(matMechanicProp)
    matLibrary = [mat1]

    anba = anbax(mesh, 1, matLibrary, materials, plane_orientations, fiber_orientations, scaling_constraint=1.0e9)
    stiff_reg = anba.compute()
    mass_reg = anba.inertia()
    stiff_mat_reg = stiff_reg.getValues(range(6), range(6))
    mass_mat_reg = mass_reg.getValues(range(6), range(6))

    anba = anbax(
        mesh,
        1,
        matLibrary,
        materials,
        plane_orientations,
        fiber_orientations,
        scaling_constraint=1.0e9,
        singular=True,
    )
    stiff_sing = anba.compute()
    mass_sing = anba.inertia()
    stiff_mat_sing = stiff_sing.getValues(range(6), range(6))
    mass_mat_sing = mass_sing.getValues(range(6), range(6))

    return {
        "stiff_reg": stiff_mat_reg,
        "stiff_sing": stiff_mat_sing,
        "mass_reg": mass_mat_reg,
        "mass_sing": mass_mat_sing,
    }


def test_rotation_stiffness_regular_vs_singular(rotation_data):
    np.testing.assert_allclose(
        rotation_data["stiff_reg"], rotation_data["stiff_sing"], atol=1e-5
    )


def test_rotation_mass_regular_vs_singular(rotation_data):
    np.testing.assert_allclose(
        rotation_data["mass_reg"], rotation_data["mass_sing"], atol=1e-5
    )


def test_rotation_stiffness_reference(rotation_data):
    ref_stiff = parse_matrix(reference_rotation)
    np.testing.assert_allclose(rotation_data["stiff_reg"], ref_stiff, atol=1e-5)


reference_rotation = """5.9898449578112806e+05 -1.7724237108734714e-08 1.1500601625315664e-08 -4.1049059250476665e+02 6.4406313317207103e-11 -1.7332964943005700e-10 
-9.0152418270960920e-09 4.7576085480708818e+04 -2.8008310305338603e-09 2.0746175841077142e-13 -5.6671852577577813e+00 9.8807865623602321e-11 
2.1143702390611879e-08 -2.4252564323838121e-09 3.3807303574125483e+06 -3.1425491039876107e-13 1.2097765172479667e-11 9.7293345483135340e+02 
-4.1049059250429264e+02 2.8188994600485389e-12 -5.3146972385613745e-12 1.0850078627180384e+00 1.9460995072296966e-14 4.7246562250279446e-14 
3.3573507712346574e-11 -5.6671852577625588e+00 -2.2657525208125019e-12 1.1724186559643436e-14 2.4436613590785532e+02 -3.5550011781782319e-14 
-8.7933649533394047e-11 9.9015219573421502e-11 9.7293345483134340e+02 -1.5645854352725462e-14 5.8721692907206775e-15 1.0698777549571612e+00 """
