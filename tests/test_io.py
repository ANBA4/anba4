import pytest
import numpy as np
from anba4.io import Model, Output, Matrix, load_model_from_json, model_to_dolfin
from anba4 import anbax, utils


def test_io():
    # Create model data (similar to the example)
    model_data = {
        "mesh": {
            "points": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],  # simple quad
            "cells": [[0, 1, 3], [0, 3, 2]]  # two triangles
        },
        "degree": 1,
        "matlibrary": [
            {
                "type": "isotropic",
                "e": 1.0,
                "nu": 0.3,
                "rho": 1.0
            }
        ],
        "materials": [0, 0],
        "plane_orientations": [0.0, 0.0],
        "fiber_orientations": [0.0, 0.0],
        "scaling_constraint": 1e9,
        "singular": False
    }

    # Load model
    model = Model(**model_data)

    # Translate to Dolfin
    mesh, matLibrary, materials, plane_orientations, fiber_orientations = model_to_dolfin(model)

    # Run anba
    anba = anbax(mesh, model.degree, matLibrary, materials, plane_orientations, fiber_orientations, scaling_constraint=model.scaling_constraint, singular=model.singular)
    stiff = anba.compute()
    mass = anba.inertia()

    # Compute output properties
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

    # Assertions to demonstrate the structure
    assert isinstance(output.stiffness_matrix.data, list)
    assert len(output.stiffness_matrix.data) == 6
    assert all(len(row) == 6 for row in output.stiffness_matrix.data)
    assert isinstance(output.principle_axis_orientation, float)
    assert isinstance(output.shear_center_location, list)
    assert len(output.shear_center_location) == 2
    assert isinstance(output.mass_center_location, list)
    assert len(output.mass_center_location) == 2
    assert isinstance(output.tension_elastic_center_location, list)
    assert len(output.tension_elastic_center_location) == 2
