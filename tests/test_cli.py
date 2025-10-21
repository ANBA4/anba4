import pytest
import json
import subprocess
import tempfile
import os
from anba4.io import Output


def test_cli():
    # Create model data (same as test_io)
    model_data = {
        "mesh": {
            "points": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            "cells": [[0, 1, 3], [0, 3, 2]]
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

    # Write to temp input file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(model_data, f)
        input_file = f.name

    # Temp output file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        output_file = f.name

    try:
        # Run CLI
        result = subprocess.run(["anba4", input_file, output_file], capture_output=True, text=True)
        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        # Load output
        with open(output_file) as f:
            output_data = json.load(f)

        output = Output(**output_data)

        # Assertions (same as test_io)
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

    finally:
        # Clean up
        os.unlink(input_file)
        os.unlink(output_file)
