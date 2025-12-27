import json
import os
import dolfin
from .. import (
    initialize_anba_model,
    initialize_fe_functions,
    initialize_chains,
    compute_stiffness,
    compute_inertia,
    ComputeShearCenter,
    ComputeTensionCenter,
    ComputeMassCenter,
    DecoupleStiffness,
    PrincipalAxesRotationAngle,
)
from ..io.export import (
    import_model_json,
    serialize_matrix,
    serialize_numpy_matrix,
)
from .unit_loadcases import run_unit_loadcases


# Set shared cache dir early
os.environ["DIJITSO_CACHE_DIR"] = os.path.join(os.getcwd(), "cache")


def run_single_calculation(
    input_path: str,
    output_path: str,
    reference: str,
    voigt: str,
):
    """Run a single ANBA4 calculation."""
    # Print out to diagnose problem
    print("Dolfin version:", getattr(dolfin, "__version__", "unknown"))
    print(
        "Dolfin attributes with 'thread':",
        [attr for attr in dir(dolfin) if "thread" in attr.lower()],
    )
    print(
        "Dolfin parameters keys:",
        list(dolfin.parameters.keys())
        if hasattr(dolfin.parameters, "keys")
        else "no keys method",
    )
    # Disable parallel processing to prevent caching issues
    os.environ["OMP_NUM_THREADS"] = "1"
    dolfin.set_log_level(dolfin.LogLevel.WARNING)

    serializable_input = import_model_json(input_path)
    anbax_data = initialize_anba_model(serializable_input)
    initialize_fe_functions(anbax_data)
    initialize_chains(anbax_data)
    stiff = compute_stiffness(anbax_data)
    mass = compute_inertia(anbax_data)

    # Compute centers and angles
    shear_center = ComputeShearCenter(stiff)
    tension_center = ComputeTensionCenter(stiff)
    mass_center = ComputeMassCenter(mass)
    decoupled_stiff = DecoupleStiffness(stiff)
    principal_angle = PrincipalAxesRotationAngle(decoupled_stiff)

    output_data = {
        "stiffness": serialize_matrix(stiff),
        "mass": serialize_matrix(mass),
        "shear_center": shear_center,
        "tension_center": tension_center,
        "mass_center": mass_center,
        "decoupled_stiffness": serialize_numpy_matrix(decoupled_stiff),
        "principal_angle": principal_angle,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)

    unit_case_results = run_unit_loadcases(
        anbax_data,
        reference=reference,
        voigt=voigt,
    )

    vtu_path = f"{output_path.replace('.json', '')}_unit.vtu"
    unit_case_results.save(vtu_path)

    print(f"Unit output written to {vtu_path}")
    print(f"Outputs serialized to {output_path}")
    return output_data
