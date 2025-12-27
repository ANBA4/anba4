import pyvista as pv
from .. import stress_field, strain_field
from ..io.export import dolfin_to_pyvista_mesh


def run_unit_loadcases(
    anbax_data,
    reference: str,
    voigt: str,
) -> pv.UnstructuredGrid:
    """Run unit load cases and return a PyVista UnstructuredGrid with stress and strain cell data."""
    unit_cases = {
        "fx": {"f": [1.0, 0.0, 0.0], "m": [0.0, 0.0, 0.0]},
        "fy": {"f": [0.0, 1.0, 0.0], "m": [0.0, 0.0, 0.0]},
        "fz": {"f": [0.0, 0.0, 1.0], "m": [0.0, 0.0, 0.0]},
        "mx": {"f": [0.0, 0.0, 0.0], "m": [1.0, 0.0, 0.0]},
        "my": {"f": [0.0, 0.0, 0.0], "m": [0.0, 1.0, 0.0]},
        "mz": {"f": [0.0, 0.0, 0.0], "m": [0.0, 0.0, 1.0]},
    }

    # build unstructuredgrid for this case
    case_output = dolfin_to_pyvista_mesh(anbax_data.input_data.mesh)

    # attach outputs to cell data
    for case_name, case in unit_cases.items():
        stress = stress_field(
            anbax_data,
            case["f"],
            case["m"],
            reference=reference,
            voigt_convention=voigt,
        )
        strain = strain_field(
            anbax_data,
            case["f"],
            case["m"],
            reference=reference,
            voigt_convention=voigt,
        )

        case_output.cell_data[f"Stress_{case_name}"] = (
            stress.vector().get_local().reshape(-1, 6)
        )
        case_output.cell_data[f"Strain_{case_name}"] = (
            strain.vector().get_local().reshape(-1, 6)
        )

    return case_output
