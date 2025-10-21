import argparse
import json
from anba4.io import load_model_from_json, model_to_dolfin, Output, Matrix
from anba4 import anbax, utils


def main():
    parser = argparse.ArgumentParser(description="ANBA4 CLI")
    parser.add_argument("input", help="Input JSON file")
    parser.add_argument("output", help="Output JSON file")
    args = parser.parse_args()

    # Load model
    model = load_model_from_json(args.input)

    # Translate to Dolfin
    mesh, matLibrary, materials, plane_orientations, fiber_orientations = (
        model_to_dolfin(model)
    )

    # Run anba
    anba = anbax(
        mesh,
        model.degree,
        matLibrary,
        materials,
        plane_orientations,
        fiber_orientations,
        scaling_constraint=model.scaling_constraint,
        singular=model.singular,
    )
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
        tension_elastic_center_location=tension_center,
    )

    # Write to output JSON
    with open(args.output, "w") as f:
        json.dump(output.model_dump(), f)


if __name__ == "__main__":
    main()
