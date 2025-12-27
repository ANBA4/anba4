import argparse
import os
import multiprocessing as mp
from .single_calculation import run_single_calculation


def safe_run_single_calculation(*args):
    """Safely run single calculation, catching exceptions."""
    try:
        return run_single_calculation(*args), None
    except Exception as e:
        return None, str(e)


def main():
    """Main CLI entry point for ANBA4 computations."""
    parser = argparse.ArgumentParser(
        description="CLI tool to run ANBA4 computations from JSON input and serialize outputs to JSON."
    )
    parser.add_argument(
        "-i",
        "--inputs",
        type=str,
        nargs="+",
        required=True,
        help="Input JSON files (one for single run, multiple for batch)",
    )
    parser.add_argument(
        "-o",
        "--output-post",
        type=str,
        default="_out",
        help="Postfix for output files (e.g., _out for file_out.json)",
    )
    parser.add_argument(
        "-r",
        "--reference",
        type=str,
        default="local",
        choices=["local", "global"],
        help="Reference system for fields (local or global)",
    )
    parser.add_argument(
        "-v",
        "--voigt",
        type=str,
        default="anba",
        choices=["anba", "paraview"],
        help="Voigt convention (anba or paraview)",
    )

    args = parser.parse_args()

    inputs = args.inputs
    num_runs = len(inputs)

    # Prepare output paths
    output_paths = []
    for inp in inputs:
        base = inp.replace(".json", "")
        output_paths.append(f"{base}{args.output_post}.json")

    if num_runs == 1:
        # Single run
        run_single_calculation(
            inputs[0],
            output_paths[0],
            args.reference,
            args.voigt,
        )
    else:
        # Batch processing: precompile JIT serially with first input, then run all in parallel
        print("Pre-compiling JIT...")
        temp_output = f"{inputs[0].replace('.json', '')}_temp.json"
        run_single_calculation(
            inputs[0],
            temp_output,
            args.reference,
            args.voigt,
        )
        os.remove(temp_output)  # Remove temp file

        # Prepare params for parallel runs
        params = list(
            zip(
                inputs,
                output_paths,
                [args.reference] * num_runs,
                [args.voigt] * num_runs,
            )
        )

        # Use 'spawn' start method for clean processes
        mp.set_start_method("spawn", force=True)

        with mp.Pool(processes=min(mp.cpu_count(), len(params))) as pool:
            results = pool.starmap(safe_run_single_calculation, params)

        # Process results
        succeeded = []
        failed = []
        for inp, (result, error) in zip(inputs, results):
            if error:
                failed.append((inp, error))
            else:
                succeeded.append(inp)

        # Print summary table
        print("\nBatch processing complete.")
        print(f"Succeeded ({len(succeeded)}):")
        for inp in succeeded:
            print(f"  {inp}")
        print(f"Failed ({len(failed)}):")
        for inp, err in failed:
            print(f"  {inp}: {err}")


if __name__ == "__main__":
    main()
