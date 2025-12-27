import pytest
import glob
import os
import subprocess
import sys


def execfile(filepath, globals=None, locals=None):
    """Execute a given python file"""
    if globals is None:
        globals = {"__name__": "__main__"}
    globals.update(
        {
            "__file__": filepath,
        }
    )
    with open(filepath, "rb") as file:
        exec(compile(file.read(), filepath, "exec"), globals, locals)


@pytest.mark.parametrize(
    "example_file",
    [
        f
        for f in glob.glob(os.path.join(os.path.dirname(__file__), "../examples/*.py"))
        if "QuadMesh.py" not in f and "test_" not in f
    ],
)
def test_run_example(example_file):
    """Test that each example script runs without errors."""
    # Use subprocess to run the script in a clean environment
    result = subprocess.run(
        [sys.executable, example_file], capture_output=True, text=True
    )
    assert result.returncode == 0, f"Example {example_file} failed: {result.stderr}"
