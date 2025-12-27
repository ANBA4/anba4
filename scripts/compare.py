import re
import json
import click
from typing import List, Dict, Tuple


def extract_numbers(text: str) -> List[float]:
    """Extract all floating-point numbers from text by splitting."""
    numbers = []
    for line in text.splitlines():
        for token in line.split():
            try:
                numbers.append(float(token))
            except ValueError:
                pass
    return numbers


def split_into_sections(output: str) -> Dict[str, str]:
    """Split output into sections by filename."""
    sections = {}
    lines = output.splitlines()
    current_file = None
    current_content = []
    for line in lines:
        if line.endswith(".py"):
            if current_file:
                sections[current_file] = "\n".join(current_content)
            current_file = line.strip()
            current_content = []
        else:
            if current_file:
                current_content.append(line)
    if current_file:
        sections[current_file] = "\n".join(current_content)
    return sections


def normalize_section(text: str) -> str:
    """Normalize text by removing non-essential parts like warnings, paths, and specific strings."""
    lines = text.splitlines()
    normalized = []
    skip_patterns = [
        r"^/home/wr1/.*",  # Skip paths
        r"^qt.qpa.plugin:.*",  # Skip Qt warnings
        r"^Traceback.*",  # Skip tracebacks
        r"^File \"",
        r"^RuntimeError:.*",
        r"^M norm.*",
        r"^H norm.*",
        r"^E norm.*",
        r"^Solving.*",
    ]
    for line in lines:
        if any(re.match(pat, line) for pat in skip_patterns):
            continue
        # Remove "Mat Object: 1 MPI process type: seqdense"
        if "Mat Object" in line:
            continue
        # Remove "Stress and strain fields written to ..."
        if "written to" in line.lower():
            continue
        normalized.append(line)
    return "\n".join(normalized)


def compare_sections(
    sec1: Dict[str, str], sec2: Dict[str, str], tol: float = 1e-8
) -> Dict[str, Tuple[bool, List[Tuple[int, float, float]], List[Tuple[int, float]]]]:
    """Compare numbers in corresponding sections after normalization."""
    results = {}
    files = sorted(set(sec1.keys()) & set(sec2.keys()))
    for file in files:
        norm_text1 = normalize_section(sec1[file])
        norm_text2 = normalize_section(sec2[file])
        nums1 = extract_numbers(norm_text1)
        nums2 = extract_numbers(norm_text2)
        if len(nums1) != len(nums2):
            results[file] = (
                False,
                [],
                [],
                f"Different number of values: {len(nums1)} vs {len(nums2)}",
                len(nums1),
            )
            continue
        mismatches = []
        matches = []
        match = True
        for i, (n1, n2) in enumerate(zip(nums1, nums2)):
            if abs(n1 - n2) > tol:
                match = False
                mismatches.append((i, n1, n2))
            else:
                matches.append((i, n1))
        results[file] = (
            match,
            mismatches,
            matches,
            None if match else "Mismatches found",
            len(nums1),
        )
    return results


@click.command()
@click.argument("json_file")
@click.option(
    "--tol", default=1e-8, help="Tolerance for floating point comparison", type=float
)
def main(json_file, tol):
    """CLI to compare two test.out contents from a JSON file."""
    with open(json_file, "r") as f:
        data = json.load(f)
    files = data["files"]
    if len(files) != 2:
        print("Expected exactly two files in JSON")
        return
    output1 = files[0]["content"]
    output2 = files[1]["content"]
    print(f"Comparing {files[0]['path']} and {files[1]['path']}")
    sec1 = split_into_sections(output1)
    sec2 = split_into_sections(output2)
    comparisons = compare_sections(sec1, sec2, tol)
    for pyfile, (match, mismatches, matches, msg, num_count) in sorted(
        comparisons.items()
    ):
        if match:
            print(
                f"{pyfile}: All {num_count} extracted numbers match within tolerance {tol}"
            )
        else:
            match_count = len(matches)
            print(f"{pyfile}: DIFFERENCE - {msg}")
            print(
                f"{match_count} out of {num_count} numbers match within tolerance {tol}"
            )
            if matches:
                print("Matching numbers (index, value):")
                for idx, val in matches:
                    print(f"  Index {idx}: {val}")
            if mismatches:
                print("Mismatched numbers (index, val1, val2):")
                for idx, val1, val2 in mismatches:
                    print(f"  Index {idx}: {val1} vs {val2}")
            elif msg:
                print(msg)


if __name__ == "__main__":
    main()
