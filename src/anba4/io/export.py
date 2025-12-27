import json
import numpy as np
import pyvista as pv
import dolfin
from petsc4py import PETSc
from typing import List

from ..data.data_model import InputData, SerializableInputData


def dolfin_to_pyvista_mesh(mesh: dolfin.Mesh) -> pv.UnstructuredGrid:
    """Convert a Dolfin mesh to a PyVista UnstructuredGrid."""
    pts = np.hstack((mesh.coordinates(), np.zeros((mesh.coordinates().shape[0], 1))))
    cells = np.hstack(
        (3 * np.ones((mesh.cells().shape[0], 1)).astype(np.int64), mesh.cells())
    )
    grd = pv.UnstructuredGrid(
        cells,
        [pv.CellType.TRIANGLE for i in range(mesh.cells().shape[0])],
        pts,
    )
    return grd


def export_model_vtu(
    mesh: dolfin.Mesh,
    materials: dolfin.MeshFunction,
    fiber_orientations: dolfin.MeshFunction,
    plane_orientations: dolfin.MeshFunction,
    mesh_name: str = "mesh.vtu",
):
    """Export model to VTU format using PyVista."""
    grd = dolfin_to_pyvista_mesh(mesh)

    # pts = np.hstack((mesh.coordinates(), np.zeros((mesh.coordinates().shape[0], 1))))
    # cells = np.hstack(
    #     (3 * np.ones((mesh.cells().shape[0], 1)).astype(np.int64), mesh.cells())
    # )
    # grd = pv.UnstructuredGrid(
    #     cells,
    #     [pv.CellType.TRIANGLE for i in range(mesh.cells().shape[0])],
    #     pts,
    # )
    grd.cell_data["Materials"] = materials.array()
    grd.cell_data["FiberOrientations"] = fiber_orientations.array()
    grd.cell_data["PlaneOrientations"] = plane_orientations.array()
    grd.save(mesh_name)


def export_model_json(
    input_data: InputData,
    filename: str = "model.json",
):
    """Export model input data to JSON format using InputData's to_dict."""
    data = input_data.to_dict()
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


def import_model_json(filename: str = "model.json") -> SerializableInputData:
    """Import model input data from JSON format."""
    with open(filename, "r") as f:
        data = json.load(f)
    return SerializableInputData(**data)


def serialize_matrix(matrix: PETSc.Mat) -> List[List[float]]:
    """Serialize PETSc matrix to list of lists."""
    rows, cols = matrix.getSize()
    mat_list = []
    for i in range(rows):
        row = [float(matrix[i, j]) for j in range(cols)]
        mat_list.append(row)
    return mat_list


def serialize_numpy_matrix(matrix: np.ndarray) -> List[List[float]]:
    """Serialize numpy matrix to list of lists."""
    return [[float(val) for val in row] for row in matrix]


def serialize_field(field: dolfin.Function) -> List[List[float]]:
    """Serialize Dolfin vector field to list of lists (per component)."""
    array = field.vector().get_local()
    dim = field.function_space().ufl_element().value_shape()[0]
    reshaped = array.reshape(-1, dim)
    return [[float(val) for val in row] for row in reshaped]
