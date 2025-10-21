from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Union, Tuple, Literal
import json
from dolfin import Mesh, MeshEditor, MeshFunction
import numpy as np
from anba4 import material


class MeshIO(BaseModel):
    points: List[List[float]]  # list of points, each point is [x, y] or [x, y, z]
    cells: List[List[int]]  # list of cells, each cell is list of point indices


class OrthotropicMaterial(BaseModel):
    type: Literal["orthotropic"]
    e_xx: float
    e_yy: float
    e_zz: float
    g_xy: float
    g_xz: float
    g_yz: float
    nu_xy: float
    nu_zx: float
    nu_zy: float
    rho: float


class IsotropicMaterial(BaseModel):
    type: Literal["isotropic"]
    e: float
    nu: float
    rho: float


Material = Union[OrthotropicMaterial, IsotropicMaterial]


class Model(BaseModel):
    mesh: MeshIO
    degree: int
    matlibrary: List[Material]
    materials: List[int]
    plane_orientations: List[float]
    fiber_orientations: List[float]
    scaling_constraint: float = 1e9
    singular: bool = False

    @model_validator(mode='after')
    def validate_model(self):
        num_cells = len(self.mesh.cells)
        if len(self.materials) != num_cells:
            raise ValueError(f"materials must have length {num_cells}")
        if len(self.plane_orientations) != num_cells:
            raise ValueError(f"plane_orientations must have length {num_cells}")
        if len(self.fiber_orientations) != num_cells:
            raise ValueError(f"fiber_orientations must have length {num_cells}")
        max_mat = len(self.matlibrary) - 1
        if any(x < 0 or x > max_mat for x in self.materials):
            raise ValueError(f"materials must be between 0 and {max_mat}")
        return self


class Matrix(BaseModel):
    data: List[List[float]]


class Output(BaseModel):
    stiffness_matrix: Matrix
    mass_matrix: Matrix
    decoupled_stiffness_matrix: Matrix
    principle_axis_orientation: float
    shear_center_location: List[float]  # [x, y]
    mass_center_location: List[float]  # [x, y]
    tension_elastic_center_location: List[float]  # [x, y]


# Stress/strain output left unimplemented


def load_model_from_json(filepath: str) -> Model:
    with open(filepath) as f:
        data = json.load(f)
    return Model(**data)


def model_to_dolfin(model: Model) -> Tuple[Mesh, List, MeshFunction, MeshFunction, MeshFunction]:
    # Translate IO mesh to Dolfin mesh
    mesh = Mesh()
    editor = MeshEditor()
    # Assume 2D triangular mesh
    editor.open(mesh, "triangle", 2, 2)
    editor.init_vertices(len(model.mesh.points))
    for i, p in enumerate(model.mesh.points):
        editor.add_vertex(i, p)
    editor.init_cells(len(model.mesh.cells))
    for i, c in enumerate(model.mesh.cells):
        editor.add_cell(i, c)
    editor.close()

    # Create MeshFunctions
    materials_mf = MeshFunction("size_t", mesh, mesh.topology().dim())
    materials_mf.set_values(model.materials)
    plane_orientations_mf = MeshFunction("double", mesh, mesh.topology().dim())
    plane_orientations_mf.set_values(model.plane_orientations)
    fiber_orientations_mf = MeshFunction("double", mesh, mesh.topology().dim())
    fiber_orientations_mf.set_values(model.fiber_orientations)

    # Build material library
    matLibrary = []
    for mat in model.matlibrary:
        if mat.type == "orthotropic":
            prop = np.zeros((3,3))
            prop[0,0] = mat.e_xx
            prop[0,1] = mat.e_yy
            prop[0,2] = mat.e_zz
            prop[1,0] = mat.g_yz
            prop[1,1] = mat.g_xz
            prop[1,2] = mat.g_xy
            prop[2,0] = mat.nu_zy
            prop[2,1] = mat.nu_zx
            prop[2,2] = mat.nu_xy
            mat_obj = material.OrthotropicMaterial(prop, mat.rho)
        elif mat.type == "isotropic":
            prop = [mat.e, mat.nu]
            mat_obj = material.IsotropicMaterial(prop, mat.rho)
        matLibrary.append(mat_obj)

    return mesh, matLibrary, materials_mf, plane_orientations_mf, fiber_orientations_mf
