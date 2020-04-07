from dolfin import *
# from dolfin import compile_extension_module
import time
import math
import numpy as np
from petsc4py import PETSc
import os
import matplotlib.pyplot as plt

from anba4 import *

def build_mesh():

    width = 0.48 
    height = 0.24
    thick = 0.05

    w2 = width / 2.
    h2 = height / 2.

    nnodes = 0

    nodex = []
    nodey = []
    nid = []

    nodex.append([])
    nodey.append([])
    nid.append([])

    nodex[0].append(-w2)
    nodey[0].append(-h2)
    nid[0].append(nnodes)
    nnodes = nnodes +1

    nodex[0].append(nodex[0][0] + thick)
    nodey[0].append(-h2)
    nid[0].append(nnodes)
    nnodes = nnodes +1


    nnnodes_x = 11
    l = 0.48 - 2 * thick
    dx = l / (nnnodes_x - 3)
    for i in range(nnnodes_x - 3):
        nodex[0].append(nodex[0][i+1] + dx)
        nodey[0].append(-h2)
        nid[0].append(nnodes)
        nnodes = nnodes +1
    nodex[0].append(w2)
    nodey[0].append(-h2)
    nid[0].append(nnodes)
    nnodes = nnodes +1


    nodex.append([])
    nodey.append([])
    nid.append([])
    nodex[1] = nodex[0]
    for i in range(len(nodey[0])):
        nodey[1].append(nodey[0][i] + 0.05)
        nid[1].append(nnodes)
        nnodes = nnodes +1

    elcon = []
    for i in range(10):
        elcon.append([nid[0][i], nid[0][i+1], nid[1][i], nid[1][i+1]])


    nnodesy = 4
    dy = (height - 2 * thick) / nnodesy
    for i in range(nnodesy-1):
        nodex.append([])
        nodey.append([])
        nid.append([])
        for j in range(2):
            nodex[2+i].append(nodex[1+i][j])
            nodey[2+i].append(nodey[1+i][j] + dy)
            nid[2+i].append(nnodes)
            nnodes = nnodes +1
        elcon.append([nid[1+i][0], nid[1+i][1], nid[2+i][0], nid[2+i][1]])
        for j in range(2,0,-1):
            nodex[2+i].append(nodex[1+i][len(nodex[1+i])-j])
            nodey[2+i].append(nodey[1+i][len(nodex[1+i])-j] + dy)
            nid[2+i].append(nnodes)
            nnodes = nnodes +1
        elcon.append([nid[1+i][len(nodex[1+i])-2], nid[1+i][len(nodex[1+i])-1], nid[2+i][len(nodex[2+i])-2], nid[2+i][len(nodex[2+i])-1]])


    nodex.append(nodex[0])
    nodey.append([])
    nid.append([])
    for i in range(len(nodey[0])):
        nodey[len(nodey)-1].append(h2 - thick)
        nid[len(nodey)-1].append(nnodes)
        nnodes = nnodes +1

    elcon.append([nid[len(nodey)-2][0], nid[len(nodey)-2][1], nid[len(nodey)-1][0], nid[len(nodey)-1][1]])
    elcon.append([nid[len(nodey)-2][len(nodex[len(nodey)-2])-2], nid[len(nodey)-2][len(nodex[len(nodey)-2])-1], nid[len(nodey)-1][len(nodex[len(nodey)-1])-2], nid[len(nodey)-1][len(nodex[len(nodey)-1])-1]])

    nodex.append(nodex[0])
    nodey.append([])
    nid.append([])
    for i in range(len(nodey[0])):
        nodey[len(nodey)-1].append(h2)
        nid[len(nodey)-1].append(nnodes)
        nnodes = nnodes +1
    for i in range(10):
        elcon.append([nid[len(nodey)-2][i], nid[len(nodey)-2][i+1], nid[len(nodey)-1][i], nid[len(nodey)-1][i+1]])


    mesh = Mesh()
    me = MeshEditor()
    me.open(mesh, "quadrilateral", 2, 2)

    me.init_vertices(nnodes)
    for i in range(len(nid)):
        for j in range(len(nid[i])):
            me.add_vertex(nid[i][j], (nodex[i][j], nodey[i][j]))

    me.init_cells(len(elcon))
    for i in range(len(elcon)):
        me.add_cell(i, elcon[i])

    me.close()

    for i in range(len(nid)):
        for j in range(len(nid[i])):
            print(nid[i][j]+1, ', ', nodex[i][j], ', ', nodey[i][j])
    for i in range(len(elcon)):
        print(i+1, ', 200, 2, 2, ', elcon[i][0]+1, ', ', elcon[i][1]+1, ', ', elcon[i][3]+1, ', ', elcon[i][2]+1)

    return mesh

parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize_flags"] = "-O2"
parameters["form_compiler"]["quadrature_degree"] = 2

# Basic material parameters. 9 is needed for orthotropic materials.

e_xx = 9.8e9
e_yy = 9.8e9
e_zz = 1.42e11
g_xy = 4.8e9
g_xz = 6.0e9
g_yz = 6.0e9
nu_xy = 0.34
nu_zx = 0.3
nu_zy = 0.3
#Assmble into material mechanical property Matrix.
matMechanicProp = np.zeros((3,3))
matMechanicProp[0,0] = e_xx
matMechanicProp[0,1] = e_yy
matMechanicProp[0,2] = e_zz
matMechanicProp[1,0] = g_yz
matMechanicProp[1,1] = g_xz
matMechanicProp[1,2] = g_xy
matMechanicProp[2,0] = nu_zy
matMechanicProp[2,1] = nu_zx
matMechanicProp[2,2] = nu_xy

mesh = build_mesh()

# CompiledSubDomain
materials = MeshFunction("size_t", mesh, mesh.topology().dim())
fiber_orientations = MeshFunction("double", mesh, mesh.topology().dim())
plane_orientations = MeshFunction("double", mesh, mesh.topology().dim())
tol = 1e-14


# Rotate mesh.
theta = 23.#0.0
rotation_angle = 0.
materials.set_all(0)
fiber_orientations.set_all(theta)
plane_orientations.set_all(rotation_angle)


# rotate mesh.
rotate = Expression(("x[0] * (cos(rotation_angle)-1.0) - x[1] * sin(rotation_angle)",
    "x[0] * sin(rotation_angle) + x[1] * (cos(rotation_angle)-1.0)"), rotation_angle = rotation_angle * np.pi / 180.0,
    degree = 1)

ALE.move(mesh, rotate)

# Build material property library.
mat1 = material.OrthotropicMaterial(matMechanicProp)

matLibrary = []
matLibrary.append(mat1)


anba = anbax(mesh, 1, matLibrary, materials, plane_orientations, fiber_orientations, 1.E9)
stiff = anba.compute()
stiff.view()
