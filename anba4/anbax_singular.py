#
# Copyright (C) 2018 Marco Morandini
#
#----------------------------------------------------------------------
#
#    This file is part of Anba.
#
#    Anba is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Hanba is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Anba.  If not, see <https://www.gnu.org/licenses/>.
#
#----------------------------------------------------------------------
#

from dolfin import *
from petsc4py import PETSc

import numpy as np

from anba4.voight_notation import stressVectorToStressTensor, stressTensorToStressVector, strainVectorToStrainTensor, strainTensorToStrainVector
from anba4 import material

class anbax_singular():
    def __init__(self, mesh, degree, matLibrary, materials, plane_orientations, fiber_orientations, scaling_constraint = 1.):
        self.mesh = mesh
        self.degree = degree
        self.matLibrary = matLibrary
        self.materials = materials
        self.fiber_orientations = fiber_orientations
        self.plane_orientations = plane_orientations
        self.modulus = CompiledExpression(
            material.ElasticModulus(
                self.matLibrary,
                self.materials,
                self.plane_orientations,
                self.fiber_orientations
            ),
            degree=0
        )
        self.scaling_constraint = scaling_constraint

        # Define function on space.
        UF3_ELEMENT = VectorElement("CG", self.mesh.ufl_cell(), self.degree, 3)
        UF3 = FunctionSpace(self.mesh, UF3_ELEMENT)

        #Lagrange multipliers needed to compute the stress resultants and moment resultants
        R3_ELEMENT = VectorElement("R", self.mesh.ufl_cell(), 0, 3)
        R3 = FunctionSpace(self.mesh, R3_ELEMENT)
        R3R3_ELEMENT = MixedElement(R3_ELEMENT, R3_ELEMENT)
        R3R3 = FunctionSpace(self.mesh, R3R3_ELEMENT)
        (self.RT3F, self.RT3M) = TestFunctions(R3R3)
        self.RT3 = TestFunction(R3)


        self.b = Function(UF3)
        self.U = Function(UF3)
        self.UP = Function(UF3)
        self.UV = TestFunction(UF3)
        self.UT = TrialFunction(UF3)

        self.POS = MeshCoordinates(self.mesh)

        self.base_chains_expression = []
        self.linear_chains_expression = []
        self.Torsion = Expression(("-x[1]", "x[0]", "0."), element = UF3.ufl_element())
        self.Flex_y = Expression(("0.", "0.", "-x[0]"), element = UF3.ufl_element())
        self.Flex_x = Expression(("0.", "0.", "-x[1]"), element = UF3.ufl_element())

        self.base_chains_expression.append(Constant((0., 0., 1.)))
        self.base_chains_expression.append(self.Torsion)
        self.base_chains_expression.append(Constant((1., 0., 0.)))
        self.base_chains_expression.append(Constant((0., 1., 0.)))
        self.linear_chains_expression.append(self.Flex_y)
        self.linear_chains_expression.append(self.Flex_x)

        self.chains = [[], [], [], []]
        
        # fill chains
        for i in range(4):
            for k in range(2):
                self.chains[i].append(Function(UF3))
        for i in range(2,4):
            for k in range(2):
                self.chains[i].append(Function(UF3))

        # initialize constant chains
        for i in range(4):
            self.chains[i][0].interpolate(self.base_chains_expression[i])
        # keep torsion independent from translation
#         for i in [0, 2, 3]:
#             k = (self.chains[1][0].vector().inner(self.chains[i][0].vector())) / (self.chains[i][0].vector().inner(self.chains[i][0].vector()))
#             self.chains[1][0].vector()[:] -= k * self.chains[i][0].vector()
        self.null_space = VectorSpaceBasis([self.chains[i][0].vector() for i in range(4)])

        # initialize linear chains
        for i in range(2,4):
            self.chains[i][1].interpolate(self.linear_chains_expression[i-2])
#             self.null_space.orthogonalize(self.chains[i][1].vector());
#            for k in range(4):
#	            c = (self.chains[i][1].vector().inner(self.chains[k][0].vector())) / (self.chains[k][0].vector().inner(self.chains[k][0].vector()))
#	            self.chains[i][1].vector()[:] -= c * self.chains[k][0].vector()


    def compute(self):
        stress = self.sigma(self.U, self.UP)
        stress_n = stress[:,2]
        stress_1 = stress[:,0]
        stress_2 = stress[:,1]
        stress_s = as_tensor([[stress_1[0], stress_2[0]],
            [stress_1[1], stress_2[1]],
            [stress_1[2], stress_2[2]]])

        eps = self.epsilon(self.U, self.UP)

        ES = derivative(stress, self.U, self.UT)
        ES_t = derivative(stress_s, self.U, self.UT)
        ES_n = derivative(stress_s, self.UP, self.UT)
        En_t = derivative(stress_n, self.U, self.UT)
        En_n = derivative(stress_n, self.UP, self.UT)

        Mf = inner(self.UV, En_n) * dx
        M = assemble(Mf)

        Cf = inner(grad(self.UV), ES_n) * dx
        C = assemble(Cf)
        Hf = (inner(grad(self.UV), ES_n) - inner(self.UV, En_t)) * dx
        H = assemble(Hf)


        #the four initial solutions

        Ef = inner(grad(self.UV), ES_t) * dx
        E = assemble(Ef)

        solver = PETScKrylovSolver("cg")
        solver.parameters["relative_tolerance"] = 1.E-10
#        solver.parameters["absolute_tolerance"] = 1.E-16
#        solver.parameters["convergence_norm_type"] = "natural"
#        solver.parameters["monitor_convergence"] = True
        solver.set_operator(E)
        as_backend_type(E).set_nullspace(self.null_space)
        ptn = PETSc.NullSpace([as_backend_type(self.chains[i][0].vector()).vec() for i in range(4)])
        as_backend_type(E).mat().setTransposeNullSpace(ptn)

        S = dot(stress_n, self.RT3F) * dx + dot(cross(self.pos3d(self.POS), stress_n), self.RT3M) * dx
        L_f = derivative(S, self.UP, self.UT)
        L = assemble(L_f)
        R_f = derivative(S, self.U, self.UT)
        R = assemble(R_f)
        
        
        maxres = 0.
        for i in range(4):
            tmp = E*self.chains[i][0].vector()
            maxres = max(maxres, sqrt(tmp.inner(tmp)))
        for i in [2, 3]:
            tmp = -(H*self.chains[i][0].vector()) -(E * self.chains[i][1].vector())
            maxres = max(maxres, sqrt(tmp.inner(tmp)))
        
        if maxres > 1.E-16:
            scaling_factor = 1.E-16 / maxres;
        else:
            scaling_factor = 1.

        for i in range(4):
            self.chains[i][0].vector()[:] = self.chains[i][0].vector() * scaling_factor
        for i in [2, 3]:
            self.chains[i][1].vector()[:] = self.chains[i][1].vector() * scaling_factor
        for i in range(4):
            tmp = E*self.chains[i][0].vector()
            maxres = max(maxres, sqrt(tmp.inner(tmp)))
        for i in [2, 3]:
            tmp = -(H*self.chains[i][0].vector()) -(E * self.chains[i][1].vector())
            maxres = max(maxres, sqrt(tmp.inner(tmp)))

        resk = []
        # solve E d1 = -H d0
        for i in range(2):
            self.b.vector()[:] = -(H*self.chains[i][0].vector())
            self.null_space.orthogonalize(self.b.vector());
            print('Solving ',i)
            solver.solve(E, self.chains[i][1].vector(), self.b.vector())
#             self.null_space.orthogonalize(self.chains[i][1].vector());
            for k in range(4):
                c = (self.chains[i][1].vector().inner(self.chains[k][0].vector())) / (self.chains[k][0].vector().inner(self.chains[k][0].vector()))
                self.chains[i][1].vector()[:] -= c * self.chains[k][0].vector()
            res = -(H*self.chains[i][1].vector())+(M*self.chains[i][0].vector())
            resk.append(res.inner(self.chains[i][0].vector()))

        # solve E d2 = M d0 - H d1
        for i in [2, 3]:
            self.b.vector()[:] = -(H*self.chains[i][1].vector())+(M*self.chains[i][0].vector())
#             self.null_space.orthogonalize(self.b.vector());
            print('Solving ',i,0)
            solver.solve(E, self.chains[i][2].vector(), self.b.vector())



        a = np.zeros((2,2))
        b = np.zeros((2,1))
        for i in [2, 3]:
            self.b.vector()[:] = -(H*self.chains[i][2].vector())+(M*self.chains[i][1].vector())
            for k in range(4):
                print(self.b.vector().inner(self.chains[k][0].vector()))        
        for i in [2, 3]:
            self.b.vector()[:] = -(H*self.chains[i][2].vector())+(M*self.chains[i][1].vector())
            for k in range(2):
                b[k] = self.b.vector().inner(self.chains[k][0].vector())
                for ii in range(2):
                    #a[ii, k] = (-(H*self.chains[ii][0].vector())).inner(self.chains[k][0].vector()) / normk
                    a[k, ii] = (-(H*self.chains[ii][1].vector())+(M*self.chains[ii][0].vector())).inner(self.chains[k][0].vector())
            x = np.linalg.solve(a, b)
            for ii in range(2):
                self.chains[i][2].vector()[:] -= x[ii] * self.chains[ii][1].vector()
                self.chains[i][1].vector()[:] -= x[ii] * self.chains[ii][0].vector()
        #asd
        
        for i in [2, 3]:
            print('Solving ',i,1)
            self.b.vector()[:] = -(H*self.chains[i][2].vector())+(M*self.chains[i][1].vector())
            solver.solve(E, self.chains[i][3].vector(), self.b.vector())
#             self.null_space.orthogonalize(self.chains[i][3].vector());
            for k in range(4):
                c = (self.chains[i][3].vector().inner(self.chains[k][0].vector())) / (self.chains[k][0].vector().inner(self.chains[k][0].vector()))
                self.chains[i][3].vector()[:] -= c * self.chains[k][0].vector()

        # solve E d3 = M d1 - H d2
        for i in range(4):
            print("\nChain "+ str(i) +":")
            for k in range(len(self.chains[i])//2, len(self.chains[i])):
                print("(row" + str(k) + ") = ", assemble(inner(self.chains[i][k], self.chains[i][k]) * dx))

        print("")
        for i in range(4):
            ll = len(self.chains[i])
            for k in range(ll//2, 0, -1):
                res =  E * self.chains[i][ll-k].vector() + H * self.chains[i][ll-1-k].vector()
                if ll-1-k > 0:
                    res -= M * self.chains[i][ll-2-k].vector()
                res = as_backend_type(res).vec()
                print('residual chain',i,'order',ll-k, res.dot(res))
        print("")


        row1_col = []
        row2_col = []
        for i in range(6):
            row1_col.append(as_backend_type(self.chains[0][0].vector().copy()).vec())
            row2_col.append(as_backend_type(self.chains[0][0].vector().copy()).vec())

        M_p = as_backend_type(M).mat()
        C_p = as_backend_type(C).mat()
        E_p = as_backend_type(E).mat()
        S = PETSc.Mat().createDense([6, 6])
        S.setPreallocationDense(None)

        B = PETSc.Mat().createDense([6, 6])
        B.setPreallocationDense(None)

        G = PETSc.Mat().createDense([6, 6])
        G.setPreallocationDense(None)

        g = PETSc.Vec().createMPI(6)
        b = PETSc.Vec().createMPI(6)

        Stiff = PETSc.Mat().createDense([6, 6])
        Stiff.setPreallocationDense(None)



        col = -1
        for i in range(4):
            ll = len(self.chains[i])
            for k in range(ll//2, 0, -1):
                col = col + 1
                M_p.mult(as_backend_type(self.chains[i][ll-1-k].vector()).vec(), row1_col[col])
                C_p.multTransposeAdd(as_backend_type(self.chains[i][ll-k].vector()).vec(), row1_col[col], row1_col[col])
                C_p.mult(as_backend_type(self.chains[i][ll-1-k].vector()).vec(), row2_col[col])
                E_p.multAdd(as_backend_type(self.chains[i][ll-k].vector()).vec(), row2_col[col], row2_col[col])


        #print dir(PETSc.Vec)

        row = -1
        for i in range(4):
            ll = len(self.chains[i])
            for k in range(ll//2, 0, -1):
                row = row + 1
                for c in range(6):
                    S.setValues(row, c, as_backend_type(self.chains[i][ll-1-k].vector()).vec().dot(row1_col[c]) +
                        as_backend_type(self.chains[i][ll-k].vector()).vec().dot(row2_col[c]))
                B.setValues(row, range(6), as_backend_type(L * self.chains[i][ll-1-k].vector() + R * self.chains[i][ll-k].vector()).vec())

        S.assemble()
        B.assemble()

        ksp = PETSc.KSP()
        ksp.create()
        ksp.setOperators(S)


        for i in range(6):
            ksp.solve(B.getColumnVector(i), g)
            G.setValues(range(6), i, g)

        G.assemble()

        G.transposeMatMult(S, B)
        B.matMult(G, Stiff)
        
        return Stiff
        
    def sigma(self, u, up):
        "Return second Piola–Kirchhoff stress tensor."
        et = self.epsilon(u, up)
        ev = strainTensorToStrainVector(et)
        elasticMatrix = as_matrix(((self.modulus[0],self.modulus[1],self.modulus[2],self.modulus[3],self.modulus[4],self.modulus[5]),\
                                   (self.modulus[6],self.modulus[7],self.modulus[8],self.modulus[9],self.modulus[10],self.modulus[11]),\
                                   (self.modulus[12],self.modulus[13],self.modulus[14],self.modulus[15],self.modulus[16],self.modulus[17]),\
                                   (self.modulus[18],self.modulus[19],self.modulus[20],self.modulus[21],self.modulus[22],self.modulus[23]),\
                                   (self.modulus[24],self.modulus[25],self.modulus[26],self.modulus[27],self.modulus[28],self.modulus[29]),\
                                   (self.modulus[30],self.modulus[31],self.modulus[32],self.modulus[33],self.modulus[34],self.modulus[35])))
        sv = elasticMatrix * ev
        st = stressVectorToStressTensor(sv)
        return st

    def epsilon(self, u, up):
        "Return symmetric 3D infinitesimal strain tensor."
        return 0.5*(self.grad3d(u, up).T + self.grad3d(u, up))

    def grad3d(self, u, up):
        "Return 3d gradient."
        g = grad(u)
        return as_tensor([[g[0,0], g[0,1], up[0]],[g[1,0], g[1,1], up[1]],[g[2,0], g[2,1], up[2]]])

    def pos3d(self, POS):
        "Return node coordinates Vector."
        return as_vector([POS[0], POS[1], 0.])
