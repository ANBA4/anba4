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

from voight_notation import stressVectorToStressTensor, stressTensorToStressVector, strainVectorToStrainTensor, strainTensorToStrainVector
from material import material

class anbax():
    def __init__(self, mesh, degree, matLibrary, materials, fiber_orientations, plane_orientations, scaling_constraint = 1.):
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
        sp = parameters["reorder_dofs_serial"]
        parameters["reorder_dofs_serial"] = False
        R3R3_ELEMENT = MixedElement(R3_ELEMENT, R3_ELEMENT)
        R3R3 = FunctionSpace(self.mesh, R3R3_ELEMENT)
        parameters["reorder_dofs_serial"] = sp
        (self.RT3F, self.RT3M) = TestFunctions(R3R3)
        self.RT3 = TestFunction(R3)


        #Lagrange multipliers needed to impose the BCs
        R4_ELEMENT = VectorElement("R", self.mesh.ufl_cell(), 0, 4)
        R4 = FunctionSpace(self.mesh, R4_ELEMENT)
        sp = parameters["reorder_dofs_serial"]
        parameters["reorder_dofs_serial"] = False
        UF3R4_ELEMENT = MixedElement(UF3_ELEMENT, R4_ELEMENT)
        UF3R4 = FunctionSpace(self.mesh, UF3R4_ELEMENT)
        parameters["reorder_dofs_serial"] = sp

        self.UL = Function(UF3R4)
        (self.U, self.L) = split(self.UL)
        self.ULP = Function(UF3R4)
        (self.UP, self.LP) = split(self.ULP)
        self.ULV = TestFunction(UF3R4)
        (self.UV, self.LV) = TestFunctions(UF3R4)
        self.ULT = TrialFunction(UF3R4)
        (self.UT, self.LT) = TrialFunctions(UF3R4)

        self.POS = MeshCoordinates(self.mesh)

        self.base_chains_expression = []
        self.linear_chains_expression = []
        self.Torsion = Expression(("-x[1]", "x[0]", "0.", "0.", "0.", "0.", "0."), element = UF3R4.ufl_element())
        self.Flex_y = Expression(("0.", "0.", "-x[0]", "0.", "0.", "0.", "0."), element = UF3R4.ufl_element())
        self.Flex_x = Expression(("0.", "0.", "-x[1]", "0.", "0.", "0.", "0."), element = UF3R4.ufl_element())

        self.base_chains_expression.append(Constant((0., 0., 1., 0., 0., 0., 0.)))
        self.base_chains_expression.append(self.Torsion)
        self.base_chains_expression.append(Constant((1., 0., 0., 0., 0., 0., 0.)))
        self.base_chains_expression.append(Constant((0., 1., 0., 0., 0., 0., 0.)))
        self.linear_chains_expression.append(self.Flex_y)
        self.linear_chains_expression.append(self.Flex_x)

        self.chains = [[], [], [], []]
        self.chains_d = [[], [], [], []]
        self.chains_l = [[], [], [], []]
        for i in range(4):
	        for k in range(2):
		        self.chains[i].append(Function(UF3R4))
		        (d, l) = split(self.chains[i][len(self.chains[i])-1])
		        self.chains_d[i].append(d)
		        self.chains_l[i].append(l)
	        self.chains[i][0].interpolate(self.base_chains_expression[i])
        for k in [2, 3]:
	        c = (self.chains[1][0].vector().inner(self.chains[k][0].vector())) / (self.chains[k][0].vector().inner(self.chains[k][0].vector()))
	        self.chains[1][0].vector()[:] -= c * self.chains[k][0].vector()

        for i in range(2,4):
            for k in range(2):
                self.chains[i].append(Function(UF3R4))
                (d, l) = split(self.chains[i][len(self.chains[i])-1])
                self.chains_d[i].append(d)
                self.chains_l[i].append(l)
            self.chains[i][1].interpolate(self.linear_chains_expression[i-2])
            c = (self.chains[i][1].vector().inner(self.chains[0][0].vector())) / (self.chains[0][0].vector().inner(self.chains[0][0].vector()))
            self.chains[i][1].vector()[:] -= c * self.chains[0][0].vector()


        #torsion should not include components along inplane translation
        for i in [0, 2, 3]:
	        k = (self.chains[1][0].vector().inner(self.chains[i][0].vector())) / (self.chains[i][0].vector().inner(self.chains[i][0].vector()))
	        self.chains[1][0].vector()[:] -= k * self.chains[i][0].vector()

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

        Escal = Constant(self.scaling_constraint)
        Ef = inner(grad(self.UV), ES_t) * dx
        Ef += (self.LV[0] * self.UT[0] + self.LV[1] * self.UT[1] + self.LV[2] * self.UT[2]) * Escal * dx
        Ef += self.LV[3] * dot(self.UT, self.chains_d[1][0]) * Escal * dx
        Ef += (self.UV[0] * self.LT[0] + self.UV[1] * self.LT[1] + self.UV[2] * self.LT[2]) * Escal * dx
        Ef += self.LT[3] * dot(self.UV, self.chains_d[1][0]) * Escal * dx
        E = assemble(Ef)

        S = dot(stress_n, self.RT3F) * dx + dot(cross(self.pos3d(self.POS), stress_n), self.RT3M) * dx
        L_f = derivative(S, self.UP, self.UT)
        L = assemble(L_f)
        R_f = derivative(S, self.U, self.UT)
        R = assemble(R_f)
        #print as_backend_type(M).mat().view()

        resk = []
        # solve E d1 = -H d0
        for i in range(2):
	        solve(E, self.chains[i][1].vector(), -(H*self.chains[i][0].vector()))
	        #(ou1, ol1) = chains[i][1].split(True)
	        res = -(H*self.chains[i][1].vector())+(M*self.chains[i][0].vector())
	        resk.append(res.inner(self.chains[i][0].vector()))


        # solve E d2 = M d0 - H d1
        for i in [2, 3]:
        # 	rhs = -(H*chains[i][1].vector())+(M*chains[i][0].vector())
        # 	for j in range(2):
        # 		mul = rhs.inner(chains[j][0].vector())
        # 		chains[i][0].vector()[:] -= chains[j][0].vector() * (mul / resk[j])
        # 		chains[i][1].vector()[:] -= chains[j][1].vector() * (mul / resk[j])
	        solve(E, self.chains[i][2].vector(), -(H*self.chains[i][1].vector())+(M*self.chains[i][0].vector()))

	        rhs = -(H*self.chains[i][2].vector())+(M*self.chains[i][1].vector())
	        for j in range(2):
		        mul = rhs.inner(self.chains[j][0].vector())
		        self.chains[i][1].vector()[:] -= self.chains[j][0].vector() * (mul / resk[j])
		        self.chains[i][2].vector()[:] -= self.chains[j][1].vector() * (mul / resk[j])
	        solve(E, self.chains[i][3].vector(), -(H*self.chains[i][2].vector())+(M*self.chains[i][1].vector()))

        # solve E d3 = M d1 - H d2
        for i in range(4):
	        print("\nChain "+ str(i) +":")
	        for k in range(len(self.chains[i])//2, len(self.chains[i])):
		        print("(d" + str(k) + ", d" + str(k)+ ") = ", assemble(inner(self.chains_d[i][k], self.chains_d[i][k]) * dx))
		        print("(l" + str(k) + ", l" + str(k)+ ") = ", assemble(inner(self.chains_l[i][k], self.chains_l[i][k]) * dx))
		        (d0p, l0p) = self.chains[i][k].split(True)
		        #plot(d0p, title="d" + str(k))
		        #print(X3DOM().html(d0p))
	        #interactive()

        # len=2 range(1,0,-1) -> k = 1 len()-1-k len()-k
        # len=4 range(2,0,-1) -> k = 2 len()-1-k=1 len()-k=2
        #                        k = 1 len()-1-k=2 len()-k=3

        print("")
        for i in range(4):
	        ll = len(self.chains[i])
	        for k in range(ll//2, 0, -1):
		        res =  E * self.chains[i][ll-k].vector() + H * self.chains[i][ll-1-k].vector()
		        if ll-1-k > 0:
			        res -= M * self.chains[i][ll-2-k].vector()
		        res = as_backend_type(res).vec()
		        print('residual chain',i,'order',ll , res.dot(res))
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
