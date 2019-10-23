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

from anba4.voight_notation import stressVectorToStressTensor, \
    stressTensorToStressVector, stressTensorToParaviewStressVector, \
    strainVectorToStrainTensor, strainTensorToStrainVector
from anba4 import material

class anbax():
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
        self.RotatedStress_modulus = CompiledExpression(
            material.RotatedStressElasticModulus(
                self.matLibrary,
                self.materials,
                self.plane_orientations,
                self.fiber_orientations
            ),
            degree=0
        )
        self.density = CompiledExpression(
            material.MaterialDensity(
                self.matLibrary,
                self.materials
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
        (self.RV3F, self.RV3M) = TestFunctions(R3R3)
        (self.RT3F, self.RT3M) = TrialFunctions(R3R3)

        #STRESS_ELEMENT = TensorElement("DG", self.mesh.ufl_cell(), 0, (3, 3))
        STRESS_ELEMENT = VectorElement("DG", self.mesh.ufl_cell(), 0, 6)
        STRESS_FS = FunctionSpace(self.mesh, STRESS_ELEMENT)
        self.STRESS = Function(STRESS_FS, name = "stress tensor")

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

        # fill chains
        for i in range(4):
            for k in range(2):
                self.chains[i].append(Function(UF3R4))
        for i in range(2,4):
            for k in range(2):
                self.chains[i].append(Function(UF3R4))

        # initialize constant chains
        for i in range(4):
            self.chains[i][0].interpolate(self.base_chains_expression[i])
        # keep torsion independent from translation
        for i in [0, 2, 3]:
            k = (self.chains[1][0].vector().inner(self.chains[i][0].vector())) / (self.chains[i][0].vector().inner(self.chains[i][0].vector()))
            self.chains[1][0].vector()[:] -= k * self.chains[i][0].vector()
        self.null_space = VectorSpaceBasis([self.chains[i][0].vector() for i in range(4)])

        # initialize linear chains
        for i in range(2,4):
            self.chains[i][1].interpolate(self.linear_chains_expression[i-2])
            self.null_space.orthogonalize(self.chains[i][1].vector());

        for i in range(4):
            for k in range(2):
                (d, l) = split(self.chains[i][k])
                self.chains_d[i].append(d)
                self.chains_l[i].append(l)

        for i in range(2,4):
            for k in range(2,4):
                (d, l) = split(self.chains[i][k])
                self.chains_d[i].append(d)
                self.chains_l[i].append(l)

    def inertia(self):
        Mf  = dot(self.RV3F, self.RT3F) * self.density[0] * dx
        Mf += dot(self.RV3F, cross(self.pos3d(self.POS), self.RT3M)) * self.density[0] * dx
        Mf += dot(cross(self.pos3d(self.POS), self.RV3F), self.RT3M) * self.density[0] * dx
        Mf += dot(cross(self.pos3d(self.POS), self.RV3M), cross(self.pos3d(self.POS), self.RT3M)) * self.density[0] * dx
        MM = assemble(Mf)
        M = as_backend_type(MM).mat()
        Mass = PETSc.Mat()#.createDense([6, 6])
        #Mass.setUp()
        #Mass.view()
        #M.copy(Mass, PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
        M.convert('dense', Mass)
        return Mass

    def compute(self):
        stress = self.Sigma(self.U, self.UP)
        stress_n = stress[:,2]
        stress_1 = stress[:,0]
        stress_2 = stress[:,1]
        stress_s = as_tensor([[stress_1[0], stress_2[0]],
            [stress_1[1], stress_2[1]],
            [stress_1[2], stress_2[2]]])

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

        S = dot(stress_n, self.RV3F) * dx + dot(cross(self.pos3d(self.POS), stress_n), self.RV3M) * dx
        L_f = derivative(S, self.UP, self.UT)
        L = assemble(L_f)
        R_f = derivative(S, self.U, self.UT)
        R = assemble(R_f)

        resk = []
        # solve E d1 = -H d0
        for i in range(2):
            solve(E, self.chains[i][1].vector(), -(H*self.chains[i][0].vector()))
#             for k in range(4):
#                 c = (self.chains[i][1].vector().inner(self.chains[k][0].vector())) / (self.chains[k][0].vector().inner(self.chains[k][0].vector()))
#                 self.chains[i][1].vector()[:] -= c * self.chains[k][0].vector()
#             #(ou1, ol1) = chains[i][1].split(True)
            res = -(H*self.chains[i][1].vector())+(M*self.chains[i][0].vector())
            resk.append(res.inner(self.chains[i][0].vector()))


        # solve E d2 = M d0 - H d1
        for i in [2, 3]:
#             rhs = -(H*self.chains[i][1].vector())+(M*self.chains[i][0].vector())
#             for j in range(2):
#                 mul = rhs.inner(self.chains[j][0].vector())
#                 self.chains[i][0].vector()[:] -= self.chains[j][0].vector() * (mul / resk[j])
#                 self.chains[i][1].vector()[:] -= self.chains[j][1].vector() * (mul / resk[j])
            solve(E, self.chains[i][2].vector(), -(H*self.chains[i][1].vector())+(M*self.chains[i][0].vector()))
            for k in range(4):
                c = (self.chains[i][2].vector().inner(self.chains[k][0].vector())) / (self.chains[k][0].vector().inner(self.chains[k][0].vector()))
                self.chains[i][2].vector()[:] -= c * self.chains[k][0].vector()

#             rhs = -(H*self.chains[i][2].vector())+(M*self.chains[i][1].vector())
#             for j in range(2):
#                 mul = rhs.inner(self.chains[j][0].vector())
#                 self.chains[i][1].vector()[:] -= self.chains[j][0].vector() * (mul / resk[j])
#                 self.chains[i][2].vector()[:] -= self.chains[j][1].vector() * (mul / resk[j])
            solve(E, self.chains[i][3].vector(), -(H*self.chains[i][2].vector())+(M*self.chains[i][1].vector()))
            for k in range(4):
                c = (self.chains[i][3].vector().inner(self.chains[k][0].vector())) / (self.chains[k][0].vector().inner(self.chains[k][0].vector()))
                self.chains[i][3].vector()[:] -= c * self.chains[k][0].vector()

        # solve E d3 = M d1 - H d2
        for i in range(4):
            for k in range(len(self.chains[i])//2, len(self.chains[i])):
                (d0p, l0p) = self.chains[i][k].split(True)

        # len=2 range(1,0,-1) -> k = 1 len()-1-k len()-k
        # len=4 range(2,0,-1) -> k = 2 len()-1-k=1 len()-k=2
        #                        k = 1 len()-1-k=2 len()-k=3

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
        S.setUp()

        self.B = PETSc.Mat().createDense([6, 6])
        self.B.setUp()

        self.G = PETSc.Mat().createDense([6, 6])
        self.G.setUp()

        g = PETSc.Vec().createMPI(6)
        b = PETSc.Vec().createMPI(6)

        self.Stiff = PETSc.Mat().createDense([6, 6])
        self.Stiff.setUp()



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
                self.B.setValues(row, range(6), as_backend_type(L * self.chains[i][ll-1-k].vector() + R * self.chains[i][ll-k].vector()).vec())

        S.assemble()
        self.B.assemble()

        ksp = PETSc.KSP()
        ksp.create()
        ksp.setOperators(S)
        ksp.setType(ksp.Type.PREONLY)   # Just use the preconditioner without a Krylov method
        pc = ksp.getPC()                # Preconditioner
        pc.setType(pc.Type.LU)          # Use a direct solve


        for i in range(6):
            ksp.solve(self.B.getColumnVector(i), g)
            self.G.setValues(range(6), i, g)

        self.G.assemble()

        self.G.transposeMatMult(S, self.B)
        self.B.matMult(self.G, self.Stiff)
        
        return self.Stiff

    def Sigma(self, u, up):
        "Return second Piola–Kirchhoff stress tensor."
        return self.sigma_helper(u, up, self.modulus)

    def RotatedSigma(self, u, up):
        "Return second Piola–Kirchhoff stress tensor."
        return self.sigma_helper(u, up, self.RotatedStress_modulus)

    def sigma_helper(self, u, up, mod):
        "Return second Piola–Kirchhoff stress tensor."
        et = self.epsilon(u, up)
        ev = strainTensorToStrainVector(et)
#         elasticMatrix = self.modulus
        elasticMatrix = as_matrix(((mod[0],mod[1],mod[2],mod[3],mod[4],mod[5]),\
                                   (mod[6],mod[7],mod[8],mod[9],mod[10],mod[11]),\
                                   (mod[12],mod[13],mod[14],mod[15],mod[16],mod[17]),\
                                   (mod[18],mod[19],mod[20],mod[21],mod[22],mod[23]),\
                                   (mod[24],mod[25],mod[26],mod[27],mod[28],mod[29]),\
                                   (mod[30],mod[31],mod[32],mod[33],mod[34],mod[35])))
        sv = elasticMatrix * ev
        st = stressVectorToStressTensor(sv)
        return st

    def epsilon(self, u, up):
        "Return symmetric 3D infinitesimal strain tensor."
        g3 = self.grad3d(u, up)
        return 0.5*(g3.T + g3)

    def grad3d(self, u, up):
        "Return 3d gradient."
        g = grad(u)
        return as_tensor([[g[0,0], g[0,1], up[0]],[g[1,0], g[1,1], up[1]],[g[2,0], g[2,1], up[2]]])

    def pos3d(self, POS):
        "Return node coordinates Vector."
        return as_vector([POS[0], POS[1], 0.])

    def local_project(self, v, V, u=None):
        """Element-wise projection using LocalSolver"""
        dv = TrialFunction(V)
        v_ = TestFunction(V)
        a_proj = inner(dv, v_)*dx
        b_proj = inner(v, v_)*dx
        solver = LocalSolver(a_proj, b_proj)
        solver.factorize()
        if u is None:
            u = Function(V)
            solver.solve_local_rhs(u)
            return u
        else:
            solver.solve_local_rhs(u)
            return

    def stress_field(self, force, moment, reference = "local", voigt_convention = "anba"):
        if reference == "local":
            stress_comp = self.RotatedSigma
        elif reference == "global":
            stress_comp = self.Sigma
        else:
            raise ValueError('reference argument should be equal to either to\"local\" or to "global", got \"' + reference + '\" instead')
        if voigt_convention == "anba":
            vector_conversion = stressTensorToStressVector
        elif voigt_convention == "paraview":
            vector_conversion = stressTensorToParaviewStressVector
        else:
            raise ValueError('voigt_convention argument should be equal to either to\"anba\" or to "paraview", got \"' + voigt_convention + '\" instead')

        eigensol_magnitudes = PETSc.Vec().createMPI(6)

        AzInt = PETSc.Vec().createMPI(6)

        AzInt.setValues(range(3), force)
        AzInt.setValues(range(3, 6), moment)
        
        ksp = PETSc.KSP()
        ksp.create()
        ksp.setOperators(self.B)
        ksp.setType(ksp.Type.PREONLY)   # Just use the preconditioner without a Krylov method
        pc = ksp.getPC()                # Preconditioner
        pc.setType(pc.Type.LU)          # Use a direct solve
        
        ksp.solve(AzInt, eigensol_magnitudes)
        
        self.UL.vector()[:] = 0.
        self.ULP.vector()[:] = 0.
        row = -1
        for i in range(4):
            ll = len(self.chains[i])
            for k in range(ll//2, 0, -1):
                row = row + 1
                self.UL.vector()[:] += self.chains[i][ll-k].vector() * eigensol_magnitudes[row]
                self.ULP.vector()[:] += self.chains[i][ll-1-k].vector() * eigensol_magnitudes[row]
        self.local_project(vector_conversion(stress_comp(self.U, self.UP)), self.STRESS.ufl_function_space(), self.STRESS)
#        self.local_project(stress_comp(self.U, self.UP), self.STRESS.ufl_function_space(), self.STRESS)
