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
import time
from petsc4py import PETSc

parameters["form_compiler"]["optimize"] = True
#parameters["reorder_dofs_serial"] = False

#dolfin.set_log_active(True)
#dolfin.set_log_level(1)


E  = Constant(1.)
E_code = """
#include <cstdlib>
class E : public Expression
{
public:

  // Create expression with 1 components
  E() : Expression() {}

  // Function for evaluating expression on each cell
  void eval(Array<double>& values, const Array<double>& x, const ufc::cell& cell) const
  {
    values[0] = rand() / RAND_MAX + 1.;
    //values[0] = x[0] > 0.? 1.:2.;
  }
};
"""
#E  = Expression(cppcode=E_code)
nu = Constant(0.33)

mu    = E / (2.0*(1.0 + nu))
lmbda = E*nu / ((1.0 + nu)*(1.0 - 2.0*nu))

def grad3d(u, up):
	"Return 3d gradient."
	g = grad(u)
	return as_tensor([[g[0,0], g[0,1], up[0]],[g[1,0], g[1,1], up[1]],[g[2,0], g[2,1], up[2]]])

def epsilon(u, up):
	"Return symmetric 3D deformation tensor."
# 	return 0.5*((grad3d(u, up).T+Identity(3)) * (grad3d(u, up)+Identity(3)) - Identity(3)) #
	return 0.5*(grad3d(u, up).T + grad3d(u, up)) #

def pos3d(POS):
	return as_vector([POS[0], POS[1], 0.])

def sigma(u, up):
	"Return stress tensor."
	eps = epsilon(u, up)
	return 2.0*mu*eps + lmbda*tr(eps)*Identity(3)


mesh = UnitSquareMesh(10, 10)
ALE.move(mesh, Constant([-0.5, -0.5]))

UF3_ELEMENT = VectorElement("CG", mesh.ufl_cell(), 2, 3)
UF3 = FunctionSpace(mesh, UF3_ELEMENT)


#Lagrange multipliers needed to compute the stress resultants and moment resultants
R3_ELEMENT = VectorElement("R", mesh.ufl_cell(), 0, 3)
R3 = FunctionSpace(mesh, R3_ELEMENT)
sp = parameters["reorder_dofs_serial"]
parameters["reorder_dofs_serial"] = False
R3R3_ELEMENT = MixedElement(R3_ELEMENT, R3_ELEMENT)
R3R3 = FunctionSpace(mesh, R3R3_ELEMENT)
parameters["reorder_dofs_serial"] = sp
(RT3F, RT3M) = TestFunctions(R3R3)
RT3 = TestFunction(R3)


#Lagrange multipliers needed to impose the BCs
R4_ELEMENT = VectorElement("R", mesh.ufl_cell(), 0, 4)
R4 = FunctionSpace(mesh, R4_ELEMENT)
sp = parameters["reorder_dofs_serial"]
parameters["reorder_dofs_serial"] = False
UF3R4_ELEMENT = MixedElement(UF3_ELEMENT, R4_ELEMENT)
UF3R4 = FunctionSpace(mesh, UF3R4_ELEMENT)
parameters["reorder_dofs_serial"] = sp

U = Function(UF3)
L = Function(R4)

UP = Function(UF3)
LP = Function(R4)

(UV, LV) = TestFunctions(UF3R4)

(UT, LT) = TrialFunctions(UF3R4)
(UTP, LTP) = TrialFunctions(UF3R4)

LTT = TestFunction(R4)

POS = MeshCoordinates(mesh)


stress = sigma(U, UP)
stress_n = stress[:,2]
stress_1 = stress[:,0]
stress_2 = stress[:,1]
stress_s = as_tensor([[stress_1[0], stress_2[0]], 
	[stress_1[1], stress_2[1]], 
	[stress_1[2], stress_2[2]]])

eps = epsilon(U, UP)

ES = derivative(stress, U, UT)
ES_t = derivative(stress_s, U, UT)
ES_n = derivative(stress_s, UP, UTP)
En_t = derivative(stress_n, U, UT)
En_n = derivative(stress_n, UP, UTP)

Mf = inner(UV, En_n) * dx
M = assemble(Mf)


Cf = inner(grad(UV), ES_n) * dx
C = assemble(Cf)
Hf = (inner(grad(UV), ES_n) - inner(UV, En_t)) * dx 
H = assemble(Hf)


#the four initial solutions
base_chains_expression = []
linear_chains_expression = []
Torsion = Expression(("-x[1]", "x[0]", "0.", "0.", "0.", "0.", "0."), element = UF3R4.ufl_element())
Flex_y = Expression(("0.", "0.", "-x[0]", "0.", "0.", "0.", "0."), element = UF3R4.ufl_element())
Flex_x = Expression(("0.", "0.", "-x[1]", "0.", "0.", "0.", "0."), element = UF3R4.ufl_element())

base_chains_expression.append(Constant((0., 0., 1., 0., 0., 0., 0.)))
base_chains_expression.append(Torsion)
base_chains_expression.append(Constant((1., 0., 0., 0., 0., 0., 0.)))
base_chains_expression.append(Constant((0., 1., 0., 0., 0., 0., 0.)))
linear_chains_expression.append(Flex_y)
linear_chains_expression.append(Flex_x)

chains = [[], [], [], []]
chains_d = [[], [], [], []]
chains_l = [[], [], [], []]
for i in range(4):
	for k in range(2):
		chains[i].append(Function(UF3R4))
		(d, l) = split(chains[i][len(chains[i])-1])
		chains_d[i].append(d)
		chains_l[i].append(l)
	chains[i][0].interpolate(base_chains_expression[i])
for k in [2, 3]:
	c = (chains[1][0].vector().inner(chains[k][0].vector())) / (chains[k][0].vector().inner(chains[k][0].vector()))
	chains[1][0].vector()[:] -= c * chains[k][0].vector()

for i in [2, 3]:
	for k in range(2):
		chains[i].append(Function(UF3R4))
		(d, l) = split(chains[i][len(chains[i])-1])
		chains_d[i].append(d)
		chains_l[i].append(l)
	chains[i][1].interpolate(linear_chains_expression[i-2])
	c = (chains[i][1].vector().inner(chains[0][0].vector())) / (chains[0][0].vector().inner(chains[0][0].vector()))
	chains[i][1].vector()[:] -= c * chains[0][0].vector()


#torsion should not include components along inplane translation
for i in [0, 2, 3]:
	k = (chains[1][0].vector().inner(chains[i][0].vector())) / (chains[i][0].vector().inner(chains[i][0].vector()))
	chains[1][0].vector()[:] -= k * chains[i][0].vector()

Ef = inner(grad(UV), ES_t) * dx
Ef += (LV[0] * UT[0] + LV[1] * UT[1] + LV[2] * UT[2]) * dx
Ef += LV[3] * dot(UT, chains_d[1][0]) * dx
Ef += (UV[0] * LT[0] + UV[1] * LT[1] + UV[2] * LT[2]) * dx
Ef += LT[3] * dot(UV, chains_d[1][0]) * dx
E = assemble(Ef)

S = dot(stress_n, RT3F) * dx + dot(cross(pos3d(POS), stress_n), RT3M) * dx
L_f = derivative(S, UP, UTP)
L = assemble(L_f)
R_f = derivative(S, U, UT)
R = assemble(R_f)
#print as_backend_type(M).mat().view()

resk = []
# solve E d1 = -H d0
for i in range(2):
	solve(E, chains[i][1].vector(), -(H*chains[i][0].vector()))
	#(ou1, ol1) = chains[i][1].split(True)
	res = -(H*chains[i][1].vector())+(M*chains[i][0].vector())
	resk.append(res.inner(chains[i][0].vector()))
		

# solve E d2 = M d0 - H d1
for i in [2, 3]:
# 	rhs = -(H*chains[i][1].vector())+(M*chains[i][0].vector())
# 	for j in range(2):
# 		mul = rhs.inner(chains[j][0].vector())
# 		chains[i][0].vector()[:] -= chains[j][0].vector() * (mul / resk[j])
# 		chains[i][1].vector()[:] -= chains[j][1].vector() * (mul / resk[j])
	solve(E, chains[i][2].vector(), -(H*chains[i][1].vector())+(M*chains[i][0].vector()))

	rhs = -(H*chains[i][2].vector())+(M*chains[i][1].vector())
	for j in range(2):
		mul = rhs.inner(chains[j][0].vector())
		chains[i][1].vector()[:] -= chains[j][0].vector() * (mul / resk[j])
		chains[i][2].vector()[:] -= chains[j][1].vector() * (mul / resk[j])
	solve(E, chains[i][3].vector(), -(H*chains[i][2].vector())+(M*chains[i][1].vector()))

# solve E d3 = M d1 - H d2
for i in range(4):
	print("\nChain "+ str(i) +":")
	for k in range(len(chains[i])//2, len(chains[i])):
		print("(d" + str(k) + ", d" + str(k)+ ") = ", assemble(inner(chains_d[i][k], chains_d[i][k]) * dx))
		print("(l" + str(k) + ", l" + str(k)+ ") = ", assemble(inner(chains_l[i][k], chains_l[i][k]) * dx))
		(d0p, l0p) = chains[i][k].split(True)
		#plot(d0p, title="d" + str(k))
		#print(X3DOM().html(d0p))
	#interactive()
	
# len=2 range(1,0,-1) -> k = 1 len()-1-k len()-k
# len=4 range(2,0,-1) -> k = 2 len()-1-k=1 len()-k=2
#                        k = 1 len()-1-k=2 len()-k=3

print("")
for i in range(4):
	ll = len(chains[i])
	for k in range(ll//2, 0, -1):
		res =  E * chains[i][ll-k].vector() + H * chains[i][ll-1-k].vector()
		if ll-1-k > 0:
			res -= M * chains[i][ll-2-k].vector()
		res = as_backend_type(res).vec()
		print('residual chain',i,'order',ll , res.dot(res))
print("")


row1_col = []
row2_col = []
for i in range(6):
	row1_col.append(as_backend_type(chains[0][0].vector().copy()).vec())
	row2_col.append(as_backend_type(chains[0][0].vector().copy()).vec())

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
	ll = len(chains[i])
	for k in range(ll//2, 0, -1):
		col = col + 1
		M_p.mult(as_backend_type(chains[i][ll-1-k].vector()).vec(), row1_col[col])
		C_p.multTransposeAdd(as_backend_type(chains[i][ll-k].vector()).vec(), row1_col[col], row1_col[col])
		C_p.mult(as_backend_type(chains[i][ll-1-k].vector()).vec(), row2_col[col])
		E_p.multAdd(as_backend_type(chains[i][ll-k].vector()).vec(), row2_col[col], row2_col[col])


#print dir(PETSc.Vec)

row = -1
for i in range(4):
	ll = len(chains[i])
	for k in range(ll//2, 0, -1):
		row = row + 1
		for c in range(6):
			S.setValues(row, c, as_backend_type(chains[i][ll-1-k].vector()).vec().dot(row1_col[c]) + 
				as_backend_type(chains[i][ll-k].vector()).vec().dot(row2_col[c]))
		B.setValues(row, range(6), as_backend_type(L * chains[i][ll-1-k].vector() + R * chains[i][ll-k].vector()).vec())
	
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
Stiff.view()
