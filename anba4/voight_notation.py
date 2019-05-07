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

from dolfin import as_tensor, as_vector

def stressVectorToStressTensor(sv):
    "Transform a contracted stress vector to stress tensor."
    # stress vector are written in the order:
    # s11 s22 s33 s23 s13 s12
    return as_tensor([[sv[0], sv[5], sv[4]], [sv[5], sv[1], sv[3]], [sv[4], sv[3], sv[2]]])

def stressTensorToStressVector(st):
    "Transform a stress tensor to stress contracted stress vector."
    return as_vector([st[0,0], st[1,1], st[2,2], st[1,2], st[0,2], st[0,1]])

def stressTensorToParaviewStressVector(st):
    "Transform a stress tensor to stress contracted stress vector."
    return as_vector([st[0,0], st[1,1], st[2,2], st[0,1], st[1,2], st[0,2]])

def strainVectorToStrainTensor(ev):
    "Transform a engeering strain to strain vector."
    return as_tensor([[ev[0], 0.5*ev[5], 0.5*ev[4]], [0.5*ev[5], ev[1], 0.5*ev[3]], [0.5*ev[4],0.5*ev[3],ev[2]]])

def strainTensorToStrainVector(et):
    "Transform a strain tensor to engeering strain."
    return as_vector([et[0,0], et[1,1], et[2,2], 2.0*et[1,2], 2.0*et[0,2], 2.0*et[0,1]])
