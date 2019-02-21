#
# Copyright (C) 2018 Marco Morandini
# Copyright (C) 2018 Wenguo Zhu
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
import os

from dolfin import compile_cpp_code

pwd = os.path.dirname(os.path.abspath(__file__))
with open(pwd + "/material.cpp", "r") as f:
    material_code = f.read()
#material = compile_cpp_code(material_code, cppargs=('-g', '-O0'))
material_cpp = compile_cpp_code(material_code)
