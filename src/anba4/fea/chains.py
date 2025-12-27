#
# Copyright (C) 2018 Marco Morandini
#
# ----------------------------------------------------------------------
#
#    This file is part of Anba.
#
#    Anba is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
# ----------------------------------------------------------------------
#

# from dolfin import *
import dolfin
from ..data.data_model import AnbaData


def initialize_chains(data: AnbaData) -> AnbaData:
    """Initialize chains for anisotropic beam analysis."""
    singular = data.input_data.singular
    if singular:
        element = data.fe_functions.UF3.ufl_element()
    else:
        element = data.fe_functions.UF3R4.ufl_element()
    data.chains.base_chains_expression = []
    data.chains.linear_chains_expression = []

    if singular:
        torsion_expr = ("-x[1]", "x[0]", "0.")
        flex_y_expr = ("0.", "0.", "-x[0]")
        flex_x_expr = ("0.", "0.", "-x[1]")
    else:
        torsion_expr = ("-x[1]", "x[0]", "0.", "0.", "0.", "0.", "0.")
        flex_y_expr = ("0.", "0.", "-x[0]", "0.", "0.", "0.", "0.")
        flex_x_expr = ("0.", "0.", "-x[1]", "0.", "0.", "0.", "0.")

    data.chains.Torsion = dolfin.Expression(torsion_expr, element=element)
    data.chains.Flex_y = dolfin.Expression(flex_y_expr, element=element)
    data.chains.Flex_x = dolfin.Expression(flex_x_expr, element=element)

    if singular:
        data.chains.base_chains_expression.append(dolfin.Constant((0.0, 0.0, 1.0)))
        data.chains.base_chains_expression.append(data.chains.Torsion)
        data.chains.base_chains_expression.append(dolfin.Constant((1.0, 0.0, 0.0)))
        data.chains.base_chains_expression.append(dolfin.Constant((0.0, 1.0, 0.0)))
        data.chains.linear_chains_expression.append(data.chains.Flex_y)
        data.chains.linear_chains_expression.append(data.chains.Flex_x)
    else:
        data.chains.base_chains_expression.append(
            dolfin.Constant((0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0))
        )
        data.chains.base_chains_expression.append(data.chains.Torsion)
        data.chains.base_chains_expression.append(
            dolfin.Constant((1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        )
        data.chains.base_chains_expression.append(
            dolfin.Constant((0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        )
        data.chains.linear_chains_expression.append(data.chains.Flex_y)
        data.chains.linear_chains_expression.append(data.chains.Flex_x)

    data.chains.chains = [[], [], [], []]
    data.chains.chains_d = [[], [], [], []]
    data.chains.chains_l = [[], [], [], []] if not singular else None

    # fill chains
    for i in range(4):
        for k in range(2):
            data.chains.chains[i].append(
                dolfin.Function(
                    data.fe_functions.UF3R4 if not singular else data.fe_functions.UF3
                )
            )
    for i in range(2, 4):
        for k in range(2):
            data.chains.chains[i].append(
                dolfin.Function(
                    data.fe_functions.UF3R4 if not singular else data.fe_functions.UF3
                )
            )

    # initialize constant chains
    for i in range(4):
        data.chains.chains[i][0].interpolate(data.chains.base_chains_expression[i])
    # keep torsion independent from translation
    for i in [0, 2, 3]:
        k = (
            data.chains.chains[1][0].vector().inner(data.chains.chains[i][0].vector())
        ) / (data.chains.chains[i][0].vector().inner(data.chains.chains[i][0].vector()))
        data.chains.chains[1][0].vector()[:] -= k * data.chains.chains[i][0].vector()

    # unit norm chains
    tmpnorm = []
    for i in range(4):
        tmpnorm.append(data.chains.chains[i][0].vector().norm("l2"))
        data.chains.chains[i][0].vector()[:] *= 1.0 / tmpnorm[i]
    # null space
    data.output_data.null_space = dolfin.VectorSpaceBasis(
        [data.chains.chains[i][0].vector() for i in range(4)]
    )

    # initialize linear chains
    for i in range(2, 4):
        data.chains.chains[i][1].interpolate(
            data.chains.linear_chains_expression[i - 2]
        )
        data.chains.chains[i][1].vector()[:] *= 1.0 / tmpnorm[i]
        data.output_data.null_space.orthogonalize(data.chains.chains[i][1].vector())
    del tmpnorm

    if not singular:
        for i in range(4):
            for k in range(2):
                (d, l_part) = dolfin.split(data.chains.chains[i][k])
                data.chains.chains_d[i].append(d)
                data.chains.chains_l[i].append(l_part)

        for i in range(2, 4):
            for k in range(2, 4):
                (d, l_part) = dolfin.split(data.chains.chains[i][k])
                data.chains.chains_d[i].append(d)
                data.chains.chains_l[i].append(l_part)
    else:
        data.chains.chains_d = (
            data.chains.chains
        )  # In singular mode, chains are directly the functions
    return data
