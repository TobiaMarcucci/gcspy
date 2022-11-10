import numpy as np
import cvxpy as cp

from cvxpy.constraints.zero import Zero
from cvxpy.constraints.nonpos import NonNeg
from cvxpy.constraints.second_order import SOC
from cvxpy.constraints.psd import PSD
from cvxpy.constraints.exponential import ExpCone
from cvxpy.constraints.power import PowCone3D


def to_cone_program(cost, constraints):
    
    # Translates the problem of minimizing the cost subject to the constraints
    # into a cone program.
    prob = cp.Problem(cp.Minimize(cost), constraints)
    opts = {'use_quad_obj': False}
    chain = prob._construct_chain(solver_opts=opts)
    chain.reductions = chain.reductions[:-1]
    cone_prob = chain.apply(prob)[0]
    # data, _, invdata = prob.get_problem_data(cp.SCS, solver_opts=opts) #

    # Extracts the offsets in the cost and the constraints.
    cone_data = {}
    cd = cone_prob.c.toarray().flatten()
    Ab = cone_prob.A.toarray().reshape((-1, len(cd)), order='F')
    cone_data['c'] = cd[:-1]
    cone_data['d'] = cd[-1]
    cone_data['A'] = Ab[:, :-1]
    cone_data['b'] = Ab[:, -1]
    # cone_data['A'] = - data['A'] #
    # cone_data['b'] = data['b'] #
    # cone_data['c'] = data['c'] #
    # cone_data['d'] = invdata[-1]['offset'] #

    # Extracts the type and the size of each cone constraint.
    # Still not correct, might get a sym variable and break.
    cone_data['cones'] = [(type(c), c.size) for c in cone_prob.constraints]
    # cone_data['cones'] = [(type(c), c.size) for c in data['param_prob'].constraints] #

    # Maps the id of each variable in the cone program to the
    cone_data['windows'] = {}
    for x in cone_prob.variables:
        start = cone_prob.var_id_to_col[x.id]
    # for x in data['param_prob'].variables: #
    #     start = data['param_prob'].var_id_to_col[x.id] #
        cone_data['windows'][x] = range(start, start + x.size)
    
    return cone_data


def constrain_in_cone(Axb, cone):

    if cone == Zero:
         return Zero(Axb)

    elif cone == NonNeg:
         return NonNeg(Axb)

    elif cone == SOC:
        return SOC(Axb[0], Axb[1:])

    elif cone == ExpCone: # not sure about this
        assert len(Axb.shape) == 1
        step = Axb.size // 3
        return ExpCone(Axb[:step], Axb[step:-step], Axb[-step:])

    elif cone == PSD: # not sure about this
        assert len(Axb.shape) == 1
        n = int(Axb.size ** .5)
        Axb_mat = cp.reshape(Axb, (n, n))
        return PSD(Axb_mat)

    elif cone == PowCone3D:
        raise NotImplementedError

    else:
        raise NotImplementedError


def flatten(x_old, x_new):

    if x_old.attributes['diag']:
        return diag(x_new)

    elif x_old.is_symmetric() and x_old.size > 1:
        return x_new[np.tril_indices(x_old.shape[0])[::-1]]

    elif len(x_old.shape) > 1:
        return cp.vec(x_new)

    else:
        return x_new


def cone_perspective(cone_data, variable_map, t):

    # Loop through the variables.
    cost = cone_data['d'] * t
    Axbt = cone_data['b'] * t
    auxiliary_variables = []
    for x_old, window in cone_data['windows'].items():
        if x_old in variable_map:
            x_new = flatten(x_old, variable_map[x_old])
        else:
            x_new = cp.Variable(len(window))
            auxiliary_variables.append(x_new)
        cost = cost + cone_data['c'][window] @ x_new
        Axbt = Axbt + cone_data['A'][:, window] @ x_new

    # Loop through the constraints.
    constraints = [t >= 0]
    row = 0
    for cone, size in cone_data['cones']:
        constraints.append(constrain_in_cone(Axbt[row : row + size], cone))
        row += size

    return cost, constraints, auxiliary_variables
