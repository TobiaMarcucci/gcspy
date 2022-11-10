import numpy as np
import cvxpy as cp
from numbers import Number


def sym2num(cost, constraints):
    
    # Apply reductions to auxiliary program.
    prob = cp.Problem(cp.Minimize(cost), constraints)
    chain = prob._construct_chain()
    chain.reductions = chain.reductions[:-1]
    cone_prob = chain.apply(prob)[0]
    
    # Map variable ids to range in the columns of A.
    var_range = {}
    for x in cone_prob.variables:
        start = cone_prob.var_id_to_col[x.id]
        var_range[x.id] = range(start, start + x.size)
        
    return cone_prob, var_range


def sym2num_cost(cost):
    
    # Apply reductions with no constraints.
    cone_prob, var_range = sym2num(cost, [])
    
    # Extract cost matrices.
    cd = cone_prob.c.toarray().flatten()
    c = {x_id: cd[x_range] for x_id, x_range in var_range.items()}
    d = cd[-1]
    
    return c, d


def sym2num_constraint(const):
    
    # Apply reductions with zero cost.
    cone_prob, var_range = sym2num(0, [const])

    # Extract constraint matrices.
    m = cone_prob.c.shape[0]
    Ab = cone_prob.A.toarray().reshape((-1, m), order='F')
    A = {x_id: Ab[:, x_range] for x_id, x_range in var_range.items()}
    b = Ab[:, -1]

    # Extract cone.
    assert len(cone_prob.constraints) == 1
    K = type(cone_prob.constraints[0])
    
    return A, b, K


def dcp2cone(cost, constraints):
    
    # Auxiliary dcp.
    prob = cp.Problem(cp.Minimize(cost), constraints)
    assert prob.is_dcp()
    
    # Translate to cone program.
    reduction = cp.reductions.Dcp2Cone(prob)
    cone_prob = reduction.reduce()
    cone_cost = cone_prob.objective.expr
    cone_constraints = cone_prob.constraints
    
    # Extract matrices.
    matrices = {}
    if isinstance(cost, Number):
        matrices['c'] = {}
        matrices['d'] = cost
    else:
        matrices['c'], matrices['d'] = sym2num_cost(cone_cost)
    num_constraints = [sym2num_constraint(c) for c in cone_constraints]
    matrices['A'] = [c[0] for c in num_constraints]
    matrices['b'] = [c[1] for c in num_constraints]
    matrices['K'] = [c[2] for c in num_constraints]

    # Store data.
    var_ids = [x.id for x in prob.variables()]
    aux_variables = [x for x in cone_prob.variables() if x.id not in var_ids]
    data = {'matrices': matrices,
            'variables': prob.variables(),
            'aux_variables': aux_variables}
    
    return data


def constrain_in_cone(Axb, K):

    if K == cp.constraints.Zero:
         return cp.constraints.Zero(Axb)

    elif K == cp.constraints.NonNeg:
         return cp.constraints.NonNeg(Axb)

    elif K == cp.constraints.SOC:
        return cp.constraints.SOC(Axb[0], Axb[1:])

    elif K == cp.constraints.ExpCone: # not sure about this
        assert len(Axb.shape) == 1
        step = Axb.size // 3
        return cp.constraints.ExpCone(Axb[:step], Axb[step:-step], Axb[-step:])

    elif K == cp.constraints.PSD:
        assert len(Axb.shape) == 1
        n = int(Axb.size ** .5)
        Axb_mat = cp.reshape(Axb, (n, n))
        return cp.constraints.PSD(Axb_mat)

    elif K == cp.constraints.PowCone3D:
        raise NotImplementedError

    else:
        raise NotImplementedError


def vec_as(x_new, x_old):
    
    if x_old.attributes['diag']:
        return cp.diag(x_new)

    elif x_old.attributes['symmetric']:
        return x_new[np.tril_indices(x_old.shape[0])[::-1]]

    else:
        return cp.vec(x_new)


def cone_perspective(cone_data, substitution, t):

    # Unpack cone data.
    matrices = cone_data['matrices']
    variables = cone_data['variables'] + cone_data['aux_variables']

    # Dictionary from variable id to vectorized evaluation point.
    evaluation = {x.id: vec_as(x, x) for x in variables}
    for x, x_new in substitution.items():
        evaluation[x.id] = vec_as(x_new, x)

    # Cost.
    multiply = lambda a, b: sum(a[x.id] @ b[x.id] for x in variables if x.id in a)
    cost = matrices['d'] * t + multiply(matrices['c'], evaluation)

    # Constraints.
    constraints = []
    for A, b, K in zip(matrices['A'], matrices['b'], matrices['K']):
        Axb = b * t + multiply(A, evaluation)
        constraints.append(constrain_in_cone(Axb, K))

    return cost, constraints
