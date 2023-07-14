import cvxpy as cp
import numpy as np


def shortest_path(gcs, s, t, tol=1e-4):

    # compute conic programs on edges and vertices
    gcs.to_conic()

    # binary variables
    yv = cp.Variable(gcs.num_vertices(), boolean=True)
    ye = cp.Variable(gcs.num_edges(), boolean=True)

    # continuous variables
    zv = np.array([cp.Variable(v.conic.num_variables) for v in gcs.vertices])
    ze = np.array([cp.Variable(e.conic.num_variables) for e in gcs.edges])
    ze_out = np.array([cp.Variable(e.tail.conic.num_variables) for e in gcs.edges])
    ze_inc = np.array([cp.Variable(e.head.conic.num_variables) for e in gcs.edges])

    constraints = []
    cost = 0

    for i, v in enumerate(gcs.vertices):
        
        # cost on the vertices including domain constraint
        cost += v.conic.eval_cost(zv[i], yv[i])
        constraints += v.conic.eval_constraints(zv[i], yv[i])
        
        inc_edges = gcs.incoming_indices(v)
        out_edges = gcs.outgoing_indices(v)
        
        # constraints on source variables
        if v == s:
            constraints.append(cp.sum(ye[inc_edges]) == 0)
            constraints.append(cp.sum(ye[out_edges]) == 1)
            constraints.append(yv[i] == sum(ye[out_edges]))
            constraints.append(zv[i] == sum(ze_out[out_edges]))
            
        # constraints on target variables
        elif v == t:
            constraints.append(cp.sum(ye[out_edges]) == 0)
            constraints.append(cp.sum(ye[inc_edges]) == 1)
            constraints.append(yv[i] == sum(ye[inc_edges]))
            constraints.append(zv[i] == sum(ze_inc[inc_edges]))
            
        # constraints on variables of remaining vertices
        else:
            constraints.append(yv[i] == sum(ye[out_edges]))
            constraints.append(yv[i] == sum(ye[inc_edges]))
            constraints.append(yv[i] <= 1)
            constraints.append(zv[i] == sum(ze_out[out_edges]))
            constraints.append(zv[i] == sum(ze_inc[inc_edges]))
            
    for k, e in enumerate(gcs.edges):
        
        # cost on the edges including domain constraint
        cost += e.conic.eval_cost(ze[k], ye[k])
        constraints += e.conic.eval_constraints(ze[k], ye[k])
        constraints += e.tail.conic.eval_constraints(ze_out[k], ye[k])
        constraints += e.head.conic.eval_constraints(ze_inc[k], ye[k])
        
        # euqate auxiliary variables on the egdes
        for variable in e.tail.variables:
            ze_var = e.conic.select_variable(variable, ze[k])
            ze_out_var = e.tail.conic.select_variable(variable, ze_out[k])
            constraints.append(ze_var == ze_out_var)
            
        for variable in e.head.variables:
            ze_var = e.conic.select_variable(variable, ze[k])
            ze_inc_var = e.head.conic.select_variable(variable, ze_inc[k])
            constraints.append(ze_var == ze_inc_var)

    # solve shortest path
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve()

    if prob.status == 'optimal':

        # set values for vertices
        for i, v in enumerate(gcs.vertices):
            v.value = yv[i].value
            for variable in v.variables:
                if prob.status == "optimal" and v.value > tol:
                    zv_var = v.conic.select_variable(variable, zv[i].value)
                    variable.value = zv_var / v.value
                else:
                    variable.value = np.full(variable.shape, np.nan)

        # set values for edges
        for k, e in enumerate(gcs.edges):
            e.value = ye[k].value
            for variable in e.variables:
                if prob.status == "optimal" and e.value > tol:
                    ze_var = e.conic.select_variable(variable, ze[k].value)
                    variable.value = ze_var / e.value
                else:
                    variable.value = np.full(variable.shape, np.nan)

    return prob
