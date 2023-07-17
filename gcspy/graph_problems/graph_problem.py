import cvxpy as cp
import numpy as np


def graph_problem(gcs, problem):

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

    cost = 0
    constraints = []

    for i, v in enumerate(gcs.vertices):
        
        # cost on the vertices including domain constraint
        cost += v.conic.eval_cost(zv[i], yv[i])
        constraints += v.conic.eval_constraints(zv[i], yv[i])
        
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

    probelm_specific_constraints = problem(gcs, yv, ye, zv, ze_out, ze_inc)
    constraints += probelm_specific_constraints

    # solve problem
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve()

    if prob.status == 'optimal':
        tol = 1e-4

        # set values for vertices
        for i, v in enumerate(gcs.vertices):
            v.value = yv[i].value
            for variable in v.variables:
                if prob.status == "optimal" and v.value > tol:
                    zv_var = v.conic.select_variable(variable, zv[i].value)
                    variable.value = zv_var / v.value
                else:
                    variable.value = None

        # set values for edges
        for k, e in enumerate(gcs.edges):
            e.value = ye[k].value
            for variable in e.variables:
                if prob.status == "optimal" and e.value > tol:
                    ze_var = e.conic.select_variable(variable, ze[k].value)
                    variable.value = ze_var / e.value
                else:
                    variable.value = None

    return prob
