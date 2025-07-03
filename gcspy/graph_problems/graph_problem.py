import cvxpy as cp
import numpy as np

def graph_problem(convex_graph, problem, binary=True, callback=None, tol=1e-4, **kwargs):

    # if binary set all the variables to boolean
    if binary:
        for vertex in convex_graph.vertices:
            vertex.y.attributes['boolean'] = True
        for edge in convex_graph.edges:
            edge.y.attributes['boolean'] = True

    # translate problem to conic graph
    conic_graph = convex_graph.to_conic()

    # binary variables
    yv = conic_graph.vertex_binaries()
    ye = conic_graph.edge_binaries()

    # continuous variables for the vertices
    # these are numpy arrays since we want to access them using lists of indices
    xv = np.array([cp.Variable(vertex.size) for vertex in conic_graph.vertices])
    zv = np.array([cp.Variable(vertex.size) for vertex in conic_graph.vertices])

    # continuous variables for the edges
    xe = np.array([cp.Variable(edge.additional_size) for edge in conic_graph.edges])
    ze = np.array([cp.Variable(edge.additional_size) for edge in conic_graph.edges])
    ze_tail = np.array([cp.Variable(edge.tail.size) for edge in conic_graph.edges])
    ze_head = np.array([cp.Variable(edge.head.size) for edge in conic_graph.edges])

    # cost and constraints on the vertices
    cost = 0
    constraints = []
    for i, vertex in enumerate(conic_graph.vertices):
        cost += vertex.evaluate_cost(zv[i], yv[i])
        constraints += vertex.evaluate_constraints(zv[i], yv[i])
        constraints += vertex.evaluate_constraints(xv[i] - zv[i], 1 - yv[i])

    # cost and constraints on the edges
    for k, edge in enumerate(conic_graph.edges):
        cost += edge.evaluate_cost(ze_tail[k], ze_head[k], ze[k], ye[k])
        constraints += edge.evaluate_constraints(ze_tail[k], ze_head[k], ze[k], ye[k])
        constraints += edge.evaluate_constraints(x_tail - ze_tail[k], x_head - ze_head[k], xe[k] - ze[k], 1 - ye[k])

        # tail constraints
        x_tail = xv[conic_graph.vertex_index(edge.tail)]
        constraints += edge.tail.evaluate_constraints(ze_tail[k], ye[k])
        constraints += edge.tail.evaluate_constraints(x_tail - ze_tail[k], 1 - ye[k])

        # head constraints
        x_head = xv[conic_graph.vertex_index(edge.head)]
        constraints += edge.head.evaluate_constraints(ze_head[k], ye[k])
        constraints += edge.head.evaluate_constraints(x_head - ze_head[k], 1 - ye[k])

    # add the problem specific constraints
    constraints += problem(conic_graph, xv, zv, ze_tail, ze_head)

    # solve problem
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(**kwargs)

    # run callback if one is provided
    if callback is not None:
        while True:
            new_constraints = callback(yv, ye)
            if len(new_constraints) == 0:
                break
            constraints += new_constraints
            prob = cp.Problem(cp.Minimize(cost), constraints)
            prob.solve()

    # set values of vertex variables
    for vertex, xvi in zip(convex_graph.vertices, xv):
        for variable in vertex.variables:
            if xvi.value is None:
                variable.value = None
            else:
                variable.value = get_variable_value(variable, xvi.value, vertex.id_to_cols)

    # set values of edge variables
    for edge, xek in zip(convex_graph.edges, xe):
        for variable in edge.variables:
            if xek.value is None:
                variable.value = None
            else:
                variable.value = get_variable_value(variable, xek.value, edge.id_to_cols)

    return prob

def get_variable_value(variable, x, id_to_cols):

    # external variable
    if not variable.id in id_to_cols:
        raise ValueError('External variable in the retrieval of the optimal solution.')
    
    value = x[id_to_cols[variable.id]]
    if variable.is_matrix():
        if variable.is_symmetric():
            n = variable.shape[0]
            full = np.zeros((n, n))
            full[np.triu_indices(n)] = value
            value = full + full.T
            value[np.diag_indices(n)] /= 2
        else:
            value = value.reshape(variable.shape, order='F')

    return value