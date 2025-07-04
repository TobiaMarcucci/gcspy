import cvxpy as cp
import numpy as np

def conic_graph_problem(conic_graph, additional_constraints, binary=True, callback=None, tol=1e-4, **kwargs):

    # if binary set all the y variables to boolean
    if binary:
        for vertex in conic_graph.vertices:
            vertex.y.attributes['boolean'] = True
        for edge in conic_graph.edges:
            edge.y.attributes['boolean'] = True

    # collect vectors of binary variables
    yv = conic_graph.vertex_binaries()
    ye = conic_graph.edge_binaries()

    # continuous variables for the vertices
    # these are numpy arrays since we want to access them using lists of indices
    xv = np.array([cp.Variable(vertex.size) for vertex in conic_graph.vertices])
    zv = np.array([cp.Variable(vertex.size) for vertex in conic_graph.vertices])

    # continuous variables for the edges
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

    # edge cost
    for k, edge in enumerate(conic_graph.edges):
        cost += edge.evaluate_cost(ze_tail[k], ze_head[k], ze[k], ye[k])

        # tail constraints
        x_tail = xv[conic_graph.vertex_index(edge.tail)]
        constraints += edge.tail.evaluate_constraints(ze_tail[k], ye[k])
        constraints += edge.tail.evaluate_constraints(x_tail - ze_tail[k], 1 - ye[k])

        # head constraints
        x_head = xv[conic_graph.vertex_index(edge.head)]
        constraints += edge.head.evaluate_constraints(ze_head[k], ye[k])
        constraints += edge.head.evaluate_constraints(x_head - ze_head[k], 1 - ye[k])

        # edge constraints
        constraints += edge.evaluate_constraints(ze_tail[k], ze_head[k], ze[k], ye[k])

    # add the problem specific constraints
    constraints += additional_constraints(conic_graph, xv, zv, ze_tail, ze_head)

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

    # get values
    xv_value = [x.value for x in xv]
    xe_value = []
    for (z, y) in zip(ze, ye.value):
        if y is not None and y > tol:
            xe_value.append(z.value / y)
        else:
            xe_value.append(None)

    return prob, xv_value, xe_value

def convex_graph_problem(convex_graph, problem, binary=True, callback=None, tol=1e-4, **kwargs):

    # solve conic version of the problem
    conic_graph = convex_graph.to_conic()
    prob, xv, xe = conic_graph_problem(conic_graph, problem, binary, callback, tol, **kwargs)

    # get back value of vertex variables
    for convex_vertex, x in zip(convex_graph.vertices, xv):
        for convex_variable in convex_vertex.variables:
            if x is None:
                convex_variable.value = None
            else:
                conic_vertex = conic_graph.get_vertex(convex_vertex.name)
                convex_variable.value = conic_vertex.get_convex_variable_value(convex_variable, x)

    # get back value of edge variables
    for convex_edge, x in zip(convex_graph.edges, xe):
        for convex_variable in convex_edge.variables:
            if x is None:
                convex_variable.value = None
            else:
                conic_edge = conic_graph.get_edge(*convex_edge.name)
                convex_variable.value = conic_edge.get_convex_variable_value(convex_variable, x)

    return prob