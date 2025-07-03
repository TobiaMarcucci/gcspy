import cvxpy as cp

def problem_setup(graph):

    # binary variables
    yv = graph.vertex_binaries()
    ye = graph.edge_binaries()

    # continuous variables for the vertices
    xv = [cp.Variable(vertex.size) for vertex in graph.vertices]
    zv = [cp.Variable(vertex.size) for vertex in graph.vertices]

    # continuous variables for the edges
    xe = [cp.Variable(edge.additional_size) for edge in graph.edges]
    ze = [cp.Variable(edge.additional_size) for edge in graph.edges]
    ze_tail = [cp.Variable(edge.tail.size) for edge in graph.edges]
    ze_head = [cp.Variable(edge.head.size) for edge in graph.edges]

    cost = 0
    constraints = []

    # cost and constraints on the vertices
    for i, vertex in enumerate(graph.vertices):
        cost += vertex.evaluate_cost(zv[i], yv[i])
        constraints += vertex.evaluate_constraints(zv[i], yv[i])
        constraints += vertex.evaluate_constraints(xv[i] - zv[i], 1 - yv[i])

    # cost and constraints on the edges
    for k, edge in enumerate(graph.edges):
        cost += edge.evaluate_cost(ze_tail[k], ze_head[k], ze[k], ye[k])
        constraints += edge.evaluate_constraints(ze_tail[k], ze_head[k], ze[k], ye[k])
        constraints += edge.tail.evaluate_constraints(ze_tail[k], ye[k])
        constraints += edge.head.evaluate_constraints(ze_head[k], ye[k])
        x_tail = xv[graph.vertex_index(edge.tail)]
        x_head = xv[graph.vertex_index(edge.head)]
        # TODO: double check that the following constraints are actually needed
        constraints += edge.evaluate_constraints(x_tail - ze_tail[k], x_head - ze_head[k], xe[k] - ze[k], 1 - ye[k])
        constraints += edge.tail.evaluate_constraints(x_tail - ze_tail[k], 1 - ye[k])
        constraints += edge.head.evaluate_constraints(x_head - ze_head[k], 1 - ye[k])

    return cost, constraints
