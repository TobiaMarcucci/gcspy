import cvxpy as cp
import numpy as np

def shortest_path(conic_graph, source_name, target_name, binary, tol=1e-4, **kwargs):

    # function that allows to add variables of size zero
    safe_variable = lambda size: cp.Variable(size) if size > 0 else np.array([])

    # binary variables
    yv = cp.Variable(conic_graph.num_vertices(), boolean=binary)
    ye = cp.Variable(conic_graph.num_edges(), boolean=binary)

    # auxiliary continuous varibales
    zv = np.array([cp.Variable(vertex.size) for vertex in conic_graph.vertices])
    ze = np.array([safe_variable(edge.slack_size) for edge in conic_graph.edges])
    ze_tail = np.array([cp.Variable(edge.tail.size) for edge in conic_graph.edges])
    ze_head = np.array([cp.Variable(edge.head.size) for edge in conic_graph.edges])

    # edge costs and constraints
    cost = 0
    constraints = []
    for k, edge in enumerate(conic_graph.edges):
        cost += edge.evaluate_cost(ze_tail[k], ze_head[k], ze[k], ye[k])
        constraints += edge.evaluate_constraints(ze_tail[k], ze_head[k], ze[k], ye[k])
        constraints += edge.tail.evaluate_constraints(ze_tail[k], ye[k])
        constraints += edge.head.evaluate_constraints(ze_head[k], ye[k])

    # shortest path constraints
    for i, vertex in enumerate(conic_graph.vertices):
        inc = conic_graph.incoming_edge_indices(vertex)
        out = conic_graph.outgoing_edge_indices(vertex)

        # source cost and constraints
        if vertex.name == source_name:
            cost += vertex.evaluate_cost(zv[i])
            constraints += [yv[i] == 1, 1 == sum(ye[out]), zv[i] == sum(ze_tail[out])]
            for k in inc:
                constraints += [ye[k] == 0, ze_head[k] == 0]

        # target cost and constraints
        elif vertex.name == target_name:
            cost += vertex.evaluate_cost(zv[i])
            constraints += [yv[i] == 1, 1 == sum(ye[inc]), zv[i] == sum(ze_head[inc])]
            for k in out:
                constraints += [ye[k] == 0, ze_tail[k] == 0]

        # cost and constraints on other vertices
        else:
            cost += vertex.evaluate_cost(zv[i], yv[i])
            constraints += [
                yv[i] <= 1,
                yv[i] == sum(ye[inc]),
                yv[i] == sum(ye[out]),
                zv[i] == sum(ze_head[inc]),
                zv[i] == sum(ze_tail[out])]
           
    # solve problem
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(**kwargs)

    # if problem is not solved to optimality
    if prob.status != 'optimal':
        xv = np.full(conic_graph.num_vertices(), None)
        xe = np.full(conic_graph.num_edges(), None)
        yv = xv
        ye = xe
        return prob, xv, yv, xe, ye
    
    # set edge variable values
    ye_value = ye.value
    xe_value = []
    for z, y_value in zip(ze, ye_value):
        if y_value < tol:
            xe_value.append(None)
        elif z.size == 0:
            xe_value.append(np.array([]))
        else:
            xe_value.append(z.value / y_value)
    
    # set vertex variable values
    yv_value = yv.value
    xv_value = []
    for z, y_value in zip(zv, yv_value):
        if y_value < tol:
            xv_value.append(None)
        else:
            xv_value.append(z.value / y_value)
            
    return prob, xv_value, yv_value, xe_value, ye_value