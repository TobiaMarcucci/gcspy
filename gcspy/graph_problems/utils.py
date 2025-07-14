import cvxpy as cp
import numpy as np

def define_variables(conic_graph, binary):

    # binary variables
    yv = cp.Variable(conic_graph.num_vertices(), boolean=binary)
    ye = cp.Variable(conic_graph.num_edges(), boolean=binary)
    
    # function that allows to add variables of size zero
    safe_variable = lambda size: cp.Variable(size) if size > 0 else np.array([])

    # auxiliary continuous varibales
    zv = np.array([cp.Variable(vertex.size) for vertex in conic_graph.vertices])
    ze = np.array([safe_variable(edge.slack_size) for edge in conic_graph.edges])
    ze_tail = np.array([cp.Variable(edge.tail.size) for edge in conic_graph.edges])
    ze_head = np.array([cp.Variable(edge.head.size) for edge in conic_graph.edges])

    return yv, zv, ye, ze, ze_tail, ze_head

def enforce_edge_programs(conic_graph, ye, ze, ze_tail, ze_head):

    # edge costs and constraints
    cost = 0
    constraints = []
    for k, edge in enumerate(conic_graph.edges):
        cost += edge.evaluate_cost(ze_tail[k], ze_head[k], ze[k], ye[k])
        constraints += edge.evaluate_constraints(ze_tail[k], ze_head[k], ze[k], ye[k])
        constraints += edge.tail.evaluate_constraints(ze_tail[k], ye[k])
        constraints += edge.head.evaluate_constraints(ze_head[k], ye[k])

    return cost, constraints

def get_solution(conic_graph, prob, ye, ze, yv, zv, tol):

    # if problem is not solved to optimality
    if prob.status != "optimal":
        xv_value = np.full(conic_graph.num_vertices(), None)
        xe_value = np.full(conic_graph.num_edges(), None)
        yv_value = xv_value
        ye_value = xe_value

    # if problem is solved to optimality
    else:

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