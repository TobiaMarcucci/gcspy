import cvxpy as cp
import numpy as np

def define_variables(conic_graph, binary):

    # Binary variables.
    yv = cp.Variable(conic_graph.num_vertices(), boolean=binary)
    ye = cp.Variable(conic_graph.num_edges(), boolean=binary)
    
    # Function that allows adding variables of zero size.
    safe_variable = lambda size: cp.Variable(size) if size > 0 else np.array([])

    # Auxiliary continuous varibales.
    zv = np.array([cp.Variable(vertex.size) for vertex in conic_graph.vertices])
    ze = np.array([safe_variable(edge.slack_size) for edge in conic_graph.edges])
    ze_tail = np.array([cp.Variable(edge.tail.size) for edge in conic_graph.edges])
    ze_head = np.array([cp.Variable(edge.head.size) for edge in conic_graph.edges])

    return yv, zv, ye, ze, ze_tail, ze_head

def enforce_edge_programs(conic_graph, ye, ze, ze_tail, ze_head):

    # Edge costs and constraints.
    cost = 0
    constraints = []
    for k, edge in enumerate(conic_graph.edges):
        cost += edge.cost_homogenization(ze_tail[k], ze_head[k], ze[k], ye[k])
        constraints += edge.constraint_homogenization(ze_tail[k], ze_head[k], ze[k], ye[k])
        constraints += edge.tail.constraint_homogenization(ze_tail[k], ye[k])
        constraints += edge.head.constraint_homogenization(ze_head[k], ye[k])

    return cost, constraints

def get_solution(conic_graph, prob, ye, ze, yv, zv, tol):

    # Set vertex variable values.
    for vertex, y, z in zip(conic_graph.vertices, yv, zv):
        vertex.binary_variable.value = y.value
        if y.value is not None and y.value > tol:
            vertex.x.value = z.value / y.value
        else:
            vertex.x.value = None

    # set edge variable values
    for edge, y, z in zip(conic_graph.edges, ye, ze):
        edge.binary_variable.value = y.value
        if y.value is not None and y.value > tol:
            edge.x.value = np.concatenate((
                edge.tail.x.value,
                edge.head.x.value,
                z.value / y.value))

    return prob