import numpy as np
import cvxpy as cp
import gurobipy as gp
from gurobipy import GRB

def constrain_in_cone(model, z, K):

    # Linear constraints.
    if K == cp.Zero:
        model.addConstr(z == 0)
    elif K == cp.NonNeg:
        model.addConstr(z >= 0)
    elif K == cp.NonPos:
        model.addConstr(z <= 0)

    # Second order cone constraint.
    elif K == cp.SOC:
        s = model.addVar() # Nonnegative slack variable.
        model.addConstr(z[0] == s)
        model.addConstr(z[1:] @ z[1:] <= s ** 2) # Convex for gurobi.

    # TODO: support all cone constraints.
    else:
        raise NotImplementedError

def cost_homogenization(prog, x, y):
    return prog.c @ x + prog.d * y

def constraint_homogenization(model, prog, x, y):
    z = prog.A @ x + prog.b * y
    start = 0
    for cone_type, cone_size in prog.K:
        stop = start + cone_size
        constrain_in_cone(model, z[start:stop], cone_type)
        start = stop

def edge_cost_homogenization(edge, xv, xw, xe, y):
    edge._check_vector_sizes(xv, xw, xe)
    x = gp.concatenate((xv, xw, xe))
    return cost_homogenization(edge, x, y)
    
def edge_constraint_homogenization(model, edge, xv, xw, xe, y):
    edge._check_vector_sizes(xv, xw, xe)
    x = gp.concatenate((xv, xw, xe))
    constraint_homogenization(model, edge, x, y)

def define_variables(model, conic_graph, binary):

    # Binary variables.
    vtype = GRB.BINARY if binary else GRB.CONTINUOUS
    ye = model.addMVar(conic_graph.num_edges(), vtype=vtype)
    
    # Auxiliary continuous varibales.
    add_var = lambda n : model.addMVar(n, lb=-np.inf)
    zv = np.array([add_var(vertex.size) for vertex in conic_graph.vertices])
    ze = np.array([add_var(edge.slack_size) for edge in conic_graph.edges])
    ze_tail = np.array([add_var(edge.tail.size) for edge in conic_graph.edges])
    ze_head = np.array([add_var(edge.head.size) for edge in conic_graph.edges])

    return zv, ye, ze, ze_tail, ze_head

def enforce_edge_programs(model, conic_graph, ye, ze, ze_tail, ze_head):

    # Edge costs and constraints.
    cost = 0
    for k, edge in enumerate(conic_graph.edges):
        cost += edge_cost_homogenization(edge, ze_tail[k], ze_head[k], ze[k], ye[k])
        edge_constraint_homogenization(model, edge, ze_tail[k], ze_head[k], ze[k], ye[k])
        constraint_homogenization(model, edge.tail, ze_tail[k], ye[k])
        constraint_homogenization(model, edge.head, ze_head[k], ye[k])

    return cost

def get_solution(conic_graph, zv, ye, ze, tol):

    # Set vertex variable values.
    for vertex, z in zip(conic_graph.vertices, zv):
        vertex.binary_variable.value = 1
        vertex.x.value = z.X

    # Set edge variable values.
    for edge, y, z in zip(conic_graph.edges, ye, ze):
        edge.binary_variable.value = y.X
        if y.X is not None and y.X > tol:
            edge.x.value = np.concatenate((
                edge.tail.x.value,
                edge.head.x.value,
                z.X / y.X))