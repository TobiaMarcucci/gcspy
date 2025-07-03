import cvxpy as cp
from gcspy.graph_problems import graph_problem

def shortest_path_constraints(graph, xv, zv, ze_tail, ze_head, s, t):

    # binary variables
    yv = graph.vertex_binaries()
    ye = graph.edge_binaries()

    # add all constraints one vertex at the time
    constraints = []
    for i, vertex in enumerate(graph.vertices):
        inc = graph.incoming_indices(vertex)
        out = graph.outgoing_indices(vertex)

        # source constraints
        if vertex == s:
            constraints += [
                yv[i] == 1,
                cp.sum(ye[inc]) == 0,
                cp.sum(ye[out]) == 1,
                zv[i] == xv[i],
                zv[i] == cp.sum(ze_tail[out]),
            ]

        # target constraints
        elif vertex == t:
            constraints += [
                yv[i] == 1,
                cp.sum(ye[inc]) == 1,
                cp.sum(ye[out]) == 0,
                zv[i] == xv[i],
                zv[i] == cp.sum(ze_head[inc]),
            ]

        # all other vertices constraints
        else:
            constraints += [
                yv[i] == cp.sum(ye[inc]),
                yv[i] == cp.sum(ye[out]),
                zv[i] == cp.sum(ze_tail[out]),
                zv[i] == cp.sum(ze_head[inc]),
            ]
            
    return constraints

def solve_shortest_path(convex_graph, s, t, binary=True, **kwargs):
    additional_constraints = lambda *args: shortest_path_constraints(*args, s, t)
    return graph_problem(convex_graph, additional_constraints, binary, callback=None, **kwargs)
