import cvxpy as cp
import numpy as np
from gcsopt.graph_problems.utils import define_variables, enforce_edge_programs, get_solution

def facility_location(conic_graph, binary, tol, **kwargs):

    # define variables
    yv, zv, ye, ze, ze_tail, ze_head = define_variables(conic_graph, binary)

    # edge costs and constraints
    cost, constraints = enforce_edge_programs(conic_graph, ye, ze, ze_tail, ze_head)

    # constraints on the vertices
    for i, vertex in enumerate(conic_graph.vertices):
        inc = conic_graph.incoming_edge_indices(vertex)
        out = conic_graph.outgoing_edge_indices(vertex)

        # check that graph topology is correct
        if len(inc) > 0 and len(out) > 0:
            raise ValueError("Graph is not bipartite.")

        # user vertex
        if len(inc) > 0:
            cost += vertex.cost_homogenization(zv[i], 1)
            constraints += [
                yv[i] == 1,
                sum(ye[inc]) == 1,
                sum(ze_head[inc]) == zv[i]]
        
        # facility vertex
        else:
            cost += vertex.cost_homogenization(zv[i], yv[i])
            constraints.append(yv[i] <= 1)

    # constraints on the edges
    for k, edge in enumerate(conic_graph.edges):
        i = conic_graph.vertex_index(edge.tail)
        constraints += edge.tail.constraint_homogenization(zv[i] - ze_tail[k], yv[i] - ye[k])

    # solve problem
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(**kwargs)

    return get_solution(conic_graph, prob, ye, ze, yv, zv, tol)
