import cvxpy as cp
from gcsopt.graph_problems.utils import (define_variables,
    enforce_edge_programs, set_solution)

def facility_location(conic_graph, binary, tol, **kwargs):

    # Define variables.
    yv, zv, ye, ze, ze_tail, ze_head = define_variables(conic_graph, binary)

    # Enforce edge costs and constraints.
    cost, constraints = enforce_edge_programs(conic_graph, ye, ze, ze_tail, ze_head)

    # Enforce vertex costs and constraints.
    for i, vertex in enumerate(conic_graph.vertices):
        inc = conic_graph.incoming_edge_indices(vertex)
        out = conic_graph.outgoing_edge_indices(vertex)

        # Check that graph topology is correct.
        if len(inc) > 0 and len(out) > 0:
            raise ValueError("Graph is not bipartite.")

        # User vertices.
        if len(inc) > 0:
            cost += vertex.cost_homogenization(zv[i], 1)
            constraints += [
                yv[i] == 1,
                sum(ye[inc]) == 1,
                sum(ze_head[inc]) == zv[i]]
        
        # Facility vertices.
        else:
            cost += vertex.cost_homogenization(zv[i], yv[i])
            constraints.append(yv[i] <= 1)

    # Edge constraints.
    for k, edge in enumerate(conic_graph.edges):
        i = conic_graph.vertex_index(edge.tail)
        constraints += edge.tail.constraint_homogenization(zv[i] - ze_tail[k], yv[i] - ye[k])

    # Solve problem and set solution.
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(**kwargs)
    set_solution(conic_graph, prob, ye, ze, yv, zv, tol)