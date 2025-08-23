import cvxpy as cp
import numpy as np
from gcsopt.graph_problems.utils import (define_variables, enforce_edge_programs,
    get_solution, subtour_elimination_constraints)

def traveling_salesman(conic_graph, subtour_elimination, binary, tol, callback=None, **kwargs):

    # Define variables.
    yv, zv, ye, ze, ze_tail, ze_head = define_variables(conic_graph, binary)

    # Edge costs and constraints.
    cost, constraints = enforce_edge_programs(conic_graph, ye, ze, ze_tail, ze_head)

    # Vertex costs and constraints.
    for i, vertex in enumerate(conic_graph.vertices):
        cost += vertex.cost_homogenization(zv[i], 1)

        # Directed graphs.
        if conic_graph.directed:
            inc = conic_graph.incoming_edge_indices(vertex)
            out = conic_graph.outgoing_edge_indices(vertex)
            constraints += [
                sum(ye[inc]) == 1,
                sum(ye[out]) == 1,
                sum(ze_head[inc]) == zv[i],
                sum(ze_tail[out]) == zv[i]]
            
        # Undirected graphs graphs.
        else:
            incident = conic_graph.incident_edge_indices(vertex)
            inc = [k for k in incident if conic_graph.edges[k].head == vertex]
            out = [k for k in incident if conic_graph.edges[k].tail == vertex]
            constraints += [
                sum(ye[incident]) == 2,
                sum(ze_head[inc]) + sum(ze_tail[out]) == 2 * zv[i]]
            for k in inc:
                constraints += vertex.constraint_homogenization(zv[i] - ze_head[k], 1 - ye[k])
            for k in out:
                constraints += vertex.constraint_homogenization(zv[i] - ze_tail[k], 1 - ye[k])

    # Exponentially many subtour elimination constraints.
    if subtour_elimination:
        constraints += subtour_elimination_constraints(conic_graph, ye)

    # Solve problem.
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(**kwargs)

    # Run callback if one is provided.
    if callback is not None:
        while True:
            new_constraints = callback(None, ye)
            if len(new_constraints) == 0:
                break
            constraints += new_constraints
            prob = cp.Problem(cp.Minimize(cost), constraints)
            prob.solve(**kwargs)

    # Set value of vertex binaries.
    if prob.status == "optimal":
        yv.value = np.ones(conic_graph.num_vertices())

    return get_solution(conic_graph, prob, ye, ze, yv, zv, tol)
