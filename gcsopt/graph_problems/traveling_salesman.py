import cvxpy as cp
import numpy as np
from gcsopt.graph_problems.utils import (define_variables,
    enforce_edge_programs, set_solution, subtour_elimination_constraints)

def traveling_salesman(conic_graph, subtour_elimination, binary, tol, **kwargs):

    # Define variables.
    yv, zv, ye, ze, ze_tail, ze_head = define_variables(conic_graph, binary)

    # Enforce edge costs and constraints.
    cost, constraints = enforce_edge_programs(conic_graph, ye, ze, ze_tail, ze_head)

    # Enforce vertex costs and constraints.
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
            inc = conic_graph._incoming_edge_indices(vertex)
            out = conic_graph._outgoing_edge_indices(vertex)
            constraints += [
                sum(ye[inc + out]) == 2,
                sum(ze_head[inc]) + sum(ze_tail[out]) == 2 * zv[i]]
            for k in inc:
                constraints += vertex.constraint_homogenization(zv[i] - ze_head[k], 1 - ye[k])
            for k in out:
                constraints += vertex.constraint_homogenization(zv[i] - ze_tail[k], 1 - ye[k])

    # Exponentially many subtour elimination constraints.
    if subtour_elimination:
        constraints += subtour_elimination_constraints(conic_graph, ye)

    # Solve problem and set solution.
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(**kwargs)
    if prob.status == "optimal":
        yv.value = np.ones(conic_graph.num_vertices())
    set_solution(conic_graph, prob, ye, ze, yv, zv, tol)