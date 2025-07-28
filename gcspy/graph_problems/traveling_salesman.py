import cvxpy as cp
import numpy as np
from itertools import combinations
from gcspy.graph_problems.utils import define_variables, enforce_edge_programs, get_solution

def traveling_salesman(conic_graph, subtour_elimination, binary, tol, callback=None, **kwargs):

    # define variables
    yv, zv, ye, ze, ze_tail, ze_head = define_variables(conic_graph, binary)

    # edge costs and constraints
    cost, constraints = enforce_edge_programs(conic_graph, ye, ze, ze_tail, ze_head)

    # add all costs and constraints one vertex at the time
    for i, vertex in enumerate(conic_graph.vertices):
        cost += vertex.cost_homogenization(zv[i], 1)
        inc = conic_graph.incoming_edge_indices(vertex)
        out = conic_graph.outgoing_edge_indices(vertex)
        constraints += [
            sum(ye[inc]) == 1,
            sum(ye[out]) == 1,
            sum(ze_head[inc]) == zv[i],
            sum(ze_tail[out]) == zv[i]]

    # subtour elimination constraints for all subsets of vertices with
    # cardinality between 2 and num_vertices - 2
    if subtour_elimination:
        for subtour_size in range(2, conic_graph.num_vertices() - 1):
            for vertices in combinations(conic_graph.vertices, subtour_size):
                out = conic_graph.outgoing_edge_indices(vertices)
                constraints.append(sum(ye[out]) >= 1)

    # solve problem
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(**kwargs)

    # run callback if one is provided
    if callback is not None:
        while True:
            new_constraints = callback(None, ye)
            if len(new_constraints) == 0:
                break
            constraints += new_constraints
            prob = cp.Problem(cp.Minimize(cost), constraints)
            prob.solve(**kwargs)

    # set value of vertex binaries
    if prob.status == "optimal":
        yv.value = np.ones(conic_graph.num_vertices())

    return get_solution(conic_graph, prob, ye, ze, yv, zv, tol)