from itertools import combinations
from gcspy.graph_problems.graph_problem import convex_graph_problem

def traveling_salesman_constraints(conic_graph, xv, zv, ze_tail, ze_head, subtour_elimination):

    # binary variables
    yv = conic_graph.vertex_binaries()
    ye = conic_graph.edge_binaries()

    # add all constraints one vertex at the time
    constraints = []
    for i, v in enumerate(conic_graph.vertices):
        inc = conic_graph.incoming_indices(v)
        out = conic_graph.outgoing_indices(v)

        constraints += [
            yv[i] == 1,
            sum(ye[out]) == 1,
            sum(ye[inc]) == 1,
            zv[i] == xv[i],
            sum(ze_tail[out]) == xv[i],
            sum(ze_head[inc]) == xv[i]]

    if subtour_elimination:
        for r in range(2, conic_graph.num_vertices() - 1):
            for vertices in combinations(conic_graph.vertices, r):
                out = conic_graph.outgoing_indices(vertices)
                constraints.append(sum(ye[out]) >= 1)

    return constraints

def solve_traveling_salesman(convex_graph, subtour_elimination=True, binary=True, callback=None, **kwargs):
    additional_constraints = lambda *args: traveling_salesman_constraints(*args, subtour_elimination)
    return convex_graph_problem(convex_graph, additional_constraints, binary, callback, **kwargs)
