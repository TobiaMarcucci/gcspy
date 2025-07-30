import cvxpy as cp
import numpy as np
from itertools import combinations
from gcspy.graph_problems.utils import define_variables, enforce_edge_programs, get_solution

def minimum_spanning_tree(conic_graph, root, subtour_elimination, binary, tol, **kwargs):

    # define variables
    yv, zv, ye, ze, ze_tail, ze_head = define_variables(conic_graph, binary)

    # edge costs and constraints
    cost, constraints = enforce_edge_programs(conic_graph, ye, ze, ze_tail, ze_head)

    # constraints on the vertices
    for i, vertex in enumerate(conic_graph.vertices):
        cost += vertex.cost_homogenization(zv[i], 1)
        inc = conic_graph.incoming_edge_indices(vertex)
        if vertex.name == root.name:
            constraints += vertex.constraint_homogenization(zv[i], 1)
            constraints += [ye[k] == 0 for k in inc]
            constraints += [ze_head[k] == 0 for k in inc]
        else:
            constraints += [sum(ye[inc]) == 1, sum(ze_head[inc]) == zv[i]]

    # constraints on the edges
    for k, edge in enumerate(conic_graph.edges):
        z_tail = zv[conic_graph.vertex_index(edge.tail)]
        constraints += edge.tail.constraint_homogenization(z_tail - ze_tail[k], 1 - ye[k])

    # subtour elimination constraints for all subsets of vertices with
    # cardinality between 2 and num_vertices - 1
    if subtour_elimination:
        root = conic_graph.get_vertex(root.name)
        i = conic_graph.vertex_index(root)
        subvertices = conic_graph.vertices[:i] + conic_graph.vertices[i+1:]
        for subtour_size in range(2, conic_graph.num_vertices()):
            for vertices in combinations(subvertices, subtour_size):
                inc = conic_graph.incoming_edge_indices(vertices)
                constraints.append(sum(ye[inc]) >= 1)

    # solve problem
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(**kwargs)

    # set value of vertex binaries
    if prob.status == "optimal":
        yv.value = np.ones(conic_graph.num_vertices())

    return get_solution(conic_graph, prob, ye, ze, yv, zv, tol)
        