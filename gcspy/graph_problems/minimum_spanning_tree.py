import cvxpy as cp
import numpy as np
from itertools import combinations
from gcspy.graph_problems.utils import define_variables, enforce_edge_programs, get_solution

def minimum_spanning_tree(conic_graph, root, subtour_elimination, binary, tol, **kwargs):
    """
    See ILP formulation from https://www.cs.cmu.edu/afs/cs.cmu.edu/academic/class/15850-f20/www/notes/lec2.pdf
    """

    # Define variables.
    yv, zv, ye, ze, ze_tail, ze_head = define_variables(conic_graph, binary)

    # Edge basic costs and constraints.
    cost, constraints = enforce_edge_programs(conic_graph, ye, ze, ze_tail, ze_head)

    # Cost and constraints on the vertices.
    for i, vertex in enumerate(conic_graph.vertices):
        cost += vertex.cost_homogenization(zv[i], 1)

        # Constraints on incoming edges.
        inc = conic_graph.incoming_edge_indices(vertex)
        if vertex.name == root.name:
            constraints += vertex.constraint_homogenization(zv[i], 1)
            constraints += [ye[k] == 0 for k in inc]
            constraints += [ze_head[k] == 0 for k in inc]
        else:
            constraints += [sum(ye[inc]) == 1, sum(ze_head[inc]) == zv[i]]

        # Constraints on outgoing edges.
        for k in conic_graph.outgoing_edge_indices(vertex):
            constraints += vertex.constraint_homogenization(zv[i] - ze_tail[k], 1 - ye[k])

    # Subtour elimination constraints for all subsets of vertices with
    # cardinality between 2 and num_vertices - 1.
    if subtour_elimination:
        root = conic_graph.get_vertex(root.name)
        i = conic_graph.vertex_index(root)
        subvertices = conic_graph.vertices[:i] + conic_graph.vertices[i+1:]
        for subtour_size in range(2, conic_graph.num_vertices()):
            for vertices in combinations(subvertices, subtour_size):
                inc = conic_graph.incoming_edge_indices(vertices)
                constraints.append(sum(ye[inc]) >= 1)

    # Solve problem.
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(**kwargs)

    # Set value of vertex binaries.
    if prob.status == "optimal":
        yv.value = np.ones(conic_graph.num_vertices())

    return get_solution(conic_graph, prob, ye, ze, yv, zv, tol)
        