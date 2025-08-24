import cvxpy as cp
import numpy as np
from itertools import combinations
from gcsopt.graph_problems.utils import (define_variables,
    enforce_edge_programs, subtour_elimination_constraints, set_solution)

def undirected_minimum_spanning_tree(conic_graph, subtour_elimination, binary, tol, **kwargs):
    """
    Here we use the subtour-elimination formulation, which is also perfect.
    """

    # Check that graph is undirected.
    if conic_graph.directed:
        raise ValueError("Called MSTP for undirected graphs on a directed graph.")

    # Define variables.
    yv, zv, ye, ze, ze_tail, ze_head = define_variables(conic_graph, binary)

    # Enforce edge costs and constraints.
    cost, constraints = enforce_edge_programs(conic_graph, ye, ze, ze_tail, ze_head)

    # Number of edges in the tree.
    constraints.append(sum(ye) == conic_graph.num_vertices() - 1)

    # Enforce vertex costs and constraints.
    for i, vertex in enumerate(conic_graph.vertices):
        cost += vertex.cost_homogenization(zv[i], 1)

        # Cutset constraints for one vertex only.
        inc = conic_graph._incoming_edge_indices(vertex)
        out = conic_graph._outgoing_edge_indices(vertex)
        constraints += vertex.constraint_homogenization(
            sum(ze_head[inc]) + sum(ze_tail[out]) - zv[i],
            sum(ye[inc + out]) - 1)
        
        # Constraints implied by ye <= 1.
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

def directed_minimum_spanning_tree(conic_graph, conic_root, subtour_elimination, binary, tol, **kwargs):
    """
    This is the cutset formulation of the directed MSTP. Unlike the undirected
    case, this formulation is perfect for a directed graph, see
    https://www.cs.cmu.edu/afs/cs.cmu.edu/academic/class/15850-f20/www/notes/lec2.pdf.
    """

    # Check that graph is directed.
    if not conic_graph.directed:
        raise ValueError("Called MSTP for directed graphs on an undirected graph.")

    # Define variables.
    yv, zv, ye, ze, ze_tail, ze_head = define_variables(conic_graph, binary)

    # Enforce edge costs and constraints.
    cost, constraints = enforce_edge_programs(conic_graph, ye, ze, ze_tail, ze_head)

     # Enforce vertex costs and constraints.
    for i, vertex in enumerate(conic_graph.vertices):
        cost += vertex.cost_homogenization(zv[i], 1)

        # Constraints on incoming edges.
        inc = conic_graph.incoming_edge_indices(vertex)
        if vertex == conic_root:
            constraints += vertex.constraint_homogenization(zv[i], 1)
            constraints += [ye[k] == 0 for k in inc]
            constraints += [ze_head[k] == 0 for k in inc]
        else:
            constraints += [sum(ye[inc]) == 1, sum(ze_head[inc]) == zv[i]]

        # Constraints on outgoing edges.
        for k in conic_graph.outgoing_edge_indices(vertex):
            constraints += vertex.constraint_homogenization(zv[i] - ze_tail[k], 1 - ye[k])

    # Cutset constraints.
    if subtour_elimination:
        i = conic_graph.vertex_index(conic_root)
        subvertices = conic_graph.vertices[:i] + conic_graph.vertices[i+1:]
        for subtour_size in range(2, conic_graph.num_vertices()):
            for vertices in combinations(subvertices, subtour_size):
                inc = conic_graph.incoming_edge_indices(vertices)
                constraints.append(sum(ye[inc]) >= 1)

    # Solve problem and set solution.
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(**kwargs)
    if prob.status == "optimal":
        yv.value = np.ones(conic_graph.num_vertices())
    set_solution(conic_graph, prob, ye, ze, yv, zv, tol)

def minimum_spanning_tree(conic_graph, conic_root=None, subtour_elimination=True, binary=True, tol=1e-4, **kwargs):
    """
    Parameter root is ignored for undirected graphs.
    """
    if conic_graph.directed:
        directed_minimum_spanning_tree(conic_graph, conic_root, subtour_elimination, binary, tol, **kwargs)
    else:
        undirected_minimum_spanning_tree(conic_graph, subtour_elimination, binary, tol, **kwargs)
