import cvxpy as cp
from gcsopt.graph_problems.utils import define_variables, enforce_edge_programs, get_solution

def shortest_path(conic_graph, source, target, binary, tol, **kwargs):

    # define variables
    yv, zv, ye, ze, ze_tail, ze_head = define_variables(conic_graph, binary)

    # edge costs and constraints
    cost, constraints = enforce_edge_programs(conic_graph, ye, ze, ze_tail, ze_head)

    # shortest path constraints
    for i, vertex in enumerate(conic_graph.vertices):
        inc = conic_graph.incoming_edge_indices(vertex)
        out = conic_graph.outgoing_edge_indices(vertex)

        # source cost and constraints
        if vertex.name == source.name:
            cost += vertex.cost_homogenization(zv[i], 1)
            constraints += [yv[i] == 1, 1 == sum(ye[out]), zv[i] == sum(ze_tail[out])]
            for k in inc:
                constraints += [ye[k] == 0, ze_head[k] == 0]

        # target cost and constraints
        elif vertex.name == target.name:
            cost += vertex.cost_homogenization(zv[i], 1)
            constraints += [yv[i] == 1, 1 == sum(ye[inc]), zv[i] == sum(ze_head[inc])]
            for k in out:
                constraints += [ye[k] == 0, ze_tail[k] == 0]

        # cost and constraints on other vertices
        else:
            cost += vertex.cost_homogenization(zv[i], yv[i])
            constraints += [
                yv[i] <= 1,
                yv[i] == sum(ye[inc]),
                yv[i] == sum(ye[out]),
                zv[i] == sum(ze_head[inc]),
                zv[i] == sum(ze_tail[out])]
           
    # solve problem
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(**kwargs)

    return get_solution(conic_graph, prob, ye, ze, yv, zv, tol)
