from gcspy.graph_problems.graph_problem import convex_graph_problem

def shortest_path_constraints(conic_graph, xv, zv, ze_tail, ze_head, source, target):

    # binary variables
    yv = conic_graph.vertex_binaries()
    ye = conic_graph.edge_binaries()

    # add all constraints one vertex at the time
    constraints = []
    for i, vertex in enumerate(conic_graph.vertices):
        inc = conic_graph.incoming_indices(vertex)
        out = conic_graph.outgoing_indices(vertex)

        # source constraints
        if vertex.name == source.name:
            constraints += [
                yv[i] == 1,
                sum(ye[inc]) == 0,
                sum(ye[out]) == 1,
                zv[i] == xv[i],
                zv[i] == sum(ze_tail[out])]

        # target constraints
        elif vertex.name == target.name:
            constraints += [
                yv[i] == 1,
                sum(ye[inc]) == 1,
                sum(ye[out]) == 0,
                zv[i] == xv[i],
                zv[i] == sum(ze_head[inc])]

        # all other vertices constraints
        else:
            constraints += [
                yv[i] == sum(ye[inc]),
                yv[i] == sum(ye[out]),
                zv[i] == sum(ze_tail[out]),
                zv[i] == sum(ze_head[inc])]
            
    return constraints

def solve_shortest_path(convex_graph, source, target, binary=True, **kwargs):
    additional_constraints = lambda *args: shortest_path_constraints(*args, source, target)
    return convex_graph_problem(convex_graph, additional_constraints, binary, **kwargs)
