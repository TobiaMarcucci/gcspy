from itertools import combinations


def spanning_tree(gcs, xv, zv, ze_out, ze_inc, root):

    yv = gcs.vertex_binaries()
    ye = gcs.edge_binaries()

    constraints = []
    for i, vertex in enumerate(gcs.vertices):
        constraints.append(yv[i] == 1)
        constraints.append(zv[i] == xv[i])
        inc_edges = gcs.incoming_indices(vertex)
        if vertex == root:
            constraints.append(sum(ye[inc_edges]) == 0)
            constraints.append(sum(ze_inc[inc_edges]) == 0)
        else:
            constraints.append(sum(ye[inc_edges]) == 1)
            constraints.append(sum(ze_inc[inc_edges]) == xv[i])

    i = gcs.vertex_index(root)
    for r in range(2, gcs.num_vertices()):
        for vertices in combinations(gcs.vertices[:i] + gcs.vertices[i+1:], r):
            inc_edges = gcs.incoming_indices(vertices)
            constraints.append(sum(ye[inc_edges]) >= 1)

    for k, edge in enumerate(gcs.edges):
        i = gcs.vertex_index(edge.tail)
        constraints.append(ye[k] <= 1)
        constraints += edge.tail.conic.eval_constraints(xv[i] - ze_out[k], 1 - ye[k])

    return constraints
