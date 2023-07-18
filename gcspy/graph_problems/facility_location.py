def facility_location(gcs, xv, zv, ze_out, ze_inc, costumers, facilities):

    yv = gcs.vertex_binaries()
    ye = gcs.edge_binaries()

    vertices = sorted(costumers + facilities)
    if vertices != list(range(gcs.num_vertices())):
        raise ValueError("Costumers and facilities do not for a partition of the vertices.")

    constraints = []
    for i, vertex in enumerate(gcs.vertices):
        if i in facilities:
            constraints.append(yv[i] <= 1)
            constraints += vertex.conic.eval_constraints(xv[i] - zv[i], 1 - yv[i])
        elif i in costumers:
            constraints.append(yv[i] == 1)
            out_edges = gcs.outgoing_indices(vertex)
            constraints.append(sum(ye[out_edges]) == 1)
            constraints.append(zv[i] == sum(ze_out[out_edges]))
            constraints.append(zv[i] == xv[i])

    for k, edge in enumerate(gcs.edges):
        i = gcs.vertex_index(edge.head)
        constraints.append(yv[i] >= ye[k])
        constraints += edge.head.conic.eval_constraints(zv[i] - ze_inc[k], yv[i] - ye[k])

    return constraints
