def facility_location(gcs, yv, ye, zv, ze_out, ze_inc, costumers, facilities):

    vertices = sorted(costumers + facilities)
    if vertices != list(range(gcs.num_vertices())):
        raise ValueError("Costumers and facilities do not for a partition of the vertices.")

    constraints = []
    for i, v in enumerate(gcs.vertices):
        if i in costumers:
            out_edges = gcs.outgoing_indices(v)
            constraints.append(yv[i] == 1)
            constraints.append(sum(ye[out_edges]) == 1)
            constraints.append(zv[i] == sum(ze_out[out_edges]))
        elif i in facilities:
            constraints.append(yv[i] <= 1)

    for k, edge in enumerate(gcs.edges):
        i = gcs.vertex_index(edge.head)
        constraints += edge.head.conic.eval_constraints(zv[i] - ze_inc[k], yv[i] - ye[k])

    return constraints
