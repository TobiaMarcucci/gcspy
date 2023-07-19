def facility_location(gcs, xv, zv, ze_out, ze_inc, facilities):

    yv = gcs.vertex_binaries()
    ye = gcs.edge_binaries()

    customers = [i for i in range(gcs.num_vertices()) if i not in facilities]
    constraints = []
    for i, vertex in enumerate(gcs.vertices):
        if i in facilities:
            constraints.append(yv[i] <= 1)
            constraints += vertex.conic.eval_constraints(xv[i] - zv[i], 1 - yv[i])
        elif i in customers:
            constraints.append(yv[i] == 1)
            inc_edges = gcs.incoming_indices(vertex)
            constraints.append(sum(ye[inc_edges]) == 1)
            constraints.append(zv[i] == sum(ze_inc[inc_edges]))
            constraints.append(zv[i] == xv[i])

    for k, edge in enumerate(gcs.edges):
        i = gcs.vertex_index(edge.tail)
        constraints.append(yv[i] >= ye[k])
        constraints += edge.tail.conic.eval_constraints(zv[i] - ze_out[k], yv[i] - ye[k])

    return constraints
