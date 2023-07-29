def facility_location(gcs, xv, zv, ze_out, ze_inc):

    yv = gcs.vertex_binaries()
    ye = gcs.edge_binaries()

    constraints = []
    for i, vertex in enumerate(gcs.vertices):
        inc_edges = gcs.incoming_indices(vertex)
        if len(inc_edges) == 0: # facility
            constraints.append(yv[i] <= 1)
            constraints += vertex.conic.eval_constraints(xv[i] - zv[i], 1 - yv[i])
        else: # user
            out_edges = gcs.outgoing_indices(vertex)
            if len(out_edges) > 0:
                raise ValueError("Graph does not have facility-location topology.")
            constraints.append(yv[i] == 1)
            constraints.append(sum(ye[inc_edges]) == 1)
            constraints.append(zv[i] == sum(ze_inc[inc_edges]))
            constraints.append(zv[i] == xv[i])

    for k, edge in enumerate(gcs.edges):
        i = gcs.vertex_index(edge.tail)
        constraints.append(yv[i] >= ye[k])
        constraints += edge.tail.conic.eval_constraints(zv[i] - ze_out[k], yv[i] - ye[k])

    return constraints
