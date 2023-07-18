from itertools import combinations


def traveling_salesman(gcs, xv, zv, ze_out, ze_inc):

    yv = gcs.vertex_binaries()
    ye = gcs.edge_binaries()

    constraints = []
    for i, v in enumerate(gcs.vertices):
        inc_edges = gcs.incoming_indices(v)
        out_edges = gcs.outgoing_indices(v)

        constraints.append(yv[i] == 1)
        constraints.append(sum(ye[out_edges]) == 1)
        constraints.append(sum(ye[inc_edges]) == 1)
        constraints.append(zv[i] == sum(ze_out[out_edges]))
        constraints.append(zv[i] == sum(ze_inc[inc_edges]))
        constraints.append(zv[i] == xv[i])

    for r in range(2, gcs.num_vertices() - 1):
        for vertices in combinations(gcs.vertices, r):
            out_edges = gcs.outgoing_indices(vertices)
            constraints.append(sum(ye[out_edges]) >= 1)

    return constraints
