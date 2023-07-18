def shortest_path(gcs, xv, zv, ze_out, ze_inc, s, t):

    yv = gcs.vertex_binaries()
    ye = gcs.edge_binaries()

    constraints = []
    for i, vertex in enumerate(gcs.vertices):
        inc_edges = gcs.incoming_indices(vertex)
        out_edges = gcs.outgoing_indices(vertex)
        
        if vertex == s:
            constraints.append(sum(ye[inc_edges]) == 0)
            constraints.append(sum(ye[out_edges]) == 1)
            constraints.append(yv[i] == sum(ye[out_edges]))
            constraints.append(zv[i] == sum(ze_out[out_edges]))
            constraints.append(zv[i] == xv[i])
            
        elif vertex == t:
            constraints.append(sum(ye[out_edges]) == 0)
            constraints.append(sum(ye[inc_edges]) == 1)
            constraints.append(yv[i] == sum(ye[inc_edges]))
            constraints.append(zv[i] == sum(ze_inc[inc_edges]))
            constraints.append(zv[i] == xv[i])
            
        else:
            constraints.append(yv[i] == sum(ye[out_edges]))
            constraints.append(yv[i] == sum(ye[inc_edges]))
            constraints.append(yv[i] <= 1)
            constraints.append(zv[i] == sum(ze_out[out_edges]))
            constraints.append(zv[i] == sum(ze_inc[inc_edges]))
            constraints += vertex.conic.eval_constraints(xv[i] - zv[i], 1 - yv[i])
            
    return constraints
