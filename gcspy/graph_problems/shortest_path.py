def shortest_path(gcs, yv, ye, zv, ze_out, ze_inc, s, t):

    constraints = []
    for i, v in enumerate(gcs.vertices):
        inc_edges = gcs.incoming_indices(v)
        out_edges = gcs.outgoing_indices(v)
        
        if v == s:
            constraints.append(sum(ye[inc_edges]) == 0)
            constraints.append(sum(ye[out_edges]) == 1)
            constraints.append(yv[i] == sum(ye[out_edges]))
            constraints.append(zv[i] == sum(ze_out[out_edges]))
            
        elif v == t:
            constraints.append(sum(ye[out_edges]) == 0)
            constraints.append(sum(ye[inc_edges]) == 1)
            constraints.append(yv[i] == sum(ye[inc_edges]))
            constraints.append(zv[i] == sum(ze_inc[inc_edges]))
            
        else:
            constraints.append(yv[i] == sum(ye[out_edges]))
            constraints.append(yv[i] == sum(ye[inc_edges]))
            constraints.append(yv[i] <= 1)
            constraints.append(zv[i] == sum(ze_out[out_edges]))
            constraints.append(zv[i] == sum(ze_inc[inc_edges]))
            
    return constraints
