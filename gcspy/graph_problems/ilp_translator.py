import cvxpy as cp
import numpy as np
from gcspy.programs import ConicProgram


def find_common_vertices(gcs, a_v, a_e):
    nonzero_v = np.where(a_v != 0)[0]
    nonzero_e = np.where(a_e != 0)[0]
    if len(nonzero_v) > 1:
        return []
    elif len(nonzero_v) == 1:
        i = nonzero_v[0]
        v = gcs.vertices[i]
        incident_edges = gcs.incident_indices(v)
        if set(nonzero_e) > set(incident_edges):
            return []
        else:
            return [v]
    else:
        edges = [gcs.edges[k] for k in nonzero_e]
        spanned_vertices = [{edge.tail, edge.head} for edge in edges]
        common_vertices = set.intersection(*spanned_vertices)
        return list(common_vertices)

        
def ilp_translator(gcs, xv, zv, ze_out, ze_inc, ilp_constraints):

    # put given constraints in conic form
    ilp = ConicProgram(ilp_constraints, 0)

    # binary variables and corresponding columns in conic program
    yv = gcs.vertex_binaries()
    ye = gcs.edge_binaries()
    columns_v = [i for y in yv for i in ilp.columns[y.id]]
    columns_e = [i for y in ye for i in ilp.columns[y.id]]

    # check each line of each conic constraint
    constraints = []
    for Ai, bi, Ki in zip(ilp.A, ilp.b, ilp.K):
        for Aij, bij in zip(Ai, bi):
            a_v = Aij[columns_v]
            a_e = Aij[columns_e]

            # verify if the constraint have common vertices
            common_vertices = find_common_vertices(gcs, a_v, a_e)
            if len(common_vertices) == 0:
                continue
            for vertex in common_vertices:
                i = gcs.vertex_index(vertex)
                inc_edges = gcs.incoming_indices(vertex)
                out_edges = gcs.outgoing_indices(vertex)

                # assemble spatial constraint
                spatial_lhs = bij * xv[i] + a_v[i] * zv[i]
                spatial_lhs += sum(a_e[k] * ze_inc[k] for k in inc_edges)
                spatial_lhs += sum(a_e[k] * ze_out[k] for k in out_edges)

                # enforce equality constraints
                if Ki == cp.Zero:
                    constraints.append(spatial_lhs == 0)
                    continue

                # reassemble scalar constraint
                lhs = bij + a_v[i] * yv[i]
                lhs += sum(a_e[k] * ye[k] for k in inc_edges + out_edges)
                if Ki == cp.NonNeg:
                    constraints += vertex.conic.eval_constraints(spatial_lhs, lhs)
                elif Ki == cp.NonPos:
                    constraints += vertex.conic.eval_constraints(-spatial_lhs, -lhs)
                else:
                    raise ValueError(f"ILP constraints must be linear.")

    return constraints + ilp_constraints
