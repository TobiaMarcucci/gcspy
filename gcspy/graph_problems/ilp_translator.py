import cvxpy as cp
import numpy as np
from gcspy.programs import ConicProgram


def find_common_vertex(gcs, a_v, a_e):
    nonzero_v = np.where(a_v != 0)[0]
    nonzero_e = np.where(a_e != 0)[0]
    if len(nonzero_v) > 1:
        return
    elif len(nonzero_v) == 1:
        i = nonzero_v[0]
        v = gcs.vertices[i]
        incident_edges = gcs.incident_indices(v)
        if set(nonzero_e) > set(incident_edges):
            return
        else:
            return v
    else:
        edges = [gcs.edges[k] for k in nonzero_e]
        spanned_vertices = [{edge.tail, edge.head} for edge in edges]
        common_vertices = set.intersection(*spanned_vertices)
        if len(common_vertices) == 1:
            v = next(iter(common_vertices))
            return v

        
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

            # verify if the constraint has a common vertex
            vertex = find_common_vertex(gcs, a_v, a_e)
            if vertex is None:
                continue
            i = gcs.vertex_index(vertex)
            inc_edges = gcs.incoming_indices(vertex)
            out_edges = gcs.outgoing_indices(vertex)

            # (re)assemble scalar constraint
            lhs = a_v[i] * yv[i]
            lhs += sum(a_e[k] * ye[k] for k in inc_edges + out_edges)
            lhs += bij

            # assemble spatial constraint
            spatial_lhs = a_v[i] * zv[i]
            spatial_lhs += sum(a_e[k] * ze_inc[k] for k in inc_edges)
            spatial_lhs += sum(a_e[k] * ze_out[k] for k in out_edges)
            spatial_lhs += bij * xv[i]

            # enforce new constraints
            if Ki == cp.Zero:
                constraints.append(lhs == 0)
                constraints.append(spatial_lhs == 0)
            elif Ki == cp.NonNeg:
                constraints.append(lhs >= 0)
                constraints += vertex.conic.eval_constraints(spatial_lhs, lhs)
            elif Ki == cp.NonPos:
                constraints.append(lhs <= 0)
                constraints += vertex.conic.eval_constraints(-spatial_lhs, -lhs)
            else:
                raise ValueError(f"ILP constraints must be linear.")

    return constraints
