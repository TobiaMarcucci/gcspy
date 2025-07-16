import cvxpy as cp
import numpy as np
from gcspy.programs import ConvexProgram
from gcspy.graph_problems.utils import define_variables, get_solution

def from_ilp(conic_graph, convex_yv, convex_ye, ilp_constraints, binary, tol, **kwargs):

    # put given constraints in conic form
    convex_ilp = ConvexProgram()
    # next line is not elegant but I cannot use convex_ilp.add_variables
    convex_ilp.variables = [y for y in convex_yv] + [y for y in convex_ye]
    convex_ilp.add_constraints(ilp_constraints)
    conic_ilp = convex_ilp.to_conic()

    # indices of vertex and edge binaries in conic program, uses the fact
    # that the variables are scalars
    idx_v = [conic_ilp.convex_id_to_conic_idx[y.id].start for y in convex_yv]
    idx_e = [conic_ilp.convex_id_to_conic_idx[y.id].start for y in convex_ye]

    # define variables
    yv, zv, ye, ze, ze_tail, ze_head = define_variables(conic_graph, binary)
    xv = np.array([cp.Variable(vertex.size) for vertex in conic_graph.vertices])

    # vertex costs and constraints
    cost = 0
    constraints = []
    for i, vertex in enumerate(conic_graph.vertices):
        cost += vertex.evaluate_cost(zv[i], yv[i])
        constraints += vertex.evaluate_constraints(zv[i], yv[i])
        constraints += vertex.evaluate_constraints(xv[i] - zv[i], 1 - yv[i])

    # edge costs and constraints
    for k, edge in enumerate(conic_graph.edges):
        cost += edge.evaluate_cost(ze_tail[k], ze_head[k], ze[k], ye[k])
        constraints += edge.evaluate_constraints(ze_tail[k], ze_head[k], ze[k], ye[k])

        # tail constraints
        x_tail = xv[conic_graph.vertex_index(edge.tail)]
        constraints += edge.tail.evaluate_constraints(ze_tail[k], ye[k])
        constraints += edge.tail.evaluate_constraints(x_tail - ze_tail[k], 1 - ye[k])

        # head constraints
        x_head = xv[conic_graph.vertex_index(edge.head)]
        constraints += edge.head.evaluate_constraints(ze_head[k], ye[k])
        constraints += edge.head.evaluate_constraints(x_head - ze_head[k], 1 - ye[k])

    # check each line of each conic constraint
    for A, b, K in zip(conic_ilp.A, conic_ilp.b, conic_ilp.K):
        for Aj, bj in zip(A, b):

            # evaluate linear constraint using the problem binaries
            av = Aj[idx_v]
            ae = Aj[idx_e]
            lhs = av @ yv + ae @ ye + bj

            # if there are no shared vertices just enforce the scalar constraint
            shared_vertices = find_shared_vertices(conic_graph, av, ae)
            if not shared_vertices:
                constraints.append(K(lhs))

            # if there are shared vertices, apply Lemma 5.1 from thesis to
            # each linear constraint
            for vertex in shared_vertices:

                # assemble spatial constraints
                i = conic_graph.vertex_index(vertex)
                inc = conic_graph.incoming_edge_indices(vertex)
                out = conic_graph.outgoing_edge_indices(vertex)
                vector_lhs = bj * xv[i] + av[i] * zv[i]
                vector_lhs += sum(ae[inc] * ze_head[inc])
                vector_lhs += sum(ae[out] * ze_tail[out])

                # enforce implied constraints
                if K == cp.Zero:
                    constraints += [lhs == 0, vector_lhs == 0]
                elif K == cp.NonNeg:
                    constraints += vertex.evaluate_constraints(vector_lhs, lhs)
                elif K == cp.NonPos:
                    constraints += vertex.evaluate_constraints(-vector_lhs, -lhs)
                else:
                    raise ValueError(
                        f"Got cone of type {type(K)}. All the constraints "
                        "of the ILP must be linear.")
                
    # solve problem
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(**kwargs)

    return get_solution(conic_graph, prob, ye, ze, yv, zv, tol)
    
def find_shared_vertices(conic_graph, av, ae):
    """
    Checks if the linear function
    av^T y_v + ae^T y_e
    is amenable to the lemma that is used to generate spatial constraints
    (see Lemma 5.1 from thesis). To this end, it should be possible to rewrite
    the linear function above as
    b y_v_i + sum_{k in edges_incident_i} c_k y_e_k
    where i is the index of a vertex and k is the index of an edge incident with
    the ith vertex. This function returns all the values of i that allow such
    a decomposition.
    """

    # extract nonzero vertices and edges
    nonzero_vertices = [conic_graph.vertices[i] for i in np.nonzero(av)[0]]
    nonzero_edges = [conic_graph.edges[k] for k in np.nonzero(ae)[0]]

    # compute shared vertices among all edges
    tails_and_heads = [{edge.tail, edge.head} for edge in nonzero_edges]
    shared_vertices = set.intersection(*tails_and_heads) if tails_and_heads else set()

    # if av is zero, return all the vertices shared by the edges
    if not nonzero_vertices:
        return list(shared_vertices)
    
    # if av has one nonzero entry, there is only one candidate vertex, and
    # no other shared vertex can be different from it
    elif len(nonzero_vertices) == 1 and shared_vertices <= set(nonzero_vertices):
        return nonzero_vertices

    # in all other cases, the decomposition is impossible
    else:
        return []
