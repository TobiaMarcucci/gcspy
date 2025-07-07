import cvxpy as cp
import numpy as np
from gcspy.graph_problems.graph_problem import ConicGraphProblem
from gcspy.programs import ConvexProgram

class ConicGraphProblemFromILP(ConicGraphProblem):

    def __init__(self, conic_graph, binary, yv, ye, ilp_constraints):

        # initialize parent class
        super().__init__(conic_graph, binary)

        # put given constraints in conic form
        convex_ilp = ConvexProgram()
        convex_ilp.variables = [y for y in yv] + [y for y in ye]
        convex_ilp.add_constraints(ilp_constraints)
        conic_ilp = convex_ilp.to_conic()

        # columns in conic program for vertex and edge binaries
        columns_v = [conic_ilp.convex_id_to_conic_idx[y.id].start for y in yv]
        columns_e = [conic_ilp.convex_id_to_conic_idx[y.id].start for y in ye]

        # check each line of each conic constraint
        for A, b, K in zip(conic_ilp.A, conic_ilp.b, conic_ilp.K):
            for Aj, bj in zip(A, b):

                # express linear constraint using the problem binaries
                av = Aj[columns_v]
                ae = Aj[columns_e]
                lhs = av @ self.yv + ae @ self.ye + bj

                # if there are no shared vertices just enforce the scalar constraint
                shared_vertices = self.find_shared_vertices(av, ae)
                if not shared_vertices:
                    constraint = K(lhs)
                    self.constraints.append(constraint)

                # apply Lemma 5.1 from thesis to each linear constraint with
                # shared vertices
                for vertex in shared_vertices:

                    # assemble spatial constraints
                    i = conic_graph.vertex_index(vertex)
                    inc = conic_graph.incoming_edge_indices(vertex)
                    out = conic_graph.outgoing_edge_indices(vertex)
                    spatial_lhs = bj * self.xv[i] + av[i] * self.zv[i]
                    spatial_lhs += sum(ae[inc] * self.ze_head[inc])
                    spatial_lhs += sum(ae[out] * self.ze_tail[out])

                    # if equality constraints
                    if K == cp.Zero:
                        self.constraints += [lhs == 0, spatial_lhs == 0]
                    elif K == cp.NonNeg:
                        self.constraints += vertex.evaluate_constraints(spatial_lhs, lhs)
                    elif K == cp.NonPos:
                        self.constraints += vertex.evaluate_constraints(-spatial_lhs, -lhs)
                    else:
                        raise ValueError(
                            f"Got cone of type {type(K)}. "
                            "All the constraints of the ILP must be linear.")
    
    def find_shared_vertices(self, av, ae):
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

        # if av has more than one nonzero entry, the decomposition is impossible
        nonzero_vertices = [self.conic_graph.vertices[i] for i in np.nonzero(av)[0]]
        if len(nonzero_vertices) > 1:
            return []
        
        # if av is zero, return all the vertices shared by the edges
        edges = [self.conic_graph.edges[k] for k in np.nonzero(ae)[0]]
        tails_and_heads = [{edge.tail, edge.head} for edge in edges]
        shared_vertices = set.intersection(*tails_and_heads) if tails_and_heads else set()
        if len(nonzero_vertices) == 0:
            return list(shared_vertices)
        
        # if av has one nonzero entry, there is only one candidate vertex
        if set(nonzero_vertices) >= shared_vertices:
            return nonzero_vertices
        else:
            return []