from gcspy.graph_problems.graph_problem import ConicGraphProblem

class ConicFacilityLocationProblem(ConicGraphProblem):

    def __init__(self, conic_graph, binary):

        # initialize parent class
        super().__init__(conic_graph, binary)

        # constraints on the vertices
        for i, vertex in enumerate(conic_graph.vertices):
            inc = conic_graph.incoming_indices(vertex)
            out = conic_graph.outgoing_indices(vertex)

            # user vertex
            if len(inc) > 0:
                if len(out) > 0:
                    raise ValueError("Graph does not have facility-location topology.")
                self.constraints.append(self.yv[i] == 1)
                self.constraints.append(sum(self.ye[inc]) == 1)
                self.constraints.append(self.zv[i] == sum(self.ze_head[inc]))
                self.constraints.append(self.zv[i] == self.xv[i])

        # constraints on the edges
        for k, edge in enumerate(conic_graph.edges):
            i = conic_graph.vertex_index(edge.tail)
            self.constraints.append(self.yv[i] >= self.ye[k])
            self.constraints += edge.tail.evaluate_constraints(self.zv[i] - self.ze_tail[k], self.yv[i] - self.ye[k])