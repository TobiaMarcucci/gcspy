from gcspy.graph_problems.graph_problem import ConicGraphProblem

class ConicShortestPathProblem(ConicGraphProblem):

    def __init__(self, conic_graph, source_name, target_name):

        # initialize parent class
        super().__init__(conic_graph)
            
        # add all constraints one vertex at the time
        for i, vertex in enumerate(conic_graph.vertices):
            inc = conic_graph.incoming_indices(vertex)
            out = conic_graph.outgoing_indices(vertex)

            # source constraints
            if vertex.name == source_name:
                self.constraints += [
                    self.yv[i] == 1,
                    sum(self.ye[inc]) == 0,
                    sum(self.ye[out]) == 1,
                    self.zv[i] == self.xv[i],
                    self.zv[i] == sum(self.ze_tail[out])]

            # target constraints
            elif vertex.name == target_name:
                self.constraints += [
                    self.yv[i] == 1,
                    sum(self.ye[inc]) == 1,
                    sum(self.ye[out]) == 0,
                    self.zv[i] == self.xv[i],
                    self.zv[i] == sum(self.ze_head[inc])]

            # all other vertices constraints
            else:
                self.constraints += [
                    self.yv[i] == sum(self.ye[inc]),
                    self.yv[i] == sum(self.ye[out]),
                    self.zv[i] == sum(self.ze_tail[out]),
                    self.zv[i] == sum(self.ze_head[inc])]