from itertools import combinations
from gcspy.graph_problems.graph_problem import ConicGraphProblem

class ConicTravelingSalesmanProblem(ConicGraphProblem):

    def __init__(self, conic_graph, subtour_elimination):

        # initialize parent class
        super().__init__(conic_graph)

        # add all constraints one vertex at the time
        for i, v in enumerate(conic_graph.vertices):
            inc = conic_graph.incoming_indices(v)
            out = conic_graph.outgoing_indices(v)

            self.constraints += [
                self.yv[i] == 1,
                sum(self.ye[out]) == 1,
                sum(self.ye[inc]) == 1,
                self.zv[i] == self.xv[i],
                sum(self.ze_tail[out]) == self.xv[i],
                sum(self.ze_head[inc]) == self.xv[i]]

        if subtour_elimination:
            for r in range(2, conic_graph.num_vertices() - 1):
                for vertices in combinations(conic_graph.vertices, r):
                    out = conic_graph.outgoing_indices(vertices)
                    self.constraints.append(sum(self.ye[out]) >= 1)