from itertools import combinations
from gcspy.graph_problems.graph_problem import ConicGraphProblem

class ConicTravelingSalesmanProblem(ConicGraphProblem):

    def __init__(self, conic_graph, binary, subtour_elimination):

        # initialize parent class
        super().__init__(conic_graph, binary)

        # add all constraints one vertex at the time
        for i, v in enumerate(conic_graph.vertices):
            inc = conic_graph.incoming_edge_indices(v)
            out = conic_graph.outgoing_edge_indices(v)
            self.constraints += [
                self.yv[i] == 1,
                sum(self.ye[out]) == 1,
                sum(self.ye[inc]) == 1,
                self.zv[i] == self.xv[i],
                sum(self.ze_tail[out]) == self.xv[i],
                sum(self.ze_head[inc]) == self.xv[i]]

        # subtour elimination constraints for all subsets of vertices with
        # cardinality between 2 and num_vertices - 2
        if subtour_elimination:
            for size in range(2, conic_graph.num_vertices() - 1):
                for vertices in combinations(conic_graph.vertices, size):
                    out = conic_graph.outgoing_edge_indices(vertices)
                    self.constraints.append(sum(self.ye[out]) >= 1)