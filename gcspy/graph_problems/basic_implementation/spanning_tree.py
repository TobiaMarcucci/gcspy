from itertools import combinations
from gcspy.graph_problems.graph_problem import ConicGraphProblem

class ConicSpanningTreeProblem(ConicGraphProblem):

    def __init__(self, conic_graph, root_name, subtour_elimination, binary):

        # initialize parent class
        super().__init__(conic_graph, binary)

        # constraints on the vertices
        for i, vertex in enumerate(conic_graph.vertices):
            inc = conic_graph.incoming_edge_indices(vertex)
            self.constraints.append(self.yv[i] == 1)
            self.constraints.append(self.zv[i] == self.xv[i])
            if vertex.name == root_name:
                self.constraints.append(sum(self.ye[inc]) == 0)
                self.constraints.append(sum(self.ze_head[inc]) == 0)
            else:
                self.constraints.append(sum(self.ye[inc]) == 1)
                self.constraints.append(sum(self.ze_head[inc]) == self.xv[i])

        # subtour elimination constraints for all subsets of vertices with
        # cardinality between 2 and num_vertices - 1
        if subtour_elimination:
            root = conic_graph.get_vertex(root_name)
            i = conic_graph.vertex_index(root)
            subvertices = conic_graph.vertices[:i] + conic_graph.vertices[i+1:]
            for subtour_size in range(2, conic_graph.num_vertices()):
                for vertices in combinations(subvertices, subtour_size):
                    inc = conic_graph.incoming_edge_indices(vertices)
                    self.constraints.append(sum(self.ye[inc]) >= 1)

        