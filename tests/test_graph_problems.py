import unittest
import numpy as np
import cvxpy as cp
from gcspy import GraphOfConicSets, GraphOfConvexSets
from itertools import product, combinations

class TestGraphProblems(unittest.TestCase):
    side = 3
    radius = .3
    binary = "MOSEK" in cp.installed_solvers() or "GUROBI" in cp.installed_solvers()

    def get_convex_graph(self):
        """
        Implemented in this form because using a setUp method was leading to a
        segmentation fault.
        """

        # Initialize graph.
        graph = GraphOfConvexSets()

        # Vertices.
        for i in range(self.side):
            for j in range(self.side):
                v = graph.add_vertex((i, j))
                x = v.add_variable(2)
                c = np.array([i, j])
                v.add_constraint(cp.norm2(x - c) <= self.radius)

        # Edges.
        for i, j, k, l in product(range(self.side), repeat=4):
            dx = abs(i - k)
            dy = abs(j - l)
            if dx <= 1 and dy <= 1 and dx + dy != 0:
                tail = graph.get_vertex((i, j))
                head = graph.get_vertex((k, l))
                edge = graph.add_edge(tail, head)
                edge.add_cost(cp.norm2(head.variables[0] - tail.variables[0]))

        return graph
    
    def get_conic_graph(self):
        """
        Implemented in this form because using a setUp method was leading to a
        segmentation fault.
        """

        # Initialize graph.
        conic_graph = GraphOfConicSets()

        # Vertices.
        vertex_size = 2
        A = np.eye(3)[:, 1:]
        K = (cp.SOC, 3)
        for i in range(self.side):
            for j in range(self.side):
                vertex = conic_graph.add_vertex((i, j), vertex_size)
                b = np.array([self.radius, -i, -j])
                vertex.add_constraint(A, b, K)

        # Edges.
        edge_size = 5
        c = np.array([0, 0, 0, 0, 1])
        d = 0
        A = np.array([
            [0, 0, 0, 0, 1],
            [-1, 0, 1, 0, 0],
            [0, -1, 0, 1, 0],
        ])
        b = np.zeros(3)
        K = (cp.SOC, 3)
        for i, j, k, l in product(range(self.side), repeat=4):
            dx = abs(i - k)
            dy = abs(j - l)
            if dx <= 1 and dy <= 1 and dx + dy != 0:
                tail = conic_graph.get_vertex((i, j))
                head = conic_graph.get_vertex((k, l))
                edge = conic_graph.add_edge(tail, head, edge_size)
                edge.add_cost(c, d)
                edge.add_constraint(A, b, K)

        return conic_graph

    def test_solve_shortest_path(self):

        # Repeat for convex and conic graph.
        graphs = [self.get_conic_graph(), self.get_convex_graph()]
        for graph in graphs:

            # Solve problem and check optimal value.
            source = graph.vertices[0]
            target = graph.vertices[-1]
            prob = graph.solve_shortest_path(source, target, binary=self.binary)
            expected_value = 2.2284271247532996
            self.assertAlmostEqual(prob.value, expected_value, places=4)

    def test_solve_shortest_path_from_ilp(self):

        # Repeat for convex and conic graph.
        graphs = [self.get_conic_graph(), self.get_convex_graph()]
        for graph in graphs:

            # Binary variables.
            yv = graph.vertex_binaries()
            ye = graph.edge_binaries()

            # Vertex constraints.
            source = graph.vertices[0]
            target = graph.vertices[-1]
            ilp_constraints = []
            for i, vertex in enumerate(graph.vertices):
                is_source = 1 if vertex == source else 0
                is_target = 1 if vertex == target else 0
                inc = graph.incoming_edge_indices(vertex)
                out = graph.outgoing_edge_indices(vertex)
                ilp_constraints += [
                    yv[i] == sum(ye[inc]) + is_source,
                    yv[i] == sum(ye[out]) + is_target]

            # Solve problem and check optimal value.
            prob = graph.solve_from_ilp(ilp_constraints, binary=self.binary)
            expected_value = 2.2284271247532996
            self.assertAlmostEqual(prob.value, expected_value, places=4)

    def test_solve_traveling_salesman(self):

        # Repeat for convex and conic graph.
        graphs = [self.get_conic_graph(), self.get_convex_graph()]
        for graph in graphs:

            # Solve problem and check optimal value.
            prob = graph.solve_traveling_salesman(binary=self.binary)
            expected_value = 6.718354360344848 if self.binary else 4.014213562373667
            self.assertAlmostEqual(prob.value, expected_value, places=4)

    def test_solve_traveling_salesman_from_ilp(self):
            
        # Repeat for convex and conic graph.
        graphs = [self.get_conic_graph(), self.get_convex_graph()]
        for graph in graphs:

            # Binary variables.
            yv = graph.vertex_binaries()
            ye = graph.edge_binaries()

            # Vertex constraints.
            ilp_constraints = []
            for i, vertex in enumerate(graph.vertices):
                inc = graph.incoming_edge_indices(vertex)
                out = graph.outgoing_edge_indices(vertex)
                ilp_constraints += [yv[i] == 1, sum(ye[out]) == 1, sum(ye[inc]) == 1]

            # Subtour elimnation constraints.
            for subtour_size in range(2, graph.num_vertices() - 1):
                for vertices in combinations(graph.vertices, subtour_size):
                    out = graph.outgoing_edge_indices(vertices)
                    ilp_constraints.append(sum(ye[out]) >= 1)

            # Solve problem and check optimal value.
            prob = graph.solve_from_ilp(ilp_constraints, binary=self.binary)
            expected_value = 6.718354360344848 if self.binary else 4.014213562373667
            self.assertAlmostEqual(prob.value, expected_value, places=4)

    def test_solve_spanning_tree(self):
        
        # Repeat for convex and conic graph.
        graphs = [self.get_conic_graph(), self.get_convex_graph()]
        for graph in graphs:
        
            # Solve problem and check optimal value.
            root = graph.vertices[0]
            prob = graph.solve_minimum_spanning_tree(root, binary=self.binary)
            expected_value = 5.273360961108411 if self.binary else 3.2
            self.assertAlmostEqual(prob.value, expected_value, places=4)

    def test_solve_spanning_tree_from_ilp(self):
            
        # Repeat for convex and conic graph.
        graphs = [self.get_conic_graph(), self.get_convex_graph()]
        for graph in graphs:

            # Binary variables.
            yv = graph.vertex_binaries()
            ye = graph.edge_binaries()

            # Vertex constraints.
            root = graph.vertices[0]
            ilp_constraints = []
            root_index = graph.vertex_index(root)
            for i, vertex in enumerate(graph.vertices):
                inc = graph.incoming_edge_indices(vertex)
                inc_flow = 0 if i == root_index else 1
                ilp_constraints += [yv[i] == 1, sum(ye[inc]) == inc_flow]

            # Subtour elimnation constraints.
            for subtour_size in range(2, graph.num_vertices()):
                for vertices in combinations(graph.vertices[1:], subtour_size):
                    inc = graph.incoming_edge_indices(vertices)
                    ilp_constraints.append(sum(ye[inc]) >= 1)

            # Solve problem and check optimal value.
            prob = graph.solve_from_ilp(ilp_constraints, binary=self.binary)
            expected_value = 5.273360961108411 if self.binary else 3.2
            self.assertAlmostEqual(prob.value, expected_value, places=4)

if __name__ == '__main__':
    unittest.main()
