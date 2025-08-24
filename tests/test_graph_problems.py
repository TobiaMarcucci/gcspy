import unittest
import numpy as np
import cvxpy as cp
from gcsopt import GraphOfConicSets, GraphOfConvexSets
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
            graph.solve_shortest_path(source, target, binary=self.binary)
            expected_value = 2.2284271247532996
            self.assertAlmostEqual(graph.value, expected_value, places=4)

        # Problem where vertex variables have different dimensions.
        graph = GraphOfConvexSets()

        # 1d vertex.
        v0 = graph.add_vertex(0)
        x0 = v0.add_variable(1)
        v0.add_constraints([x0 >= 0, x0 <= 2])
        z0 = cp.hstack([x0, 0])

        # 2d vertex.
        v1 = graph.add_vertex(1)
        z1 = v1.add_variable(2)
        v1.add_constraints([z1 >= 1, z1 <= 3])

        # 1d vertex.
        v2 = graph.add_vertex(2)
        x2 = v2.add_variable(1)
        v2.add_constraints([x2 >= 2, x2 <= 4])
        z2 = cp.hstack([x2, 4])

        # Two edges.
        e0 = graph.add_edge(v0, v1)
        e1 = graph.add_edge(v1, v2)
        e0.add_cost(cp.sum_squares(z1 - z0))
        e1.add_cost(cp.sum_squares(z2 - z1))

        # Solve problem and check result.
        graph.solve_shortest_path(v0, v2, binary=self.binary)
        self.assertAlmostEqual(graph.value, 8, places=4)
        self.assertAlmostEqual(x0.value[0], 2, places=3)
        self.assertAlmostEqual(x2.value[0], 2, places=3)
        np.testing.assert_array_almost_equal(z1.value, [2, 2], decimal=3)
        for yv in graph.vertex_binaries():
            self.assertAlmostEqual(yv.value, 1, places=4)
        for ye in graph.edge_binaries():
            self.assertAlmostEqual(ye.value, 1, places=4)

    def test_solve_infeasible_shortest_path(self):

        # Initialize empty directed graph.
        graph = GraphOfConvexSets(directed=True)

        # Add source vertex with circular set.
        s = graph.add_vertex("s")
        xs = s.add_variable(2)
        cs = [-2, 0] # Center of the source circle.
        s.add_constraint(cp.norm2(xs - cs) <= 1)

        # Add target vertex with circular set.
        t = graph.add_vertex("t")
        xt = t.add_variable(2)
        ct = [2, 0] # Center of the target circle.
        t.add_constraint(cp.norm2(xt - ct) <= 1)

        # Add edge from source to target.
        e = graph.add_edge(s, t)
        e.add_constraint(xt == xs)

        # Solve shortest path problem from source to target.
        graph.solve_shortest_path(s, t, binary=self.binary)

        # Check solution.
        self.assertTrue(np.isinf(graph.value))
        self.assertIsNone(xs.value)
        self.assertIsNone(xt.value)

    def test_solve_shortest_path_without_edge_costs(self):

        # Initialize graph.
        graph = GraphOfConvexSets()

        # Add vertices, variables, and costs.
        v = [graph.add_vertex(i) for i in range(4)]
        x = [vi.add_variable((2, 2)) for vi in v]
        for vi, xi in zip(v, x):
            vi.add_cost(cp.norm2(xi[1] - xi[0]))

        # Add vertex constraints.
        def constrain_in_box(v, x, l, u):
            for xi in x:
                v.add_constraints([xi >= l, xi <= u])
        constrain_in_box(v[0], x[0], [0, 0], [3, 3])
        constrain_in_box(v[1], x[1], [2, 2.1], [5, 5.1])
        constrain_in_box(v[2], x[2], [2, -2], [5, 1])
        constrain_in_box(v[3], x[3], [4, 0], [7, 3])

        # Add start and goal points.
        v[0].add_constraints([x[0][0] == [1.5, 1.5]])
        v[3].add_constraints([x[3][1] == [5.5, 1.5]])

        # Add edges.
        edges = [graph.add_edge(v[0], v[1]),
                 graph.add_edge(v[0], v[2]),
                 graph.add_edge(v[1], v[3]),
                 graph.add_edge(v[2], v[3])]

        # Add edge constraints.
        for e in edges:
            end_tail = e.tail.variables[0][1]
            start_head = e.head.variables[0][0]
            e.add_constraint(end_tail == start_head) 

        # Solve problem relaxation (which for this problem is exact).
        graph.solve_shortest_path(v[0], v[3], binary=False)
        
        # Check optimal value.
        expected_value = 2 * np.sqrt(1.5 ** 2 + .5 ** 2) + 1
        self.assertAlmostEqual(graph.value, expected_value, places=4)

        # Check optimal solution.
        x0 = np.array([[1.5, 1.5], [3, 1]])
        x2 = np.array([[3, 1], [4, 1]])
        x3 = np.array([[4, 1], [5.5, 1.5]])
        np.testing.assert_array_almost_equal(x[0].value, x0, decimal=4)
        np.testing.assert_array_almost_equal(x[2].value, x2, decimal=4)
        np.testing.assert_array_almost_equal(x[3].value, x3, decimal=4)
        self.assertIsNone(x[1].value)

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
            graph.solve_from_ilp(ilp_constraints, binary=self.binary)
            expected_value = 2.2284271247532996
            self.assertAlmostEqual(graph.value, expected_value, places=4)

    def test_solve_traveling_salesman(self):

        # Repeat for convex and conic graph.
        graphs = [self.get_conic_graph(), self.get_convex_graph()]
        for graph in graphs:

            # Solve problem and check optimal value.
            graph.solve_traveling_salesman(binary=self.binary)
            expected_value = 6.718354360344848 if self.binary else 4.014213562373667
            self.assertAlmostEqual(graph.value, expected_value, places=4)

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
            graph.solve_from_ilp(ilp_constraints, binary=self.binary)
            expected_value = 6.718354360344848 if self.binary else 4.014213562373667
            self.assertAlmostEqual(graph.value, expected_value, places=4)

    def test_solve_spanning_tree(self):
        
        # Repeat for convex and conic graph.
        graphs = [self.get_conic_graph(), self.get_convex_graph()]
        for graph in graphs:
        
            # Solve problem and check optimal value.
            root = graph.vertices[0]
            graph.solve_minimum_spanning_tree(root, binary=self.binary)
            expected_value = 5.273360961108411 if self.binary else 3.2
            self.assertAlmostEqual(graph.value, expected_value, places=4)

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
            graph.solve_from_ilp(ilp_constraints, binary=self.binary)
            expected_value = 5.273360961108411 if self.binary else 3.2
            self.assertAlmostEqual(graph.value, expected_value, places=4)

if __name__ == '__main__':
    unittest.main()
