import unittest
from gcsopt import GraphOfConvexSets

class TestGraphs(unittest.TestCase):

    def test_directed_graph(self):

        # Make graph.
        n_vertices = 5
        graph = GraphOfConvexSets(directed=True)
        for i in range(n_vertices):
                graph.add_vertex(i)
                if i > 0:
                    tail = graph.get_vertex(i - 1)
                    head = graph.get_vertex(i)
                    graph.add_edge(tail, head)

        # Test various methods.
        self.assertTrue(graph.has_vertex(0))
        self.assertFalse(graph.has_vertex(n_vertices))
        self.assertTrue(graph.has_edge((0, 1)))
        self.assertFalse(graph.has_edge((1, 0)))
        self.assertFalse(graph.has_edge((0, 2)))
        self.assertRaises(ValueError, graph.add_vertex, 0)
        self.assertRaises(ValueError, graph.add_edge, tail, head)
        self.assertEqual(graph.get_vertex(0), graph.vertices[0])
        self.assertRaises(ValueError, graph.get_vertex, n_vertices)
        self.assertEqual(graph.get_edge(0, 1), graph.edges[0])
        self.assertRaises(ValueError, graph.get_edge, 1, 0)
        self.assertRaises(ValueError, graph.get_edge, 0, 2)
        self.assertEqual(graph.vertex_index(graph.vertices[0]), 0)
        self.assertEqual(graph.edge_index(graph.edges[0]), 0)
        self.assertEqual(graph.incoming_edge_indices(graph.vertices[0]), [])
        self.assertEqual(graph.incoming_edge_indices(graph.vertices[1]), [0])
        self.assertEqual(graph.outgoing_edge_indices(graph.vertices[-1]), [])
        self.assertEqual(graph.outgoing_edge_indices(graph.vertices[-2]), [n_vertices - 2])
        self.assertEqual(graph.incident_edge_indices(graph.vertices[0]), [0])
        self.assertEqual(graph.incident_edge_indices(graph.vertices[1]), [0, 1])
        self.assertEqual(graph.induced_edge_indices(graph.vertices[:1]), [])
        self.assertEqual(graph.induced_edge_indices(graph.vertices[:2]), [0])
        self.assertEqual(graph.induced_edge_indices(graph.vertices[:3]), [0, 1])
        self.assertEqual(graph.num_vertices(), n_vertices)
        self.assertEqual(graph.num_edges(), n_vertices - 1)

    def test_undirected_graph(self):

        # Make graph.
        n_vertices = 5
        graph = GraphOfConvexSets(directed=False)
        for i in range(n_vertices):
                graph.add_vertex(i)
                if i > 0:
                    tail = graph.get_vertex(i - 1)
                    head = graph.get_vertex(i)
                    graph.add_edge(tail, head)

        # Test various methods.
        self.assertTrue(graph.has_vertex(0))
        self.assertFalse(graph.has_vertex(n_vertices))
        self.assertTrue(graph.has_edge((0, 1)))
        self.assertTrue(graph.has_edge((1, 0)))
        self.assertFalse(graph.has_edge((0, 2)))
        self.assertRaises(ValueError, graph.add_vertex, 0)
        self.assertRaises(ValueError, graph.add_edge, tail, head)
        self.assertRaises(ValueError, graph.add_edge, head, tail)
        self.assertEqual(graph.get_vertex(0), graph.vertices[0])
        self.assertRaises(ValueError, graph.get_vertex, n_vertices)
        self.assertEqual(graph.get_edge(0, 1), graph.edges[0])
        self.assertEqual(graph.get_edge(1, 0), graph.edges[0])
        self.assertRaises(ValueError, graph.get_edge, 0, 2)
        self.assertEqual(graph.vertex_index(graph.vertices[0]), 0)
        self.assertEqual(graph.edge_index(graph.edges[0]), 0)
        self.assertRaises(ValueError, graph.incoming_edge_indices, graph.vertices[-1])
        self.assertRaises(ValueError, graph.outgoing_edge_indices, graph.vertices[0])
        self.assertEqual(graph.incident_edge_indices(graph.vertices[0]), [0])
        self.assertEqual(graph.incident_edge_indices(graph.vertices[1]), [0, 1])
        self.assertEqual(graph.induced_edge_indices(graph.vertices[:1]), [])
        self.assertEqual(graph.induced_edge_indices(graph.vertices[:2]), [0])
        self.assertEqual(graph.induced_edge_indices(graph.vertices[:3]), [0, 1])
        self.assertEqual(graph.num_vertices(), n_vertices)
        self.assertEqual(graph.num_edges(), n_vertices - 1)

if __name__ == '__main__':
    unittest.main()
