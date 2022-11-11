import unittest
import numpy as np
import cvxpy as cp

from gcspy.graph_of_convex_sets import GraphOfConvexSets
from gcspy.shortest_path_problem import ShortestPathProblem


class TestShortestPathProblem(unittest.TestCase):

    def test_planar(self):

        # Initialize graph.
        gcs = GraphOfConvexSets()

        # Add vertices.
        u = gcs.add_vertex('u')
        v = gcs.add_vertex('v')
        w = gcs.add_vertex('w')

        # Add continuous variables.
        xu = u.add_variable(2)
        xv = v.add_variable(2)
        xw = w.add_variable(2)

        # Constraints on the vertices.
        u.add_constraint(xu == 0)
        v.add_constraint(xv >= .4)
        v.add_constraint(xv <= .6)
        w.add_constraint(xw == 1)

        # Add edges.
        e = gcs.add_edge(u, v)
        f = gcs.add_edge(u, w)
        g = gcs.add_edge(v, w)
        h = gcs.add_edge(w, v)

        # Costs on the edges.
        e.add_length(cp.quad_form(xv - xu, np.eye(2)))
        f.add_length(cp.quad_form(xw - xu, np.eye(2)))
        g.add_length(cp.quad_form(xw - xv, np.eye(2)))
        h.add_length(cp.norm(xv - xw, 2))

        # Solve spp.
        spp = ShortestPathProblem(gcs)
        sol = spp.solve(u, w, relaxation=0)

        # Check that solver takes two steps.
        self.assertAlmostEqual(sol.y[e], 1, places=4)
        self.assertAlmostEqual(sol.y[f], 0, places=4)
        self.assertAlmostEqual(sol.y[g], 1, places=4)
        self.assertAlmostEqual(sol.y[h], 0, places=4)

        # Check that the two steps have equal length.
        self.assertTrue(np.allclose(sol.x[u], [0, 0]))
        self.assertTrue(np.allclose(sol.x[v], [.5, .5]))
        self.assertTrue(np.allclose(sol.x[w], [1, 1]))
