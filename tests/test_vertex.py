import unittest
import numpy as np
import cvxpy as cp
from gcspy.vertex import Vertex

class TestConicProgram(unittest.TestCase):

    def test_init(self):
        vertex = Vertex('name')
        x = vertex.add_variable(3)
        vertex.add_cost(cp.sum(x))
        vertex.add_constraint(x >= 0)
        program_value, variable_values = vertex._solve()
        self.assertAlmostEqual(program_value, 0, places=4)
        np.testing.assert_array_almost_equal(variable_values[0], np.zeros(3), decimal=4)

    def test_to_conic_program(self):

        # linear program
        vertex = Vertex()
        self.assertIsNone(vertex.conic_program)
        self.assertIsNone(vertex.get_variable_value)
        x = vertex.add_variable(4)
        vertex.add_cost(cp.sum(x) + 1)
        vertex.add_constraint(x >= 1)
        vertex.add_constraints([x <= 6, x[0] == 2])
        vertex.to_conic_program()
        vertex_value, vertex_var_values = vertex._solve()
        conic_value, conic_var_values = vertex.conic_program._solve()
        self.assertAlmostEqual(vertex_value, conic_value, places=4)
        for var, vertex_var_value in zip(vertex.variables, vertex_var_values):
            conic_var_value = vertex.get_variable_value(var, conic_var_values)
            np.testing.assert_array_almost_equal(vertex_var_value, conic_var_value, decimal=4)

    # def test_get_feasible_point(self):
    #     vertex = Vertex()
    #     x = vertex.add_variable(4)
    #     y = vertex.add_variable(2)
    #     for variable in vertex.variables:
    #         self.assertIsNone(variable.value)
    #     vertex.add_cost(cp.sum(x) + cp.norm2(y) + 1)
    #     vertex.add_constraints([x >= 1, x <= 6, x[0] == 2, y >= 2])
    #     feasible_point = vertex.get_feasible_point()
    #     self.assertAlmostEqual(feasible_point[0][0], 2, places=4)
    #     self.assertGreater(min(feasible_point[0][1:]), 1 - 1e-4)
    #     self.assertLess(max(feasible_point[0][1:]), 6 + 1e-4)
    #     self.assertGreater(min(feasible_point[1]), 2 - 1e-4)
    #     for variable in vertex.variables:
    #         self.assertIsNone(variable.value)
    #     # TODO: check that the objective is the same as before finding the feasible point
    #     # TODO: check that the optimization for the feasible point used zero as objective

if __name__ == '__main__':
    unittest.main()
