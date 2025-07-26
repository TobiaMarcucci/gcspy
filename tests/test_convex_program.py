import unittest
import numpy as np
import cvxpy as cp
from gcspy.convex_program import ConvexProgram

class TestConvexProgram(unittest.TestCase):

    def setUp(self):

        # Linear program:
        #   minimize x1 + 2 * x2 + 3 * x3 + 4 * x4 + 1
        # subject to xi >= 1,  i = 1, 2, 3, 4
        #            x1 + x2 + x3 + x4 <= 6
        #            x4 = 2
        self.lp = ConvexProgram()
        x = self.lp.add_variable(4)
        self.lp.add_cost(x[0] + 2 * x[1] + 3 * x[2] + 4 * x[3] + 1)
        self.lp.add_constraint(x >= 1)
        self.lp.add_constraints([cp.sum(x) <= 6, x[3] == 2])

        # Second order cone program:
        #   minimize sqrt(2) * x1 + 2
        # subject to x1^2 >= x1^2 + x2^2
        #            x1 >= x2
        #            x2 >= x3
        #            x3 >= 1
        self.socp = ConvexProgram()
        x = self.socp.add_variable(3)
        self.socp.add_cost(np.sqrt(2) * x[0] + 2)
        self.socp.add_constraint(cp.SOC(x[0], x[1:]))
        self.socp.add_constraints([x[0] >= x[1], x[1] >= x[2], x[2] >= 1])

        # Semidefinite program:
        #   minimize sqrt(2) * x1 + 2
        # subject to [x1, x2, x3]
        #            [x2, x1,  0] >> 0
        #            [x3,  0, x1]
        #            x1 >= x2
        #            x2 >= x3
        #            x3 >= 1
        self.sdp = ConvexProgram()
        x = self.sdp.add_variable(3)
        X = self.sdp.add_variable((3, 3), PSD=True)
        self.sdp.add_cost(np.sqrt(2) * x[0] + 2)
        self.sdp.add_constraints([x[0] >= x[1], x[1] >= x[2], x[2] >= 1])
        self.sdp.add_constraints([X[i, i] == x[0] for i in range(3)])
        self.sdp.add_constraints([X[1,0] == x[1], X[2,0] == x[2], X[2,1] == 0])

    def test_add_variable(self):

        # Define some variables.
        prog = ConvexProgram()
        prog.add_variable(3, nonneg=True)
        prog.add_variable(3, nonpos=True)
        prog.add_variable((3, 3), symmetric=True)
        prog.add_variable((3, 3), PSD=True)
        prog.add_variable((3, 3), NSD=True)
        self.assertEqual(len(prog.variables), 5)

        # Test shapes.
        self.assertEqual(prog.variables[0].shape, (3,))
        self.assertEqual(prog.variables[1].shape, (3,))
        self.assertEqual(prog.variables[2].shape, (3, 3))
        self.assertEqual(prog.variables[3].shape, (3, 3))
        self.assertEqual(prog.variables[4].shape, (3, 3))

        # Test attributes.
        self.assertTrue(prog.variables[0].is_nonneg())
        self.assertTrue(prog.variables[1].is_nonpos())
        self.assertTrue(prog.variables[2].is_symmetric())
        self.assertTrue(prog.variables[3].is_psd())
        self.assertTrue(prog.variables[4].is_nsd())

    def test_add_cost(self):
        
        # Define some costs.
        prog = ConvexProgram()
        x = prog.add_variable(3)
        X = prog.add_variable((3, 3), PSD=True)
        prog.add_cost(3)
        prog.add_cost(cp.norm2(x) - cp.log_det(X))

        # Raise error if external variable.
        y = cp.Variable(3)
        with self.assertRaises(ValueError):
            prog.add_cost(cp.norm2(y))

    def test_add_constraint(self):

        # Define some constraints.
        prog = ConvexProgram()
        x = prog.add_variable(3)
        X = prog.add_variable((3, 3), PSD=True)
        prog.add_constraint(x >= 0)
        prog.add_constraint(np.ones(3) @ x == 0)
        prog.add_constraint(cp.log_det(X) >= 3)
        prog.add_constraints([x >= 3, x <= 11])

        # Raise error if external variable.
        y = cp.Variable(3)
        with self.assertRaises(ValueError):
            prog.add_constraint(y >= 0)

    def test_solve(self):

        # Linear program.
        program_value, variable_values = self.lp._solve()
        self.assertAlmostEqual(program_value, 15, places=4)
        np.testing.assert_array_almost_equal(variable_values[0], [1, 1, 1, 2], decimal=4)

        # Second order cone program.
        program_value, variable_values = self.socp._solve()
        self.assertAlmostEqual(program_value, 4, places=4)
        np.testing.assert_array_almost_equal(variable_values[0], [np.sqrt(2), 1, 1], decimal=4)

        # Semidefinite program.
        program_value, variable_values = self.sdp._solve()
        self.assertAlmostEqual(program_value, 4, places=4)
        np.testing.assert_array_almost_equal(variable_values[0], [np.sqrt(2), 1, 1], decimal=4)
        X_opt = np.array([[np.sqrt(2), 1, 1], [1, np.sqrt(2), 0], [1, 0, np.sqrt(2)]])
        np.testing.assert_array_almost_equal(variable_values[1], X_opt, decimal=4)

    def test_homogenization(self):

        # List of programs to be tested.
        programs = [self.lp, self.socp, self.sdp]

        # Linear program with matrix variable
        #   minimize x11 + 2 * x12 + x13 + 3 * x21 + 4 * x22 + 5 * x23 + 1
        # subject to xij >= i,  i = 1, 2, j = 1, 2, 3
        #            x11 + x12 + x21 + x22 <= 6
        #            x22 = 2
        prog = ConvexProgram()
        X = prog.add_variable((2, 3))
        prog.add_cost(X[0,0] + 2 * X[0,1] + X[0,2] + 3 * X[1,0] + 4 * X[1,1] + 5 * X[1,2] + 1)
        prog.add_constraints([X[0,0] >= 1, X[0,1] >= 1, X[0,2] >= 1,
                                     X[1,0] >= 2, X[1,1] >= 2, X[1,2] >= 2,
                                     X[0,0] + X[0,1] + X[1,0] + X[1,1] <= 6,
                                     X[1,1] == 2])
        programs.append(prog)
        
        # Miscellaneous program.
        prog = ConvexProgram()
        x = prog.add_variable(3, nonneg=True)
        y = prog.add_variable(2, nonneg=True)
        X = prog.add_variable((3, 3), PSD=True)
        Y = prog.add_variable((3, 4))
        prog.add_cost(cp.norm2(x + 2) + cp.sum_squares(y + 3) + 3)
        prog.add_cost(- cp.log_det(X) + X[0, 2] + cp.max(Y[0]))
        prog.add_constraints([x >= -3, y <= 3, X[0, 1] == 1, Y == np.ones(Y.shape)])
        programs.append(prog)

        # Infeasible program.
        prog = ConvexProgram()
        x = prog.add_variable(4, nonneg=True)
        prog.add_cost(cp.sum(x))
        prog.add_constraint(x <= -1)
        programs.append(prog)

        # Infeasible program with two variables.
        prog = ConvexProgram()
        x = prog.add_variable(4, nonneg=True)
        y = prog.add_variable(2)
        prog.add_cost(cp.sum(x) + cp.sum(y))
        prog.add_constraint(x <= -1)
        programs.append(prog)

        # Unbounded program.
        prog = ConvexProgram()
        x = prog.add_variable(4)
        prog.add_cost(cp.sum(x))
        prog.add_constraint(x <= -1)
        programs.append(prog)

        # Trivial sdp.
        prog = ConvexProgram()
        X = prog.add_variable((1, 1), PSD=True)
        prog.add_cost(- cp.log_det(X))
        prog.add_constraint(X[0, 0] == 1)
        programs.append(prog)

        # Sdp for inimum volume ellipsoid.
        for d in range(2, 6):
            points = np.vstack((np.zeros((1, d)), np.eye(d)))
            prog = ConvexProgram()
            A = prog.add_variable((d, d), PSD=True)
            b = prog.add_variable(d)
            prog.add_cost(- cp.log_det(A))
            for point in points:
                prog.add_constraint(cp.norm2(A @ point + b) <= 1)
            programs.append(prog)

        # Check that solving convex program is equal to solve conic program.
        for prog in programs:
            value, var_values = prog._solve()
            new_variables = prog.copy_variables()
            new_cost, new_constraints = prog.homogenization(new_variables, 1)
            new_prog = cp.Problem(cp.Minimize(new_cost), new_constraints)
            new_prog.solve()
            self.assertAlmostEqual(value, new_prog.value, places=4)
            for new_var, convex_var_value in zip(new_variables, var_values):
                if convex_var_value is None:
                    self.assertIsNone(new_var.value)
                else:
                    np.testing.assert_array_almost_equal(convex_var_value, new_var.value, decimal=4)

        # Program with constant cost.
        prog = ConvexProgram()
        x = prog.add_variable(4)
        prog.add_cost(1.55)
        prog.add_constraint(x >= 0)
        new_variables = prog.copy_variables()
        new_cost, new_constraints = prog.homogenization(new_variables, 1)
        new_prog = cp.Problem(cp.Minimize(new_cost), new_constraints)
        new_prog.solve()
        self.assertAlmostEqual(new_prog.value, 1.55, places=4)
        self.assertTrue(np.all(new_variables[0].value >= -1e-4))

        # Program with free variable.
        prog = ConvexProgram()
        x = prog.add_variable(4, nonneg=True)
        prog.add_cost(1.55)
        new_variables = prog.copy_variables()
        new_cost, new_constraints = prog.homogenization(new_variables, 1)
        new_prog = cp.Problem(cp.Minimize(new_cost), new_constraints)
        new_prog.solve()
        self.assertAlmostEqual(new_prog.value, 1.55, places=4)
        self.assertTrue(np.all(new_variables[0].value >= -1e-4))

        # Program with no constraints.
        prog = ConvexProgram()
        x = prog.add_variable(3)
        prog.add_cost(cp.norm_inf(x))
        new_variables = prog.copy_variables()
        new_cost, new_constraints = prog.homogenization(new_variables, 1)
        new_prog = cp.Problem(cp.Minimize(new_cost), new_constraints)
        new_prog.solve()
        self.assertAlmostEqual(new_prog.value, 0, places=4)
        x_opt = new_variables[0].value
        np.testing.assert_array_almost_equal(x_opt, np.zeros(x.size), decimal=4)

        # Program with constant cost, no variables, and no constraints.
        prog = ConvexProgram()
        prog.add_cost(1.55)
        new_variables = prog.copy_variables()
        new_cost, new_constraints = prog.homogenization(new_variables, 1)
        new_prog = cp.Problem(cp.Minimize(new_cost), new_constraints)
        new_prog.solve()
        self.assertAlmostEqual(new_prog.value, 1.55, places=4)

if __name__ == '__main__':
    unittest.main()
