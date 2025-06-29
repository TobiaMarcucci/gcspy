import unittest
import numpy as np
import cvxpy as cp
from gcspy.convex_program import ConvexProgram

class TestConicProgram(unittest.TestCase):

    def setUp(self):

        # linear program
        #   minimize x1 + 2 * x2 + 3 * x3 + 4 * x4 + 1
        # subject to xi >= 1,  i = 1, 2, 3, 4
        #            x1 + x2 + x3 + x4 <= 6
        #            x4 = 2
        self.lp = ConvexProgram()
        x = self.lp.add_variable(4)
        self.lp.add_cost(x[0] + 2 * x[1] + 3 * x[2] + 4 * x[3] + 1)
        self.lp.add_constraint(x >= 1)
        self.lp.add_constraints([cp.sum(x) <= 6, x[3] == 2])

        # second order cone program
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

        # semidefinite program
        # for the translation
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

    def test_solve(self):

        # linear program
        program_value, variable_values = self.lp._solve()
        self.assertAlmostEqual(program_value, 15)
        np.testing.assert_array_almost_equal(variable_values[0], [1, 1, 1, 2])

        # second order cone program
        program_value, variable_values = self.socp._solve()
        self.assertAlmostEqual(program_value, 4)
        print(self.socp.variables)
        print(variable_values)
        np.testing.assert_array_almost_equal(variable_values[0], [np.sqrt(2), 1, 1])

        # semidefinite program
        program_value, variable_values = self.sdp._solve()
        self.assertAlmostEqual(program_value, 4)
        np.testing.assert_array_almost_equal(variable_values[0], [np.sqrt(2), 1, 1])
        X_opt = np.array([[np.sqrt(2), 1, 1], [1, np.sqrt(2), 0], [1, 0, np.sqrt(2)]])
        np.testing.assert_array_almost_equal(variable_values[1], X_opt)

    def test_to_conic(self):

        # linear program with matrix variable
        #   minimize x11 + 2 * x12 + x13 + 3 * x21 + 4 * x22 + 5 * x23 + 1
        # subject to xij >= i,  i = 1, 2, j = 1, 2, 3
        #            x11 + x12 + x21 + x22 <= 6
        #            x22 = 2
        matrix_lp = ConvexProgram()
        X = matrix_lp.add_variable((2, 3))
        matrix_lp.add_cost(X[0,0] + 2 * X[0,1] + X[0,2] + 3 * X[1,0] + 4 * X[1,1] + 5 * X[1,2] + 1)
        matrix_lp.add_constraints([X[0,0] >= 1, X[0,1] >= 1, X[0,2] >= 1,
                                     X[1,0] >= 2, X[1,1] >= 2, X[1,2] >= 2,
                                     X[0,0] + X[0,1] + X[1,0] + X[1,1] <= 6,
                                     X[1,1] == 2])
        
        # miscellaneous program
        misc = ConvexProgram()
        x = misc.add_variable(3, nonneg=True)
        y = misc.add_variable(2, nonneg=True)
        X = misc.add_variable((3, 3), PSD=True)
        Y = misc.add_variable((3, 4))
        misc.add_cost(cp.norm2(x + 2) + cp.sum_squares(y + 3) + 3)
        misc.add_cost(cp.sum(cp.diag(X)) + X[0, 2] + cp.max(Y[0]))
        misc.add_constraints([x >= -3, y <= 3, X[0, 1] == 1, Y == np.ones(Y.shape)])

        # checks that solving as a convex program is equal to solving as a conic program
        for convex_prog in [self.lp, self.socp, self.sdp, matrix_lp, misc]:
            convex_value, convex_var_values = convex_prog._solve()
            conic_prog, get_var_value = convex_prog.to_conic_program()
            conic_value, conic_var_values = conic_prog._solve()
            self.assertAlmostEqual(convex_value, conic_value, places=6)
            for var, convex_var_value in zip(convex_prog.variables, convex_var_values):
                conic_var_value = get_var_value(var, conic_var_values)
                np.testing.assert_array_almost_equal(convex_var_value, conic_var_value)

        # constant cost
        convex_prog = ConvexProgram()
        x = convex_prog.add_variable(4)
        convex_prog.add_cost(1)
        convex_prog.add_constraint(x >= 0)
        conic_prog, get_var_value = convex_prog.to_conic_program()
        conic_value, conic_var_values = conic_prog._solve()
        self.assertAlmostEqual(conic_value, 1)
        x_opt = get_var_value(x, conic_var_values)
        self.assertTrue(np.all(x_opt >= 0))

        # program with free variable
        convex_prog = ConvexProgram()
        x = convex_prog.add_variable(4, nonneg=True)
        convex_prog.add_cost(1)
        self.assertRaises(ValueError, convex_prog.to_conic_program)

        # no constraints
        convex_prog = ConvexProgram()
        x = convex_prog.add_variable(3)
        convex_prog.add_cost(cp.norm_inf(x))
        conic_prog, get_var_value = convex_prog.to_conic_program()
        conic_value, conic_var_values = conic_prog._solve()
        self.assertAlmostEqual(conic_value, 0)
        x_opt = get_var_value(x, conic_var_values)
        np.testing.assert_array_almost_equal(x_opt, np.zeros(x.size))

        # constant cost, no variables, and no constraints
        convex_prog = ConvexProgram()
        convex_prog.add_cost(1)
        conic_prog = convex_prog.to_conic_program()[0]
        conic_value = conic_prog._solve()[0]
        self.assertAlmostEqual(conic_value, 1)

if __name__ == '__main__':
    unittest.main()
