import unittest
import numpy as np
import cvxpy as cp
from gcspy.conic_program import ConicProgram

class TestConicProgram(unittest.TestCase):

    def setUp(self):

        # linear program
        #   minimize x1 + 2 * x2 + 3 * x3 + 4 * x4 + 1
        # subject to xi >= 1,  i = 1, 2, 3, 4
        #            x1 + x2 + x3 + x4 <= 6
        #            x4 = 2
        c = np.arange(4) + 1
        d = 1
        I = np.eye(4)
        A = [I, np.ones((1, 4)), I[-1:]]
        b = [- np.ones(4), np.array([-6]), np.array([-2])]
        K = [cp.constraints.NonNeg, cp.constraints.NonPos, cp.constraints.Zero]
        self.lp = ConicProgram(c, d, A, b, K)

        # second order cone program
        #   minimize sqrt(2) * x1 + 2
        # subject to x1^2 >= x1^2 + x2^2
        #            x1 >= x2
        #            x2 >= x3
        #            x3 >= 1
        c = np.array([np.sqrt(2), 0, 0])
        d = 2
        A = [np.eye(3), np.array([[1, -1, 0], [0, 1, -1], [0, 0, 1]])]
        b = [np.zeros(3), np.array([0, 0, -1])]
        K = [cp.constraints.SOC, cp.constraints.NonNeg]
        self.socp = ConicProgram(c, d, A, b, K)

        # semidefinite program equal to the previous socp
        # see https://people.eecs.berkeley.edu/~elghaoui/Teaching/EE227A/lecture6.pdf
        # for the translation
        #   minimize sqrt(2) * x1 + 2
        # subject to [x1, x2, x3]
        #            [x2, x1,  0] >> 0
        #            [x3,  0, x1]
        #            x1 >= x2
        #            x2 >= x3
        #            x3 >= 1
        A[0] = np.array([[1, 0, 0], # entry 00
                         [0, 1, 0], # entry 10
                         [0, 0, 1], # entry 20
                         [0, 1, 0], # entry 01
                         [1, 0, 0], # entry 11
                         [0, 0, 0], # entry 21
                         [0, 0, 1], # entry 02
                         [0, 0, 0], # entry 12
                         [1, 0, 0], # entry 22
                         ])
        b[0] = np.zeros(9)
        K[0] = cp.constraints.PSD
        self.sdp = ConicProgram(c, d, A, b, K)

    def test_init(self):

        # incoherent cone constraints
        c = np.ones(3)
        d = 0
        A = [np.eye(3)]
        b = [np.ones(3), np.zeros(1)]
        K = [cp.constraints.Zero]
        self.assertRaises(ValueError, ConicProgram, c, d, A, b, K)

        # incoherent A and b
        A = [np.eye(3)]
        b = [np.ones(4)]
        self.assertRaises(ValueError, ConicProgram, c, d, A, b, K)

        # incoherent A and c
        A = [np.eye(4)]
        self.assertRaises(ValueError, ConicProgram, c, d, A, b, K)

    def test_evaluate_cost(self):

        # tested only for linear program since unaffected by cones
        x = [np.zeros(4), np.ones(4), np.arange(4)]
        costs = [1, 11, 21]
        for xi, cost in zip(x, costs):
            self.assertAlmostEqual(self.lp.evaluate_cost(xi), cost)

    def test_evaluate_constraints(self):

        # tested only for linear program since unaffected by cones
        x = [np.zeros(4), np.ones(4), np.arange(4)]
        z = [np.concatenate(self.lp.b),
             np.array([0, 0, 0, 0, -2, -1]),
             np.array([-1, 0, 1, 2, 0, 1])]
        for xi, zi in zip(x, z):
            value = np.concatenate(self.lp.evaluate_constraints(xi))
            np.testing.assert_array_almost_equal(value, zi)

    def test_solve(self):

        # linear program
        cost, x_opt = self.lp._solve()
        self.assertAlmostEqual(cost, 15)
        np.testing.assert_array_almost_equal(x_opt, [1, 1, 1, 2])

        # second order cone program
        cost, x_opt = self.socp._solve()
        self.assertAlmostEqual(cost, 4)
        np.testing.assert_array_almost_equal(x_opt, [np.sqrt(2), 1, 1])

        # semidefinite program
        cost, x_opt = self.sdp._solve()
        self.assertAlmostEqual(cost, 4)
        np.testing.assert_array_almost_equal(x_opt, [np.sqrt(2), 1, 1])

    def test_from_symbolic(self):

        # linear program
        # relies on the fact that the translation in conic
        # form does not require auxiliary variables
        x = cp.Variable(4)
        obj = x[0] + 2 * x[1] + 3 * x[2] + 4 * x[3] + 1
        constraints = [x >= 1, cp.sum(x) <= 6, x[3] == 2]
        prog = ConicProgram.from_symbolic(obj, constraints)[0]
        cost, x_opt = prog._solve()
        self.assertAlmostEqual(cost, 15)
        np.testing.assert_array_almost_equal(x_opt, [1, 1, 1, 2])

        # second order cone program
        # relies on the fact that the translation in conic
        # form does not require auxiliary variables
        x = cp.Variable(3)
        obj = np.sqrt(2) * x[0] + 2
        constraints = [cp.SOC(x[0], x[1:]), x[0] >= x[1], x[1] >= x[2], x[2] >= 1]
        prog = ConicProgram.from_symbolic(obj, constraints)[0]
        cost, x_opt = prog._solve()
        self.assertAlmostEqual(cost, 4)
        np.testing.assert_array_almost_equal(x_opt, [np.sqrt(2), 1, 1])

        # semidefinite program
        # relies on the fact that the translation in conic
        # form does not require auxiliary variables
        M = cp.bmat([[x[0], x[1], x[2]], [x[1], x[0], 0], [x[2], 0, x[0]]])
        constraints[0] = M >> 0
        prog = ConicProgram.from_symbolic(obj, constraints)[0]
        cost, x_opt = prog._solve()
        self.assertAlmostEqual(cost, 4)
        np.testing.assert_array_almost_equal(x_opt, [np.sqrt(2), 1, 1])

        # constant cost
        x = cp.Variable(4)
        prog = ConicProgram.from_symbolic(1, [x >= 0])[0]
        cost, x_opt = prog._solve()
        self.assertAlmostEqual(cost, 1)

        # no constraints
        x = cp.Variable(3)
        prog = ConicProgram.from_symbolic(cp.norm_inf(x), [])[0]
        cost, x_opt = prog._solve()
        self.assertAlmostEqual(cost, 0)

        # constant cost and no constraints
        prog = ConicProgram.from_symbolic(1, [])[0]
        cost, x_opt = prog._solve()
        self.assertAlmostEqual(cost, 1)

        # linear program with matrix variable
        #   minimize x11 + 2 * x12 + x13 + 3 * x21 + 4 * x22 + 5 * x23 + 1
        # subject to xij >= i,  i = 1, 2, j = 1, 2, 3
        #            x11 + x12 + x21 + x22 <= 6
        #            x22 = 2
        X = cp.Variable((2, 3))
        obj = X[0,0] + 2 * X[0,1] + X[0,2] + 3 * X[1,0] + 4 * X[1,1] + 5 * X[1,2] + 1
        constraints = [
            X[0,0] >= 1, X[0,1] >= 1, X[0,2] >= 1,
            X[1,0] >= 2, X[1,1] >= 2, X[1,2] >= 2,
            X[0,0] + X[0,1] + X[1,0] + X[1,1] <= 6,
            X[1,1] == 2
            ]
        prog, select_variable = ConicProgram.from_symbolic(obj, constraints)
        cost, x_opt = prog._solve()
        self.assertAlmostEqual(cost, 29)
        np.testing.assert_array_almost_equal(x_opt, [1, 2, 1, 2, 1, 2])
        X_opt = np.array([[1, 1, 1], [2, 2, 2]])
        np.testing.assert_array_almost_equal(select_variable(X, x_opt), X_opt)

if __name__ == '__main__':
    unittest.main()
