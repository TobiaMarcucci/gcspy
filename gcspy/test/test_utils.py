import unittest
import numpy as np
import cvxpy as cp

from gcspy.utils import to_cone_program, cone_program_perspective


class TestConePerspective(unittest.TestCase):

    def test_lp(self):

        # Single variable.
        x = cp.Variable(4)
        cost = np.ones(x.size) @ x + 3.5
        constraints = [x >= 1]
        variables = [x]
        self._compare_solves(cost, constraints, variables)

        # Multiple variables.
        y = cp.Variable(2)
        cost -= np.ones(y.size) @ y
        constraints.append(y <= 0)
        variables.append(y)
        self._compare_solves(cost, constraints, variables)

        # Matrix variable.
        Z = cp.Variable((4, 3))
        cost += np.ones(Z.shape[1]) @ Z[0]
        constraints.append(Z[-1] >= 1)
        for i in range(Z.shape[0] - 1):
            constraints.append(Z[i] == Z[i + 1])
        variables.append(Z)
        self._compare_solves(cost, constraints, variables)

    def test_qp(self):

        # No constraints.
        x = cp.Variable(5)
        cost = cp.quad_form(x, np.eye(x.size)) + 1.5
        constraints = []
        variables = [x]
        self._compare_solves(cost, constraints, variables)

        # Single variable.
        constraints = [x >= 1]
        self._compare_solves(cost, constraints, variables)

        # Multiple variables.
        y = cp.Variable(2)
        cost += cp.quad_form(y, np.eye(y.size))
        constraints.append(y >= -5)
        variables.append(y)
        self._compare_solves(cost, constraints, variables)

        # Matrix variable.
        Z = cp.Variable((4, 3))
        cost += cp.quad_form(Z[0], np.eye(Z.shape[1]))
        constraints.append(Z[-1] >= 1)
        for i in range(Z.shape[0] - 1):
            constraints.append(Z[i] == Z[i + 1])
        variables.append(Z)
        self._compare_solves(cost, constraints, variables)

    def test_socp(self):

        # SOC objective.
        x = cp.Variable(4)
        cost = cp.norm(x, 2) + 33
        constraints = [x >= 1]
        variables = [x]
        self._compare_solves(cost, constraints, variables)

        # SOC constraint.
        cost = np.ones(x.size) @ x
        constraints = [cp.norm(x, 2) <= 1]
        variables = [x]
        self._compare_solves(cost, constraints, variables)

        # Mixed with multiple variables.
        y = cp.Variable(2)
        cost += cp.norm(y, 2)
        constraints.append(y <= -1)
        variables.append(y)
        self._compare_solves(cost, constraints, variables)

    def test_sdp(self):

        X = cp.Variable((2, 2), symmetric=True)
        cost = - X[0, 0]
        constraints = [X >> 0, X >= 0, sum(sum(X)) == 1]
        variables = [X]
        self._compare_solves(cost, constraints, variables)

    def test_mixed_program(self):

        x = cp.Variable(4)
        cost = cp.norm(x, 2) + cp.quad_form(x, np.eye(x.size)) + np.ones(x.size) @ x + 3.5
        constraints = [x >= 1]
        variables = [x]
        self._compare_solves(cost, constraints, variables)

    def _compare_solves(self, cost, constraints, variables):

        # Solve directly.
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve()
        minimum = prob.value
        minimizer = [x.value for x in variables]

        # Solve as cone program.
        cone_data = to_cone_program(cost, constraints)
        new_variables = {x.id: cp.Variable(x.shape, **x.attributes) for x in variables}
        t = 1
        cone_cost, cone_constraints, _ = cone_program_perspective(cone_data, new_variables, t)
        cone_prob = cp.Problem(cp.Minimize(cone_cost), cone_constraints)
        cone_prob.solve()
        cone_minimum = cone_prob.value
        cone_minimizer = [y.value for y in new_variables.values()]

        # Compare solutions.
        self.assertAlmostEqual(minimum, cone_minimum, places=4)
        for x, y in zip(minimizer, cone_minimizer):
            self.assertTrue(np.allclose(x, y, atol=1e-4))
