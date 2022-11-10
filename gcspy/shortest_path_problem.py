import numpy as np
import cvxpy as cp

from gcspy.utils import to_cone_program, cone_perspective


class SpatialVariables(dict):

    def __add__(self, other):
        assert self.keys() == other.keys()
        return SpatialVariables({x: self[x] + other[x] for x in self})

    def __sub__(self, other):
        assert self.keys() == other.keys()
        return SpatialVariables({x: self[x] - other[x] for x in self})

    def __truediv__(self, c):
        return SpatialVariables({x: self[x] / c for x in self})

    def __radd__(self, other):
        return self if other == 0 else self.__add__(other)

    def __eq__(self, other):
        if isinstance(other, dict):
            assert self.keys() == other.keys()
            return [self[x] == other[x] for x in self]
        else:
            return [self[x] == other for x in self]

    @property
    def value(self):
        return SpatialVariables({x: self[x].value for x in self})
    
    @staticmethod
    def fill_as(variables):
        return SpatialVariables({x: cp.Variable(x.shape, **x.attributes) for x in variables})


class ShortestPathProblem():

    def __init__(self, gcs):

        self.gcs = gcs
        self.cyclic = gcs.has_cycles()

    def solve(self, s, t, relaxation=False, tol=1e-4):

        # Problem setup.
        cost = 0
        constraints = []
        E = self.gcs.edges
        y = {e: cp.Variable(boolean = not relaxation) for e in E}
        z = {e: SpatialVariables.fill_as(e.u.variables) for e in E}
        z1 = {e: SpatialVariables.fill_as(e.v.variables) for e in E}

        # Helpers.
        V = self.gcs.vertices
        incoming = {v: self.gcs.in_edges(v) for v in V}
        outgoing = {v: self.gcs.out_edges(v) for v in V}
        inflow = {v: sum(y[e] for e in incoming[v]) for v in V}
        outflow = {v: sum(y[e] for e in outgoing[v]) for v in V}
        spatial_inflow = {v: sum(z1[e] for e in incoming[v]) for v in V}
        spatial_outflow = {v: sum(z[e] for e in outgoing[v]) for v in V}

        # Flow at the source and the target.
        for e in incoming[s]:
            constraints.append(y[e] == 0)
            constraints.extend(z1[e] == 0)
        for e in outgoing[t]:
            constraints.append(y[e] == 0)
            constraints.extend(z[e] == 0)
        constraints.append(outflow[s] == 1)
        constraints.append(inflow[t] == 1)

        for v in self.gcs.vertices:

            # Nonnegativity.
            cone_data = to_cone_program(0, v.constraints)
            for e in outgoing[v]:
                constraints.extend(cone_perspective(cone_data, z[e], y[e])[1])
            for e in incoming[v]:
                constraints.extend(cone_perspective(cone_data, z1[e], y[e])[1])

            # Conservation of flow.
            if (v not in [s, t]) and (len(incoming[v]) + len(outgoing[v]) > 0):
                constraints.append(outflow[v] == inflow[v])
                constraints.extend(spatial_inflow[v] == spatial_outflow[v])

                if self.cyclic:

                    # Degree constraints.
                    constraints.append(outflow[v] <= 1)

                    # 2-cycle elimination.
                    for e in incoming[v]:
                        for f in outgoing[v]:
                            if e.u == f.v:
                                circulation = outflow[v] - y[e] - y[f]
                                spatial_circulation = spatial_outflow[v] - z1[e] - z[f]
                                constraints.extend(cone_perspective(cone_data, spatial_circulation, circulation)[1])

        # Edge lenghts and constraints.
        for e in self.gcs.edges:
            cone_data = to_cone_program(e.length, e.constraints)
            ze = z[e] | z1[e]
            cost_e, constraint_e = cone_perspective(cone_data, ze, y[e])[:2]
            cost += cost_e
            constraints.extend(constraint_e)

        # Solve SPP.
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve()

        # Get optimal solution.
        y_opt = {e: y[e].value for e in E}
        z_opt = {e: list(z[e].value.values()) for e in E}
        z1_opt = {e: list(z1[e].value.values()) for e in E}

        # Reconstruct optimal values of x.
        x_opt = {}
        for v in self.gcs.vertices:
            if prob.status == 'optimal':
                if v == t:
                    den = sum(y[e].value for e in incoming[v])
                    num = sum(z1[e].value for e in incoming[v])
                else:
                    den = sum(y[e].value for e in outgoing[v])
                    num = sum(z[e].value for e in outgoing[v])
                if den < tol:
                    x_opt[v] = v.get_feasible_point()
                else:
                    x_opt[v] = list((num / den).values())
            else:
                x_opt[v] = None

        return ShortestPathSolution(prob, y_opt, z_opt, z1_opt, x_opt, prob.value)


class ShortestPathSolution():

    def __init__(self, prob, y, z, z1, x, length):

        self.status = prob.status
        self.length = prob.value
        self.solve_time = prob.solver_stats.solve_time
        self.y = y
        self.z = z
        self.z1 = z1
        self.x = x
