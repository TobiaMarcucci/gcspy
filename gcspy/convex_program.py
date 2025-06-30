import numpy as np
import cvxpy as cp
from gcspy.conic_program import ConicProgram
from numbers import Number

class ConvexProgram:

    variable_attributes = ["nonneg", "nonpos", "symmetric", "PSD", "NSD"]

    def __init__(self):
        self.variables = []
        self.cost = 0
        self.constraints = []

    @property
    def _variable_ids(self):
        return [variable.id for variable in self.variables]
    
    @property
    def _cost_variable_ids(self):
        if isinstance(self.cost, Number):
            return []
        else:
            return [variable.id for variable in self.cost.variables()]
    
    @property
    def _constraint_variable_ids(self):
        return [variable.id for constraint in self.constraints for variable in constraint.variables()]
    
    def _solve(self):
        prob = cp.Problem(cp.Minimize(self.cost), self.constraints)
        prob.solve()
        variable_values = [variable.value for variable in self.variables]
        return prob.value, variable_values

    def add_variable(self, shape, **kwargs):
        for attribute in kwargs:
            if not attribute in self.variable_attributes:
                raise ValueError(f"Variable attribute {attribute} is not supported.")
        variable = cp.Variable(shape, **kwargs)
        self.variables.append(variable)
        return variable
    
    def add_cost(self, cost):
        if not isinstance(cost, Number):
            for variable in cost.variables():
                if not variable.id in self._variable_ids:
                    raise ValueError(f"Variable {variable} does not belong to this convex program.")
        self.cost += cost

    def add_constraint(self, constraint):
        for variable in constraint.variables():
            if not variable.id in self._variable_ids:
                raise ValueError(f"Variable {variable} does not belong to this convex program.")
        self.constraints.append(constraint)

    def add_constraints(self, constraints):
        for constraint in constraints:
            self.add_constraint(constraint)

    def to_conic_program(self):

        # check that problem has no free variables
        for variable in self.variables:
            if not variable.id in self._cost_variable_ids + self._constraint_variable_ids:
                    raise ValueError(f"Convex program has free variable {variable}.")

        # corner case with constant cost and no constraints
        if isinstance(self.cost, Number) and len(self.constraints) == 0:
            c = []
            d = self.cost
            A = []
            b = []
            K = []
            conic_program = ConicProgram(c, d, A, b, K)
            def get_variable_value(variable, x, reshape=True):
                return None
            return conic_program, get_variable_value

        # construct conic program from given convex program
        prob = cp.Problem(cp.Minimize(self.cost), self.constraints)
        if not prob.is_dcp():
            raise ValueError(f"Convex program is not DCP.")
        solver_opts = {"use_quad_obj": False}
        chain = prob._construct_chain(solver_opts=solver_opts)
        chain.reductions = chain.reductions[:-1]
        conic_prob = chain.apply(prob)[0]

        # cost
        cd = conic_prob.c.toarray().flatten()
        # TODO: convert to sparse format
        c = cd[:-1]
        d = cd[-1]

        # constraints
        cols = conic_prob.c.shape[0]
        Ab = conic_prob.A.toarray().reshape((-1, cols), order='F')
        # TODO: convert to sparse format
        A = []
        b = []
        K = []
        start = 0
        for constraint in conic_prob.constraints:
            stop = start + constraint.size
            A.append(Ab[start:stop, :-1])
            b.append(Ab[start:stop, -1])
            K.append(type(constraint))
            start = stop
        conic_program = ConicProgram(c, d, A, b, K)

        # dictionary that maps the id of a variable in the cost
        # and constraints to the corresponding columns in the
        # in the conic program
        var_id_to_cols = {}
        for variable in conic_prob.variables:
            start = conic_prob.var_id_to_col[variable.id]
            stop = start + variable.size
            var_id_to_cols[variable.id] = range(start, stop)

        # function that selects the entries of x that correspond
        # to a given variable in the cost and constraints
        def get_variable_value(variable, x, reshape=True):

            # external variable
            if not variable.id in var_id_to_cols:
                return None
            
            # infeasible program
            if x is None:
                return None
            
            # feasible program
            value = x[var_id_to_cols[variable.id]]
            if reshape:
                if variable.is_matrix():
                    if variable.is_symmetric():
                        n = variable.shape[0]
                        full = np.zeros((n, n))
                        full[np.triu_indices(n)] = value
                        value = full + full.T
                        value[np.diag_indices(n)] /= 2
                    else:
                        value = value.reshape(variable.shape, order='F')
            return value
        
        return conic_program, get_variable_value
