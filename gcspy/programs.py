import numpy as np
import cvxpy as cp
from gcspy.conic_program import ConicProgram
from numbers import Number

class ConicProgram:

    def __init__(self, c, d, A, b, K):

        # check that number of conic constraints is coherent
        lengths = [len(A), len(b), len(K)]
        if len(set(lengths)) != 1:
            raise ValueError("Length mismatch: "
                                f"len(A) = {len(A)}, len(b) = {len(b)}, len(K) = {len(K)}. "
                                "A, b, and K must be lists of equal length.")
        
        # check that matrices have coherent size
        for Ai, bi in zip(A, b):
            if Ai.shape != (bi.size, c.size):
                raise ValueError("Shape mismatch: "
                                    f"Ai.shape = {Ai.shape}, bi.size = {bi.size}, , c.size = {c.size}. "
                                    "The ith matrix in A must have shape (bi.size, c.size), "
                                    "where bi is the ith vector in b.")
        if not isinstance(d, Number):
            raise TypeError("d must be a scalar number.")
        
        # store data
        self.c = c
        self.d = d
        self.A = A
        self.b = b
        self.K = K
        self.size = c.size

    def evaluate_cost(self, x, t=1):
        return self.c @ x + self.d * t

    def evaluate_constraints(self, x, t=1):
        constraints = []
        for (Ai, bi, Ki) in zip(self.A, self.b, self.k):
            constraints.append(self.constrain_in_cone(Ai @ x + bi * t, Ki))
        return constraints
    
    @staticmethod
    def constrain_in_cone(z, K):
        if K == cp.constraints.Zero:
                return K(z)
        elif K == cp.constraints.NonNeg:
                return K(z)
        elif K == cp.constraints.NonPos:
                return cp.constraints.NonNeg(- z)
        elif K == cp.constraints.SOC:
            return K(z[0], z[1:])
        elif K == cp.constraints.PSD:
            n = round(z.size ** .5)
            z_mat = cp.reshape(z, (n, n), order='F')
            return K(z_mat)
        elif K == cp.constraints.ExpCone:
            z_mat = z.reshape((3, -1), order='C')
            return K(*z_mat)
        else:
            raise NotImplementedError

    def _solve(self, **kwargs):
        """
        This method is only used for testing, it is not used in the library.
        """
        x = cp.Variable(self.size)
        cost = self.evaluate_cost(x)
        constraints = self.evaluate_constraints(x)
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(**kwargs)
        return prob.value, x.value

class ConvexProgram:

    variable_attributes = ["nonneg", "nonpos", "symmetric", "PSD", "NSD"]

    def __init__(self):
        self.variables = []
        self.cost = 0
        self.constraints = []

    def add_variable(self, shape, **kwargs):
        for attribute in kwargs:
            if not attribute in self.variable_attributes:
                raise ValueError(f"Variable attribute {attribute} is not supported.")
        variable = cp.Variable(shape, **kwargs)
        self.variables.append(variable)
        return variable
    
    def add_cost(self, cost):
        if not isinstance(cost, Number):
            self.check_variables_are_defined(cost.variables())
        self.cost += cost

    def add_constraint(self, constraint):
        self.check_variables_are_defined(constraint.variables())
        self.constraints.append(constraint)

    def add_constraints(self, constraints):
        for constraint in constraints:
            self.add_constraint(constraint)

    def check_variables_are_defined(self, variables, defined_variables=None):
        if defined_variables is None:
            defined_variables = self.variables
        ids = {variable.id for variable in variables}
        defined_ids = {variable.id for variable in defined_variables}
        if not ids <= defined_ids:
            raise ValueError("A variable does not belong to this convex program.")

    def to_conic(self):
        """
        Converts this ConvexProgram into an equivalent ConicProgram, using CVXPY's reductions.
        Also returns a function that maps a solution x to values for the original variables.
        """

        # trick that ensures that all the variables are included in the conic program
        used_variable_ids = []
        for constraint in self.constraints:
            used_variable_ids.extend([variable.id for variable in constraint.variables()])
        if not isinstance(self.cost, Number):
            used_variable_ids.extend([variable.id for variable in self.cost.variables()])
        for variable in self.variables:
            if not variable.id in used_variable_ids:
                self.add_cost(0 * cp.sum(variable))
        
        # corner case with constant cost and no constraints
        id_to_cols = {}
        if isinstance(self.cost, Number) and len(self.constraints) == 0:
            c = []
            d = self.cost
            A = []
            b = []
            K = []
            conic_program = ConicProgram(c, d, A, b, K)
            # def get_variable_value(variable, x, reshape=True):
            #     return None
            return conic_program, id_to_cols

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

        # double check that variables are ordered
        ids = [variable.id for variable in conic_prob.variables]
        if ids != list(conic_prob.var_id_to_col.keys()):
            raise ValueError("Variables in the conic program are not ordered.")

        # dictionary that maps the id of a variable in the cost
        # and constraints to the corresponding columns in the
        # in the conic program
        start = 0
        for variable in conic_prob.variables:
            stop = start + variable.size
            id_to_cols[variable.id] = range(start, stop)
            start = stop

        return conic_program, id_to_cols

    def _solve(self, **kwargs):
        """
        This method is only used for testing, it is not used in the library.
        """
        prob = cp.Problem(cp.Minimize(self.cost), self.constraints)
        prob.solve(**kwargs)
        variable_values = [variable.value for variable in self.variables]
        return prob.value, variable_values