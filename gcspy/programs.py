import numpy as np
import cvxpy as cp
from numbers import Number

class ConicProgram:

    def __init__(self, c, d, A, b, K, convex_id_to_conic_idx=None):

        # check that the number of conic constraints is coherent
        if len(A) != len(b) or len(b) != len(K):
            raise ValueError(
                f"Length mismatch: len(A) = {len(A)}, len(b) = {len(b)}, "
                f"len(K) = {len(K)}. A, b, and K must have equal length.")
        
        # check that the matrices and vectors have coherent size
        for Ai, bi in zip(A, b):
            if Ai.shape != (bi.size, c.size):
                raise ValueError(
                    f"Shape mismatch: Ai.shape = {Ai.shape}, bi.size = "
                    f"{bi.size}, c.size = {c.size}. Matrix in Ai must have "
                    "shape equal to (bi.size, c.size).")
        if not isinstance(d, Number):
            raise TypeError(f"d must be a scalar, got d of type {type(d)}.")
        
        # store input data
        self.c = c
        self.d = d
        self.A = A
        self.b = b
        self.K = K
        self.size = c.size

        # dictionary with keys equal to the variable ids in the convex program
        # that generated this conic program, and values equal to the
        # corresponding variable indices in the conic program
        self.convex_id_to_conic_idx = convex_id_to_conic_idx

    def evaluate_cost(self, x, t=1):
        return self.c @ x + self.d * t

    def evaluate_constraints(self, x, t=1):
        constraints = []
        for (Ai, bi, Ki) in zip(self.A, self.b, self.K):
            constraints.append(self.constrain_in_cone(Ai @ x + bi * t, Ki))
        return constraints
    
    @staticmethod
    def constrain_in_cone(z, K):
        if K in [cp.constraints.Zero, cp.constraints.NonNeg]:
            return K(z)
        elif K == cp.constraints.NonPos: # NonPos will be deprecated
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
        
    def get_convex_variable_value(self, convex_variable, conic_x):

        # check that program is created from a convex program
        if self.convex_id_to_conic_idx is None:
            raise ValueError(
                "Conic program was not generated from a convex program.")

        # retrieve value and reshape it appropriately
        value = conic_x[self.convex_id_to_conic_idx[convex_variable.id]]
        if convex_variable.is_matrix():
            if convex_variable.is_symmetric():
                n = convex_variable.shape[0]
                full = np.zeros((n, n))
                full[np.triu_indices(n)] = value
                value = full + full.T
                value[np.diag_indices(n)] /= 2
            else:
                value = value.reshape(convex_variable.shape, order='F')
        return value

    def solve(self, **kwargs):
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

    supported_attributes = ["nonneg", "nonpos", "symmetric", "PSD", "NSD"]

    def __init__(self):
        self.variables = []
        self.cost = 0
        self.constraints = []
        self.binary_variable = cp.Variable()

    def add_variable(self, shape, **kwargs):
        for attribute in kwargs:
            if not attribute in self.supported_attributes:
                raise ValueError(f"Attribute {attribute} is not supported.")
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
        defined_ids = [variable.id for variable in defined_variables]
        for variable in variables:
            if variable.id not in defined_ids:
                raise ValueError(f"Variable {variable} is not defined.")

    def to_conic(self):
        """
        Converts this ConvexProgram into an equivalent ConicProgram, using the
        reduction chain in cvxpy.
        """

        # ensure that all the variables are included in the conic program, it
        # does so by adding a dummy cost of zero on the unused variables
        used_ids = []
        for constraint in self.constraints:
            used_ids += [variable.id for variable in constraint.variables()]
        if not isinstance(self.cost, Number):
            used_ids += [variable.id for variable in self.cost.variables()]
        for variable in self.variables:
            if not variable.id in used_ids:
                self.add_cost(0 * variable.flatten()[0])
        
        # corner case with constant cost and no constraints
        if isinstance(self.cost, Number) and len(self.constraints) == 0:
            conic_program = ConicProgram([], self.cost, [], [], [], {})
            return conic_program

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

        # dictionary that maps the id of a variable in the cost and constraints
        # to the corresponding columns in the in the conic program
        convex_id_to_conic_idx = {}
        for variable in conic_prob.variables:
            start = conic_prob.var_id_to_col[variable.id]
            stop = start + variable.size
            convex_id_to_conic_idx[variable.id] = range(start, stop)

        return ConicProgram(c, d, A, b, K, convex_id_to_conic_idx)

    def solve(self, **kwargs):
        """
        This method is only used for testing, it is not used in the library.
        """
        prob = cp.Problem(cp.Minimize(self.cost), self.constraints)
        prob.solve(**kwargs)
        variable_values = [variable.value for variable in self.variables]
        return prob.value, variable_values