import numpy as np
import cvxpy as cp
from numbers import Number

class ConicProgram:

    def __init__(self, convex_program):

        # Store variables.
        self.variables = convex_program.variables

        # Check that all the variables are used in the convex program. If not,
        # add a dummy cost of zero on the unused variables.
        used_ids = []
        for constraint in convex_program.constraints:
            used_ids += [variable.id for variable in constraint.variables()]
        if not isinstance(convex_program.cost, Number):
            used_ids += [variable.id for variable in convex_program.cost.variables()]
        for variable in convex_program.variables:
            if not variable.id in used_ids:
                convex_program.cost += 0 * variable.flatten(order='F')[0]

        # Deal with corner case with constant cost and no constraints.
        if isinstance(convex_program.cost, Number) and len(convex_program.constraints) == 0:
            self.c = np.array([])
            self.d = convex_program.cost
            self.A = np.array([])
            self.b = np.array([])
            self.K = []
            self.id_to_size = {}
            return

        # Construct conic program from given convex program.
        prob = cp.Problem(cp.Minimize(convex_program.cost), convex_program.constraints)
        if not prob.is_dcp():
            raise ValueError(f"Convex program is not DCP.")
        solver_opts = {"use_quad_obj": False}
        chain = prob._construct_chain(solver_opts=solver_opts)
        chain.reductions = chain.reductions[:-1]
        prob = chain.apply(prob)[0]

        # Define cost of conic program.
        cd = prob.c.toarray().flatten()
        self.c = cd[:-1]
        self.d = cd[-1]

        # Define constraints of conic program. (Sparse matrices are converted to
        # dense numpy arrays, since keeping them sparse seems to make things
        # slower.)
        cols = prob.c.shape[0]
        Ab = prob.A.toarray().reshape((-1, cols), order='F')
        self.A = Ab[:, :-1]
        self.b = Ab[:, -1]
        self.K = [(type(c), c.size) for c in prob.constraints]

        # Dictionary that maps the id of a variable to its size. The entries
        # are ordered as the variables appear in the vector x.
        key = lambda item: item[1]
        self.id_to_size =  dict(sorted(prob.var_id_to_col.items(), key=key))
        for variable in prob.variables:
            self.id_to_size[variable.id] = variable.size

    def check_variable_copies(self, variables):

        # Check that number of variables is equal.
        if len(variables) != len(self.variables):
            raise ValueError(
                f"Passed list of {len(variables)} variables, while this program "
                f"has {len(self.variables)} variables.")
        
        # Check that shape of each variable is equal.
        for new_var, old_var in zip(variables, self.variables):
            if new_var.shape != old_var.shape:
                raise ValueError(
                    f"Variable {new_var.id} has shape {new_var.shape}, "
                    f"while it must have shape {old_var.shape}.")
            
            # Check symmetry attribute.
            if new_var.is_symmetric() != old_var.is_symmetric():
                raise ValueError(
                    f"Symmetry of variable {new_var.id} is incorrect.")
            
    def variables_to_x(self, variables):

        # Deal with trivial case.
        if len(variables) == 0:
            return np.array([])

        # Check that passed variables are coherent with program variables.
        self.check_variable_copies(variables)

        # Get ids of variables of original convex program.
        variable_ids = [var.id for var in self.variables]

        # Assemble vector x one piece at the time.
        x = []
        for id, size in self.id_to_size.items():
            if id in variable_ids:
                variable = variables[variable_ids.index(id)]
                xi = self.variable_to_xi(variable)
            else:
                xi = cp.Variable(size)
            x.append(xi)

        return cp.hstack(x)

    def cost_homogenization(self, x, y):
        return self.c @ x + self.d * y

    def constraint_homogenization(self, x, y):
        constraints = []
        z = self.A @ x + self.b * y
        start = 0
        for Ki, size in self.K:
            stop = start + size
            zi = z[start:stop]
            constraints.append(self.constrain_in_cone(zi, Ki))
            start = stop
        return constraints
    
    def _solve(self, y=1, **kwargs):
        """
        This method is not used in the library but is useful for testing.
        """
        x = cp.Variable(self.c.size)
        cost = self.cost_homogenization(self, x, y)
        constraints = self.constraint_homogenization(x, y)
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(**kwargs)
        return prob.value, x.value

    @staticmethod
    def variable_to_xi(variable):

        # One dimensional vector.
        if len(variable.shape) == 1:
            return variable
        
        # Asymmetric matrix.
        if not variable.is_symmetric():
            return variable.flatten(order='F')
        
        # Symmetric matrix.
        n = variable.shape[0]
        return variable[np.triu_indices(n)]
    
    @staticmethod
    def constrain_in_cone(z, K):

        # Linear constraints.
        if K in [cp.Zero, cp.NonNeg]:
            return K(z)
        elif K == cp.NonPos: # NonPos will be deprecated.
            return cp.NonNeg(- z)
        
        # Second order cone constraint.
        elif K == cp.SOC:
            return K(z[0], z[1:])
        
        # Semidefinite constraint.
        elif K == cp.PSD:
            n = round(z.size ** .5)
            z_mat = cp.reshape(z, (n, n), order='F')
            return K(z_mat)
        
        # Exponential cone constraint.
        elif K == cp.ExpCone:
            z_mat = z.reshape((3, -1), order='C')
            return K(*z_mat)
        
        # TODO: support additional cone constraints.
        else:
            raise NotImplementedError
