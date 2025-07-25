import cvxpy as cp
from numbers import Number
from gcspy.conic_program import ConicProgram

class ConvexProgram:

    supported_attributes = ["nonneg", "nonpos", "symmetric", "PSD", "NSD"]

    def __init__(self):
        self.variables = []
        self.cost = 0
        self.constraints = []
        self.binary_variable = cp.Variable()

    def _solve(self, **kwargs):
        """
        This method is not used in the library but is useful for testing.
        """
        prob = cp.Problem(cp.Minimize(self.cost), self.constraints)
        prob.solve(**kwargs)
        variable_values = [variable.value for variable in self.variables]
        return prob.value, variable_values

    def add_variable(self, shape, **kwargs):
        for attribute in kwargs:
            if not attribute in self.supported_attributes:
                raise ValueError(f"Attribute {attribute} is not supported.")
        variable = cp.Variable(shape, **kwargs)
        self.variables.append(variable)
        return variable
    
    def check_variables_are_defined(self, variables, defined_variables=None):
        if defined_variables is None:
            defined_variables = self.variables
        defined_ids = [variable.id for variable in defined_variables]
        for variable in variables:
            if variable.id not in defined_ids:
                raise ValueError(f"Variable {variable} is not defined.")
    
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

    def homogenization(self, variables, y):
        conic_program = ConicProgram(self)
        cost = conic_program.cost_homogenization(variables, y)
        constraints = conic_program.constraint_homogenization(variables, y)
        return cost, constraints
    
    def copy_variables(self):
        """
        We do not want to copy over, e.g., the nonnegative attribute since the
        nonnegativity of the copied variable will be enforced by the conic
        constraints. But we want to copy over the symmetric attribute since that
        is a constraint that is not in the conic program.
        """
        return [cp.Variable(var.shape, symmetric=var.is_symmetric()) for var in self.variables]