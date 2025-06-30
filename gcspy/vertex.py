from gcspy.convex_program import ConvexProgram

class Vertex(ConvexProgram):

    def __init__(self, name=""):
        super().__init__()
        self.name = name
        self.conic_program = None
        self.get_variable_value = None

    def to_conic_program(self):
        # overwrites parent class method and stores the result
        # instead of returning it
        self.conic_program, self.get_variable_value = super().to_conic_program()

    def get_feasible_point(self, **kwargs):
        # finds a feasible point without changing the value
        # of the variables in this convex program
        original_cost = self.cost
        original_values = [variable.value for variable in self.variables]
        self.cost = 0
        self._solve(**kwargs)
        feasible_point = [variable.value for variable in self.variables]
        self.cost = original_cost
        for variable, value in zip(self.variables, original_values):
            variable.value = value
        return feasible_point
