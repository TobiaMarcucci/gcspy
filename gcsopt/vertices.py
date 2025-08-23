from gcsopt.programs import ConicProgram, ConvexProgram

class ConicVertex(ConicProgram):

    def __init__(self, name, size, id_to_range=None, binary_variable=None):
        super().__init__(size, id_to_range, binary_variable)
        self.name = name

class ConvexVertex(ConvexProgram):

    def __init__(self, name):
        super().__init__()
        self.name = name

    def to_conic(self):
        conic_program = super().to_conic()
        conic_vertex = ConicVertex(
            self.name,
            conic_program.size,
            conic_program.id_to_range,
            conic_program.binary_variable)
        conic_vertex.add_cost(conic_program.c, conic_program.d)
        conic_vertex.add_constraints(conic_program.A, conic_program.b, conic_program.K)
        return conic_vertex
