import cvxpy as cp
from gcspy.programs import ConicProgram, ConvexProgram

class ConicVertex(ConicProgram):

    def __init__(self, name, c, d, A, b, K, id_to_cols=None, binary=True):
        super.__init__(c, d, A, b, K, id_to_cols)
        self.name = name
        self.binary = binary
        self.y = cp.Variable(boolean=binary)

class ConvexVertex(ConvexProgram):

    def __init__(self, name, binary=True):
        super().__init__()
        self.name = name
        self.binary = binary
        self.y = cp.Variable(boolean=binary)

    def to_conic(self):
        conic_program = super().to_conic()
        conic_vertex = ConicVertex(
            self.name,
            conic_program.c,
            conic_program.d,
            conic_program.A,
            conic_program.b,
            conic_program.K,
            conic_program.id_to_cols,
            self.binary,
        )
        return conic_vertex
