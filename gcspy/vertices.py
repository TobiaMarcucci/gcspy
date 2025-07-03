import cvxpy as cp
from gcspy.programs import ConicProgram, ConvexProgram

class ConicVertex(ConicProgram):

    def __init__(self, name, c, d, A, b, K, binary=True):
        super.__init__(c, d, A, b, K)
        self.name = name
        self.binary = binary
        self.y = cp.Variable(boolean=binary)

class ConvexVertex(ConvexProgram):

    def __init__(self, name, binary=True):
        super().__init__()
        self.name = name
        self.binary = binary
        self.y = cp.Variable(boolean=binary)
        self.conic = None
        self.id_to_cols = None

    def to_conic(self):
        conic_program, self.id_to_cols = super().to_conic()
        self.conic = ConicVertex(
            self.name,
            conic_program.c,
            conic_program.d,
            conic_program.A,
            conic_program.b,
            conic_program.K,
            self.binary,
        )
