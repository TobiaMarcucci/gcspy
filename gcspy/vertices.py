import cvxpy as cp
from gcspy.programs import ConicProgram, ConvexProgram

class ConicVertex(ConicProgram):

    def __init__(self, name, c, d, A, b, K, id_to_cols=None, y=None):
        super().__init__(c, d, A, b, K, id_to_cols)
        self.name = name
        self.y = cp.Variable() if y is None else y

class ConvexVertex(ConvexProgram):

    def __init__(self, name):
        super().__init__()
        self.name = name
        self.y = cp.Variable()

    def to_conic(self):
        conic_program = super().to_conic()
        return ConicVertex(
            self.name,
            conic_program.c,
            conic_program.d,
            conic_program.A,
            conic_program.b,
            conic_program.K,
            conic_program.id_to_cols,
            self.y)
