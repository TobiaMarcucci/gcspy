import cvxpy as cp
from gcspy.programs import ConicProgram, ConvexProgram

class ConicVertex(ConicProgram):

    def __init__(self, name, c, d, A, b, K, convex_id_to_conic_idx=None):
        super().__init__(c, d, A, b, K, convex_id_to_conic_idx)
        self.name = name

class ConvexVertex(ConvexProgram):

    def __init__(self, name):
        super().__init__()
        self.name = name

    def to_conic(self):
        conic_program = super().to_conic()
        return ConicVertex(
            self.name,
            conic_program.c,
            conic_program.d,
            conic_program.A,
            conic_program.b,
            conic_program.K,
            conic_program.convex_id_to_conic_idx)