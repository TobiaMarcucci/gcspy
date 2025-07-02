import cvxpy as cp
from gcspy.programs import ConicProgram, ConvexProgram

class ConicVertex(ConicProgram):

    def __init__(self, name, c, d, A, b, K, boolean=True):
        super.__init__(c, d, A, b, K)
        self.name = name
        self.y = cp.Variable(boolean)

class ConvexVertex(ConvexProgram):

    def __init__(self, name, boolean=True):
        super().__init__()
        self.name = name
        self.y = cp.Variable(boolean)
        self.conic = None
        self.id_to_cols = None

    def to_conic_vertex(self):
        conic_program, self.id_to_cols = super().to_conic_program()
        c, d, A, b, K = conic_program.c, conic_program.d, conic_program.A, conic_program.b, conic_program.K
        boolean = self.y.attributes['boolean']
        self.conic = ConicVertex(self.name, c, d, A, b, K, boolean)
