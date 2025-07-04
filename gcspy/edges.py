import numpy as np
import cvxpy as cp
from gcspy.programs import ConicProgram, ConvexProgram

class ConicEdge(ConicProgram):

    def __init__(self, tail, head, c, d, A, b, K, id_to_cols=None, y=None):

        # check inputs
        super().__init__(c, d, A, b, K, id_to_cols)
        self.additional_size = self.size - tail.size - head.size
        if self.additional_size < 0:
            raise ValueError(
                f"Size mismatch: edge.size = {self.size}, tail.size = "
                f"{tail.size}, head.size = {head.size}. Size of the edge must "
                "be larger than the sum of tail and head sizes.")
        
        # store data
        self.tail = tail
        self.head = head
        self.name = (tail.name, head.name)
        self.y = cp.Variable() if y is None else y

    def check_vector_sizes(self, xv, xw, xe):
        xs = [xv, xw, xe]
        expected_sizes = [self.tail.size, self.head.size, self.additional_size]
        for x, size in zip(xs, expected_sizes):
            if x.size != size:
                ValueError(
                    f"Size mismatch: x.size = {x.size}.  Expected size {size}.")

    def evaluate_cost(self, xv, xw, xe, t=1):
        self.check_vector_sizes(xv, xw, xe)
        x = cp.hstack((xv, xw, xe))
        return super().evaluate_cost(x, t)
        
    def evaluate_constraints(self, xv, xw, xe, t=1):
        self.check_vector_sizes(xv, xw, xe)
        x = cp.hstack((xv, xw, xe))
        return super().evaluate_constraints(x, t)
    
class ConvexEdge(ConvexProgram):

    def __init__(self, tail, head):
        super().__init__()
        self.tail = tail
        self.head = head
        self.name = (self.tail.name, self.head.name)
        self.y = cp.Variable()

    def to_conic(self, conic_tail, conic_head):

        # sizes of the tail and head conic programs
        tail_size = list(conic_tail.id_to_cols.values())[-1].stop
        head_size = list(conic_head.id_to_cols.values())[-1].stop

        # translate edge program to conic program
        conic_program = super().to_conic()

        # include tail and head variables in dictionary id_to_cols
        # order of variables is (x_tail, x_head, x_edge)
        id_to_cols = conic_tail.id_to_cols.copy()
        for id, r in conic_head.id_to_cols.items():
            start = r.start + tail_size
            stop = r.stop + tail_size
            id_to_cols[id] = range(start, stop)
        start = tail_size + head_size
        for id, r in conic_program.id_to_cols.items():
            if not id in id_to_cols:
                stop = start + len(r)
                id_to_cols[id] = range(start, stop)
                start = stop

        # reorder matrices and extend them with zeros according to the extended
        # id_to_cols dictionary
        c = np.zeros(stop)
        A = [np.zeros((small_Ai.shape[0], stop)) for small_Ai in conic_program.A]
        for id, r in conic_program.id_to_cols.items():
            c[id_to_cols[id]] = conic_program.c[r]
            for Ai, small_Ai in zip(A, conic_program.A):
                Ai[:, id_to_cols[id]] = small_Ai[:, r]

        # construct conic edge
        return ConicEdge(
            conic_tail,
            conic_head,
            c,
            conic_program.d,
            A,
            conic_program.b,
            conic_program.K,
            id_to_cols,
            self.y)

    def check_variables_are_defined(self, variables):
        defined_variables = self.variables + self.tail.variables + self.head.variables
        super().check_variables_are_defined(variables, defined_variables)