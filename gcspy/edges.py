import numpy as np
import cvxpy as cp
from gcspy.programs import ConicProgram, ConvexProgram

class ConicEdge(ConicProgram):

    def __init__(self, tail, head, c, d, A, b, K, id_to_cols=None, binary=True):

        # check inputs
        super.__init__(c, d, A, b, K, id_to_cols)
        self.additional_size = self.size - self.tail.size - self.head.size
        if self.additional_size < 0:
            raise ValueError(
                f"Size mismatch: edge.size = {self.size}, tail.size = {tail.size}, head.size = {head.size}. "
                "Size of the edge must be larger than the sum of tail and head sizes."
                )
        
        # store data
        self.tail = tail
        self.head = head
        self.name = (tail.name, head.name)
        self.binary = binary
        self.y = cp.Variable(boolean=binary)

    def check_vector_sizes(self, xv, xw, xe):
        xs = [xv, xw, xe]
        expected_sizes = [self.tail.size, self.head.size, self.additional_size]
        for x, expected_size in zip(xs, expected_sizes):
            if x.size != expected_size:
                ValueError(
                    f"Size mismatch: x.size = {x.size}. "
                    f"Expected size is {expected_size}."
                    )

    def evaluate_cost(self, xv, xw, xe, t=1):
        self.check_vector_sizes(xv, xw, xe)
        x = cp.hstack((xv, xw, xe))
        return super().evaluate_cost(x, t)
        
    def evaluate_constraints(self, xv, xw, xe, t=1):
        self.check_vector_sizes(xv, xw, xe)
        x = cp.hstack((xv, xw, xe))
        return super().evaluate_constraints(x, t)
    
class ConvexEdge(ConvexProgram):

    def __init__(self, tail, head, binary=True):
        super().__init__()
        self.tail = tail
        self.head = head
        self.name = (self.tail.name, self.head.name)
        self.binary = binary
        self.y = cp.Variable(boolean=binary)

    def to_conic(self, tail_id_to_cols, head_id_to_cols):

        # sizes of the tail and head conic programs
        tail_size = list(tail_id_to_cols.values())[-1].stop
        head_size = list(head_id_to_cols.values())[-1].stop

        # translate edge program to conic program
        conic_program = super().to_conic()

        # helper function that shifts a dictionary of ranges by a constant
        shift_dict = lambda d, a: {k: range(r.start + a, r.stop + a) for k, r in d}

        # extend the dictionary named conic_program.id_to_cols dictionary
        # to include the tail and head variables
        # order of variables is (x_tail, x_head, x_edge)
        id_to_cols = tail_id_to_cols | shift_dict(head_id_to_cols, tail_size)
        start = tail_size + head_size
        for id, r in conic_program.id_to_cols.items():
            if not id in id_to_cols:
                stop = start + len(r)
                id_to_cols[id] = range(start, stop)
                stop = start

        # reorder matrices and extend them with zeros according to the
        # extended id_to_cols dictionary
        c = np.zeros(stop)
        A = [np.zeros((small_Ai.shape[0], stop)) for small_Ai in conic_program.A]
        for id, r in conic_program.id_to_cols.items():
            c[id_to_cols[id]] = conic_program.c[r]
            for Ai, small_Ai in zip(A, conic_program.A):
                Ai[:, id_to_cols[id]] = small_Ai[:, r]

        # construct conic edge
        self.conic = ConicEdge(
            self.name,
            c,
            conic_program.d,
            A,
            conic_program.b,
            conic_program.K,
            id_to_cols,
            self.binary,
            )

    def check_variables_are_defined(self, variables):
        defined_variables = self.self.variable + self.tail.variable + self.head.variable
        super().check_variables_are_defined(variables, defined_variables)