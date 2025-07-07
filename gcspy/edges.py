import numpy as np
import cvxpy as cp
from gcspy.programs import ConicProgram, ConvexProgram

class ConicEdge(ConicProgram):

    def __init__(self, tail, head, c, d, A, b, K, convex_id_to_conic_idx=None):

        # check inputs
        super().__init__(c, d, A, b, K, convex_id_to_conic_idx)
        self.slack_size = self.size - tail.size - head.size
        if self.slack_size < 0:
            raise ValueError(
                f"Size mismatch: edge.size = {self.size}, tail.size = "
                f"{tail.size}, head.size = {head.size}. Size of the edge must "
                "be larger than the sum of tail and head sizes.")
        
        # store data
        self.tail = tail
        self.head = head
        self.name = (tail.name, head.name)

    def check_vector_sizes(self, xv, xw, xe):
        xs = [xv, xw, xe]
        sizes = [self.tail.size, self.head.size, self.slack_size]
        for x, size in zip(xs, sizes):
            if x.size != size:
                ValueError(
                    f"Size mismatch: x.size = {x.size}.  Expected size {size}.")

    def evaluate_cost(self, xv, xw, xe, y=1):
        self.check_vector_sizes(xv, xw, xe)
        x = cp.hstack((xv, xw, xe))
        return super().evaluate_cost(x, y)
        
    def evaluate_constraints(self, xv, xw, xe, y=1):
        self.check_vector_sizes(xv, xw, xe)
        x = cp.hstack((xv, xw, xe))
        return super().evaluate_constraints(x, y)
    
class ConvexEdge(ConvexProgram):

    def __init__(self, tail, head):
        super().__init__()
        self.tail = tail
        self.head = head
        self.name = (self.tail.name, self.head.name)

    def to_conic(self, conic_tail, conic_head):

        # sizes of the tail and head conic programs
        tail_size = list(conic_tail.convex_id_to_conic_idx.values())[-1].stop
        head_size = list(conic_head.convex_id_to_conic_idx.values())[-1].stop

        # translate edge program to conic program
        conic_program = super().to_conic()

        # include tail and head variables in dictionary convex_id_to_conic_idx
        # order of variables is (x_tail, x_head, x_edge)
        convex_id_to_conic_idx = conic_tail.convex_id_to_conic_idx.copy()
        for id, r in conic_head.convex_id_to_conic_idx.items():
            start = r.start + tail_size
            stop = r.stop + tail_size
            convex_id_to_conic_idx[id] = range(start, stop)
        start = tail_size + head_size
        for id, r in conic_program.convex_id_to_conic_idx.items():
            if not id in convex_id_to_conic_idx:
                stop = start + len(r)
                convex_id_to_conic_idx[id] = range(start, stop)
                start = stop

        # reorder matrices and extend them with zeros according to the extended
        # convex_id_to_conic_idx dictionary
        c = np.zeros(stop)
        A = [np.zeros((small_Ai.shape[0], stop)) for small_Ai in conic_program.A]
        for id, r in conic_program.convex_id_to_conic_idx.items():
            c[convex_id_to_conic_idx[id]] = conic_program.c[r]
            for Ai, small_Ai in zip(A, conic_program.A):
                Ai[:, convex_id_to_conic_idx[id]] = small_Ai[:, r]

        # construct conic edge
        return ConicEdge(
            conic_tail,
            conic_head,
            c,
            conic_program.d,
            A,
            conic_program.b,
            conic_program.K,
            convex_id_to_conic_idx)

    def check_variables_are_defined(self, variables):
        defined_variables = self.variables + self.tail.variables + self.head.variables
        super().check_variables_are_defined(variables, defined_variables)