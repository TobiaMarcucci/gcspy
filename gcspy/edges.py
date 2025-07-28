import numpy as np
import cvxpy as cp
from gcspy.programs import ConicProgram, ConvexProgram

class ConicEdge(ConicProgram):

    def __init__(self, tail, head, c, d, A, b, K, id_to_range=None):

        # check inputs
        super().__init__(c, d, A, b, K, id_to_range)
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
                
    def stack_variables(self, xv, xw, xe):
        if self.slack_size == 0:
            return cp.hstack((xv, xw))
        else:
            return cp.hstack((xv, xw, xe))

    def cost_homogenization(self, xv, xw, xe, y):
        self.check_vector_sizes(xv, xw, xe)
        x = self.stack_variables(xv, xw, xe)
        return super().cost_homogenization(x, y)
        
    def constraint_homogenization(self, xv, xw, xe, y):
        self.check_vector_sizes(xv, xw, xe)
        x = self.stack_variables(xv, xw, xe)
        return super().constraint_homogenization(x, y)
    
class ConvexEdge(ConvexProgram):

    def __init__(self, tail, head):
        super().__init__()
        self.tail = tail
        self.head = head
        self.name = (self.tail.name, self.head.name)

    def to_conic(self, conic_tail, conic_head):

        # Sizes of the tail and head conic programs.
        tail_size = list(conic_tail.id_to_range.values())[-1].stop
        head_size = list(conic_head.id_to_range.values())[-1].stop

        # Translate edge program to conic program.
        conic_program = super().to_conic()

        # Include tail and head variables in id_to_range. Variable
        # order is (x_tail, x_head, x_edge).
        id_to_range = conic_tail.id_to_range.copy()
        for id, r in conic_head.id_to_range.items():
            start = r.start + tail_size
            stop = r.stop + tail_size
            id_to_range[id] = range(start, stop)
        start = tail_size + head_size
        for id, r in conic_program.id_to_range.items():
            if not id in id_to_range:
                stop = start + len(r)
                id_to_range[id] = range(start, stop)
                start = stop

        # Reorder matrices and extend them with zeros according to the extended
        # id_to_range dictionary.
        c = np.zeros(stop)
        A = np.zeros((conic_program.A.shape[0], stop))
        for id, r in conic_program.id_to_range.items():
            c[id_to_range[id]] = conic_program.c[r]
            A[:, id_to_range[id]] = conic_program.A[:, r]

        # construct conic edge
        return ConicEdge(
            conic_tail,
            conic_head,
            c,
            conic_program.d,
            A,
            conic_program.b,
            conic_program.K,
            id_to_range)

    def check_variables_are_defined(self, variables):
        defined_variables = self.variables + self.tail.variables + self.head.variables
        super().check_variables_are_defined(variables, defined_variables)