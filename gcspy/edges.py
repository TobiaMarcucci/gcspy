import numpy as np
import cvxpy as cp
from gcspy.programs import ConicProgram, ConvexProgram

class ConicEdge(ConicProgram):

    def __init__(self, tail, head, c, d, A, b, K, boolean=True):

        # check inputs
        super.__init__(c, d, A, b, K)
        if self.size < tail.size + head.size:
            raise ValueError("Size mismatch: edge.size = {self.size}, tail.size = {tail.size}, head.size = {head.size}. "
                             "Size of the edge must be larger than the sum of tail and head sizes.")
        
        # store data
        self.tail = tail
        self.head = head
        self.name = (tail.name, head.name)
        self.y = cp.Variable(boolean)
        
    def evaluate_constraints(self, xv, xw, xe, t=1):

        # check inputs
        xs = [xv, xw, xe]
        names = ["xv", "xw", "xe"]
        expected_sizes = [self.tail.size, self.head.size, self.size]
        for x, name, expected_size in zip(xs, names, expected_sizes):
            if x.size != expected_size:
                ValueError(f"Size mismatch: {name}.size = {x.size}. "
                           f"Expected size is {expected_size}.")
                
        # stack vectors and evaluate constraint
        x = cp.hstack((xv, xw, xe))
        return super().evaluate_constraints(x, t)
    
class ConvexEdge(ConvexProgram):

    def __init__(self, tail, head, boolean=True):
        super().__init__()
        self.tail = tail
        self.head = head
        self.name = (self.tail.name, self.head.name)
        self.y = cp.Variable(boolean)
        self.conic = None
        self.id_to_cols = None

    def to_conic(self):

        # trick that prevents free variables
        for variable in self.tail.variables + self.head.variables:
            if not variable.id in self.used_variable_ids:
                self.add_cost(0 * cp.sum(variable))

        # translate to conic
        conic_program, id_to_cols = super().to_conic_program()

        # include the tail and head variables in the id_to_cols dictionary
        shift_range = lambda r, a: range(r.start + a, r.stop + a)
        shift_dict = lambda d, a: {id: shift_range(r, a) for id, r in d}
        shifted_head_dict = shift_dict(self.tail.id_to_cols, self.tail.conic.size)
        self.id_to_cols = self.tail.id_to_cols | shifted_head_dict
        start = self.tail.conic.size + self.head.conic.size
        for id, r in id_to_cols.items():
            if not id in self.id_to_cols:
                stop = start + len(r)
                self.id_to_cols[id] = range(start, stop)
                stop = start

        # reorder matrices
        c = np.zeros(stop)
        for id, r in self.id_to_cols.items():
            if id in id_to_cols:
                c[r] = conic_program.c[id_to_cols[id]]
        A = []
        for Ai in conic_program.A:
            A.append(np.zeros(Ai.shape[0], stop))
            for id, r in self.id_to_cols.items():
                if id in id_to_cols:
                    A[-1][:, r] = Ai[:, id_to_cols[id]]

        # construct conic edge
        d, b, K = conic_program.d, conic_program.b, conic_program.K
        boolean = self.y.attributes['boolean']
        self.conic = ConicEdge(self.name, c, d, A, b, K, boolean)
    
    def _check_no_external_variables(self, variables):
        # TODO: do I need this overwrite?
        variable_ids = self.variable_ids + self.tail.variable_ids + self.head.variable_ids
        for variable in variables:
            if not variable.id in variable_ids:
                raise ValueError(f"Variable {variable} does not belong to this convex program.")