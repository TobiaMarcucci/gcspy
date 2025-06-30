import numpy as np
import cvxpy as cp
from numbers import Number

class ConicProgram:

    def __init__(self, c, d, A, b, K):
        if len(set([len(A), len(b), len(K)])) != 1:
            raise ValueError('A, b, and K must have equal length.')
        for Ai, bi in zip(A, b):
            if Ai.shape[0] != bi.size:
                raise ValueError('A and b must have coherent sizes.')
            if Ai.shape[1] != c.size:
                raise ValueError('A and c must have coherent sizes.')
        self.c = c
        self.d = d
        self.A = A
        self.b = b
        self.K = K
        self.n = len(c)

    def evaluate_cost(self, x, t=1):
        return self.c @ x + self.d * t
    
    def evaluate_constraints(self, x, t=1):
        return [Ai @ x + bi * t for Ai, bi in zip(self.A, self.b)]
    
    def get_symbolic_constraints(self, x, t=1):
        z = self.evaluate_constraints(x, t)
        return [self.constrain_in_cone(zi, Ki) for zi, Ki in zip(z, self.K)]
    
    @staticmethod
    def constrain_in_cone(z, K):
        if K == cp.constraints.Zero:
             return cp.constraints.Zero(z)
        elif K == cp.constraints.NonNeg:
             return cp.constraints.NonNeg(z)
        elif K == cp.constraints.NonPos:
             return cp.constraints.NonNeg(- z)
        elif K == cp.constraints.SOC:
            return cp.constraints.SOC(z[0], z[1:])
        elif K == cp.constraints.PSD:
            n = round(np.sqrt(z.size))
            z_mat = cp.reshape(z, (n, n))
            return cp.constraints.PSD(z_mat)
        elif K == cp.constraints.ExpCone:
            z_mat = z.reshape((3, -1), order='C')
            return cp.constraints.ExpCone(*z_mat)
        else:
            raise NotImplementedError
        
    def _solve(self, x=None, t=1):
        if self.n == 0: # corner case with no variables
            return self.d, np.array([])
        if x is None:
            x = cp.Variable(self.n)
        cost = self.evaluate_cost(x, t)
        constraints = self.get_symbolic_constraints(x, t)
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve()
        
        return prob.value, x.value