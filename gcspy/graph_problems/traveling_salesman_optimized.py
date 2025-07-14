import cvxpy as cp
import numpy as np
from itertools import combinations

class ConicTravelingSalesmanProblem:

    def __init__(self, conic_graph, subtour_elimination, binary):

        # store data
        self.conic_graph = conic_graph

        # collect vectors of binary variables
        self.ye = cp.Variable(self.conic_graph.num_edges(), boolean=binary)

        # continuous variables for the edges
        safe_variable = lambda size: cp.Variable(size) if size > 0 else np.array([])
        self.ze = np.array([safe_variable(edge.slack_size) for edge in self.conic_graph.edges])
        self.ze_tail = np.array([cp.Variable(edge.tail.size) for edge in self.conic_graph.edges])
        self.ze_head = np.array([cp.Variable(edge.head.size) for edge in self.conic_graph.edges])

        # cost of the edges
        self.cost = 0
        for k, edge in enumerate(self.conic_graph.edges):
            self.cost += edge.evaluate_cost(self.ze_tail[k], self.ze_head[k], self.ze[k], self.ye[k])

            # spread vertex cost over the edges
            # since cost is linear this does not change things
            self.cost += edge.tail.evaluate_cost(self.ze_tail[k])

        # constraints on the edges
        self.constraints = []
        for k, edge in enumerate(self.conic_graph.edges):
            self.constraints += edge.tail.evaluate_constraints(self.ze_tail[k], self.ye[k])
            self.constraints += edge.head.evaluate_constraints(self.ze_head[k], self.ye[k])
            self.constraints += edge.evaluate_constraints(self.ze_tail[k], self.ze_head[k], self.ze[k], self.ye[k])

        # add all constraints one vertex at the time
        for v in conic_graph.vertices:
            inc = conic_graph.incoming_edge_indices(v)
            out = conic_graph.outgoing_edge_indices(v)
            self.constraints += [
                sum(self.ye[out]) == 1,
                sum(self.ye[inc]) == 1,
                sum(self.ze_head[inc]) == sum(self.ze_tail[out])]

        # subtour elimination constraints for all subsets of vertices with
        # cardinality between 2 and num_vertices - 2
        if subtour_elimination:
            for subtour_size in range(2, conic_graph.num_vertices() - 1):
                for vertices in combinations(conic_graph.vertices, subtour_size):
                    out = conic_graph.outgoing_edge_indices(vertices)
                    self.constraints.append(sum(self.ye[out]) >= 1)

    def solve(self, callback=None, tol=1e-4, **kwargs):

        # solve problem
        prob = cp.Problem(cp.Minimize(self.cost), self.constraints)
        prob.solve(**kwargs)

        # run callback if one is provided
        if callback is not None:
            while True:
                new_constraints = callback(None, self.ye)
                if len(new_constraints) == 0:
                    break
                self.constraints += new_constraints
                prob = cp.Problem(cp.Minimize(self.cost), self.constraints)
                prob.solve()

        # if problem is not solved to optimality
        if prob.status != 'optimal':
            xv = np.full(self.conic_graph.num_vertices(), None)
            xe = np.full(self.conic_graph.num_edges(), None)
            yv = xv
            ye = xe

        # if problem is solved to optimality
        else:
            yv = np.ones(self.conic_graph.num_vertices())
            ye = self.ye.value
            xv = []
            for vertex in self.conic_graph.vertices:
                inc = self.conic_graph.incoming_edge_indices(vertex)
                xv.append(sum(self.ze_head[inc]).value)
            xe = []
            for z, y in zip(self.ze, ye):
                if z.size == 0:
                    xe.append(np.array([]))
                elif y > tol:
                    xe.append(z.value / y)
                else:
                    xe.append(None)

        return prob, xv, yv, xe, ye
