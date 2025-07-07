import cvxpy as cp
import numpy as np

class ConicGraphProblem:

    def __init__(self, conic_graph, binary):

        # store data
        self.conic_graph = conic_graph
        self.define_variables(binary)
        self.define_cost()
        self.define_constraints()

    def define_variables(self, binary):

        # collect vectors of binary variables
        self.yv = cp.Variable(self.conic_graph.num_vertices(), boolean=binary)
        self.ye = cp.Variable(self.conic_graph.num_edges(), boolean=binary)

        # continuous variables for the vertices
        # these are numpy arrays since we want to access them using lists of indices
        self.xv = np.array([cp.Variable(vertex.size) for vertex in self.conic_graph.vertices])
        self.zv = np.array([cp.Variable(vertex.size) for vertex in self.conic_graph.vertices])

        # continuous variables for the edges
        self.ze = np.array([cp.Variable(edge.slack_size) for edge in self.conic_graph.edges])
        self.ze_tail = np.array([cp.Variable(edge.tail.size) for edge in self.conic_graph.edges])
        self.ze_head = np.array([cp.Variable(edge.head.size) for edge in self.conic_graph.edges])

    def define_cost(self):

        # cost of the vertices
        self.cost = 0
        for i, vertex in enumerate(self.conic_graph.vertices):
            self.cost += vertex.evaluate_cost(self.zv[i], self.yv[i])

        # cost of the edges
        for k, edge in enumerate(self.conic_graph.edges):
            self.cost += edge.evaluate_cost(self.ze_tail[k], self.ze_head[k], self.ze[k], self.ye[k])

    def define_constraints(self):

        # constraints on the vertices
        self.constraints = []
        for i, vertex in enumerate(self.conic_graph.vertices):
            self.constraints += vertex.evaluate_constraints(self.zv[i], self.yv[i])
            self.constraints += vertex.evaluate_constraints(self.xv[i] - self.zv[i], 1 - self.yv[i])
        
        # constraints on the edges
        for k, edge in enumerate(self.conic_graph.edges):
            
            # tail constraints
            x_tail = self.xv[self.conic_graph.vertex_index(edge.tail)]
            self.constraints += edge.tail.evaluate_constraints(self.ze_tail[k], self.ye[k])
            self.constraints += edge.tail.evaluate_constraints(x_tail - self.ze_tail[k], 1 - self.ye[k])

            # head constraints
            x_head = self.xv[self.conic_graph.vertex_index(edge.head)]
            self.constraints += edge.head.evaluate_constraints(self.ze_head[k], self.ye[k])
            self.constraints += edge.head.evaluate_constraints(x_head - self.ze_head[k], 1 - self.ye[k])

            # edge constraints
            self.constraints += edge.evaluate_constraints(self.ze_tail[k], self.ze_head[k], self.ze[k], self.ye[k])

    def solve(self, callback=None, tol=1e-4, **kwargs):

        # solve problem
        prob = cp.Problem(cp.Minimize(self.cost), self.constraints)
        prob.solve(**kwargs)

        # run callback if one is provided
        if callback is not None:
            while True:
                new_constraints = callback(self.yv, self.ye)
                if len(new_constraints) == 0:
                    break
                self.constraints += new_constraints
                prob = cp.Problem(cp.Minimize(self.cost), self.constraints)
                prob.solve()

        # get optimal solution
        yv = self.yv.value
        ye = self.ye.value
        xv = [x.value for x in self.xv]
        xe = []
        for (z, y) in zip(self.ze, self.ye.value):
            if y is not None and y > tol:
                xe.append(z.value / y)
            else:
                xe.append(None)

        return prob, yv, xv, ye, xe
    
class ConvexGraphProblem:

    def __init__(self, convex_graph, conic_problem_class, *args, **kwargs):

        # solve problem in conic form
        self.convex_graph = convex_graph
        self.conic_graph = convex_graph.to_conic()
        self.conic_problem = conic_problem_class(self.conic_graph, *args, **kwargs)

    def solve(self, *args, **kwargs):

        # solve conic problem
        prob, yv, xv, ye, xe = self.conic_problem.solve(*args, **kwargs)

        # set value of vertex variables
        for convex_vertex, y, x in zip(self.convex_graph.vertices, yv, xv):
            convex_vertex.binary_variable.value = y
            for convex_variable in convex_vertex.variables:
                if x is None:
                    convex_variable.value = None
                else:
                    conic_vertex = self.conic_graph.get_vertex(convex_vertex.name)
                    convex_variable.value = conic_vertex.get_convex_variable_value(convex_variable, x)

        # set value of edge variables
        for convex_edge, y, x in zip(self.convex_graph.edges, ye, xe):
            convex_edge.binary_variable.value = y
            for convex_variable in convex_edge.variables:
                if x is None:
                    convex_variable.value = None
                else:
                    conic_edge = self.conic_graph.get_edge(*convex_edge.name)
                    convex_variable.value = conic_edge.get_convex_variable_value(convex_variable, x)

        return prob