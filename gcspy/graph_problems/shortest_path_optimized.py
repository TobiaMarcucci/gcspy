import cvxpy as cp
import numpy as np

class ConicShortestPathProblem:

    def __init__(self, conic_graph, source_name, target_name, binary):

        # store data
        self.conic_graph = conic_graph
        self.source_name = source_name
        self.target_name = target_name

        # variables
        self.ye = cp.Variable(conic_graph.num_edges(), boolean=binary)
        safe_variable = lambda size: cp.Variable(size) if size > 0 else np.array([])
        self.ze = np.array([safe_variable(edge.slack_size) for edge in conic_graph.edges])
        self.ze_tail = np.array([cp.Variable(edge.tail.size) for edge in conic_graph.edges])
        self.ze_head = np.array([cp.Variable(edge.head.size) for edge in conic_graph.edges])

        # costs and constraints on the edges
        self.cost = 0
        self.constraints = []
        for k, edge in enumerate(conic_graph.edges):
            self.cost += edge.evaluate_cost(self.ze_tail[k], self.ze_head[k], self.ze[k], self.ye[k])
            self.constraints += edge.evaluate_constraints(self.ze_tail[k], self.ze_head[k], self.ze[k], self.ye[k])
            self.constraints += edge.tail.evaluate_constraints(self.ze_tail[k], self.ye[k])
            self.constraints += edge.head.evaluate_constraints(self.ze_head[k], self.ye[k])

        # shortest path constraints
        for vertex in conic_graph.vertices:
            inc = conic_graph.incoming_edge_indices(vertex)
            out = conic_graph.outgoing_edge_indices(vertex)

            # source cost and constraints
            if vertex.name == source_name:
                yv = sum(self.ye[out])
                zv = sum(self.ze_tail[out])
                self.cost += vertex.evaluate_cost(zv, yv)
                self.constraints.append(yv == 1)
                if inc:
                    self.constraints.append(self.ye[inc] == 0)

            # target cost and constraints
            elif vertex.name == target_name:
                yv = sum(self.ye[inc])
                zv = sum(self.ze_head[inc])
                self.cost += vertex.evaluate_cost(zv, yv)
                self.constraints.append(yv == 1)
                if out:
                    self.constraints.append(self.ye[out] == 0)

            # cost and constraints on other vertices
            else:
                zv_inc = sum(self.ze_head[inc])
                zv_out = sum(self.ze_tail[out])
                yv_inc = sum(self.ye[inc])
                yv_out = sum(self.ye[out])
                self.cost += vertex.evaluate_cost(zv_out, yv_out)
                self.constraints += [zv_inc == zv_out, yv_inc == yv_out, yv_out <= 1]
                
    def solve(self, tol=1e-4, **kwargs):

        # solve problem
        prob = cp.Problem(cp.Minimize(self.cost), self.constraints)
        prob.solve(**kwargs)

        # if problem is not solved to optimality
        if prob.status != 'optimal':
            xv = np.full(self.conic_graph.num_vertices(), None)
            xe = np.full(self.conic_graph.num_edges(), None)
            yv = xv
            ye = xe

        # if problem is solved to optimality
        else:
            ye = self.ye.value
            yv = np.zeros(self.conic_graph.num_vertices())
            xv = []
            for i, vertex in enumerate(self.conic_graph.vertices):
                if vertex.name == self.source_name:
                    out = self.conic_graph.outgoing_edge_indices(vertex)
                    xv.append(sum(self.ze_tail[out]).value)
                    yv[i] = 1
                else:
                    inc = self.conic_graph.incoming_edge_indices(vertex)
                    yv[i] = sum(ye[inc])
                    if yv[i] > tol:
                        xv.append(sum(self.ze_head[inc]).value / yv[i])
                    else:
                        xv.append(None)
            xe = []
            for z, y in zip(self.ze, ye):
                if y <= tol:
                    xe.append(None)
                elif z.size == 0:
                    xe.append(np.array([]))
                else:
                    xe.append(z.value / y)

        return prob, xv, yv, xe, ye