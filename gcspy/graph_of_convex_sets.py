import cvxpy as cp
import numpy as np
from collections.abc import Iterable
from gcspy.programs import ConvexProgram, ConicProgram
from gcspy.graph_problems import graph_problem, shortest_path, traveling_salesman


class Vertex(ConvexProgram):

    def __init__(self, name=""):
        super().__init__()
        self.name = name
        self.value = None

    def _verify_variables(self, variables):
        ids0 = {variable.id for variable in self.variables}
        ids1 = {variable.id for variable in variables}
        if not ids0 >= ids1:
            raise ValueError("A variable does not belong to this vertex.")

    def get_feasible_point(self):
        values = [variable.value for variable in self.variables]
        prob = cp.Problem(cp.Minimize(0), self.constraints)
        prob.solve()
        feasible_point = [variable.value for variable in self.variables]
        for variable, value in zip(self.variables, values):
            variable.value = value
        return feasible_point


class Edge(ConvexProgram):

    def __init__(self, tail, head):
        super().__init__()
        self.tail = tail
        self.head = head
        self.conic_program = None
        self.name = (self.tail.name, self.head.name)
        self.value = None

    def _verify_variables(self, variables):
        edge_variables = self.variables + self.tail.variables + self.head.variables
        ids0 = {variable.id for variable in edge_variables}
        ids1 = {variable.id for variable in variables}
        if not ids0 >= ids1:
            raise ValueError("A variable does not belong to this edge.")


class GraphOfConvexSets:

    def __init__(self):
        self.vertices = []
        self.edges = []
        
    def add_vertex(self, name=""):
        vertex = Vertex(name)
        self.vertices.append(vertex)
        return vertex

    def add_edge(self, tail, head):
        edge = Edge(tail, head)
        self.edges.append(edge)
        return edge

    def get_edge(self, tail, head):
        for edge in self.edges:
            if edge.tail == tail and edge.head == head:
                return edge

    def get_vertex_by_name(self, name):
        for vertex in self.vertices:
            if vertex.name == name:
                return vertex

    def get_edge_by_name(self, tail_name, head_name):
        for edge in self.edges:
            if edge.tail.name == tail_name and edge.head.name == head_name:
                return edge

    def incoming_edges(self, v):
        if isinstance(v, Vertex):
            return [e for e in self.edges if e.head == v]
        if isinstance(v, Iterable):
            return [e for e in self.edges if e.head in v and e.tail not in v]

    def incoming_indices(self, v):
        if isinstance(v, Vertex):
            return [k for k, e in enumerate(self.edges) if e.head == v]
        if isinstance(v, Iterable):
            return [k for k, e in enumerate(self.edges) if e.head in v and e.tail not in v]

    def outgoing_edges(self, v):
        if isinstance(v, Vertex):
            return [e for e in self.edges if e.tail == v]
        if isinstance(v, Iterable):
            return [e for e in self.edges if e.tail in v and e.head not in v]

    def outgoing_indices(self, v):
        if isinstance(v, Vertex):
            return [k for k, e in enumerate(self.edges) if e.tail == v]
        if isinstance(v, Iterable):
            return [k for k, e in enumerate(self.edges) if e.tail in v and e.head not in v]

    def incident_edges(self, v):
        return self.incoming_edges(v) + self.outgoing_edges(v)

    def incident_indices(self, v):
        return self.incoming_indices(v) + self.outgoing_indices(v)

    def num_vertices(self):
        return len(self.vertices)

    def num_edges(self):
        return len(self.edges)

    def to_conic(self):
        for vertex in self.vertices:
            vertex.to_conic()
        for edge in self.edges:
            edge.to_conic()

    def shortest_path(self, s, t):
        return graph_problem(self, shortest_path, s, t)

    def traveling_salesman(self):
        return graph_problem(self, traveling_salesman)

    def graphviz(self):
        from gcspy.plot_utils import graphviz_gcs
        return graphviz_gcs(self)

    def plot_2d(self, **kwargs):
        from gcspy.plot_utils import plot_gcs_2d
        return plot_gcs_2d(self, **kwargs)

    def plot_subgraph_2d(self):
        from gcspy.plot_utils import plot_subgraph_2d
        return plot_subgraph_2d(self) 
