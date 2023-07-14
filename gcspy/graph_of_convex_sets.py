import cvxpy as cp
import numpy as np
from gcspy.programs import ConvexProgram, ConicProgram
from gcspy.problems.shortest_path import shortest_path


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

    def incoming_edges(self, vertex):
        return [edge for edge in self.edges if edge.head == vertex]

    def incoming_indices(self, vertex):
        return [k for k, edge in enumerate(self.edges) if edge.head == vertex]

    def outgoing_edges(self, vertex):
        return [edge for edge in self.edges if edge.tail == vertex]

    def outgoing_indices(self, vertex):
        return [k for k, edge in enumerate(self.edges) if edge.tail == vertex]

    def incident_edges(self, vertex):
        return self.incoming_edges(vertex) + self.outgoing_edges(vertex)

    def incident_indices(self, vertex):
        return self.incoming_indices(vertex) + self.outgoing_indices(vertex)

    def num_vertices(self):
        return len(self.vertices)

    def num_edges(self):
        return len(self.edges)

    def to_conic(self):
        for vertex in self.vertices:
            vertex.to_conic()
        for edge in self.edges:
            edge.to_conic()

    def shortest_path(self, s, t, **kwargs):
        return shortest_path(self, s, t, **kwargs)
