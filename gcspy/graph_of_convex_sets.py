import cvxpy as cp
import graphviz as gv


class ConvexProgram:

    var_attributes = ["nonneg", "nonpos", "symmetric", "diag", "PSD", "NSD"]

    def __init__(self):
        self.variables = []
        self.constraints = []
        self.cost = 0

    def add_variable(self, shape, **kwargs):
        for attribute in kwargs:
            if not attribute in self.var_attributes:
                raise ValueError(f"Variable attribute {attribute} is not allowed.")
        variable = cp.Variable(shape, **kwargs)
        self.variables.append(variable)
        return variable

    def add_constraint(self, constraint):
        self._verify_variables(constraint.variables())
        self.constraints.append(constraint)

    def add_cost(self, cost):
        self._verify_variables(cost.variables())
        self.cost += cost

    def _verify_variables(self, variables):
        raise NotImplementedError


class Vertex(ConvexProgram):

    def __init__(self, name=""):
        self.name = name
        super().__init__()

    def _verify_variables(self, variables):
        ids0 = {variable.id for variable in self.variables}
        ids1 = {variable.id for variable in variables}
        if not ids0 >= ids1:
            raise ValueError("A variable does not belong to this vertex.")


class Edge(ConvexProgram):

    def __init__(self, tail, head):
        self.tail = tail
        self.head = head
        super().__init__()

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

    def inc_edges(self, vertex):
        return [edge for edge in self.edges if edge.head == vertex]

    def out_edges(self, vertex):
        return [edge for edge in self.edges if edge.tail == vertex]

    def incident_edges(self, vertex):
        return self.incoming_edges(vertex) + self.outgoing_edges(vertex)

    def inc_vertices(self, vertex):
        return [edge.tail for edge in self.edges if edge.head == vertex]

    def out_vertices(self, vertex):
        return [edge.head for edge in self.edges if edge.tail == vertex]

    def neighbor_vertices(self, vertex):
        return self.inneighbours(vertex) + self.outneighbours(vertex)
    
    def num_vertices(self):
        return len(self.vertices)

    def num_edges(self):
        return len(self.edges)

    def graphviz(self, vertex_labels=None, edge_labels=None):
        if vertex_labels is None:
            vertex_labels = [vertex.name for vertex in self.vertices]
        if edge_labels is None:
            edge_labels = [''] * self.num_edges()
        digraph = gv.Digraph()
        for label in vertex_labels:
            digraph.node(str(label))
        for edge, label in zip(self.edges, edge_labels):
            tail = vertex_labels[self.vertices.index(edge.tail)]
            head = vertex_labels[self.vertices.index(edge.head)]
            digraph.edge(str(tail), str(head), str(label))
        return digraph
