import numpy as np
import cvxpy as cp
from collections.abc import Iterable
from gcspy.vertices import ConicVertex, ConvexVertex
from gcspy.edges import ConicEdge, ConvexEdge
from gcspy.graph_problems.facility_location import ConicFacilityLocationProblem
from gcspy.graph_problems.shortest_path import ConicShortestPathProblem
from gcspy.graph_problems.spanning_tree import ConicSpanningTreeProblem
from gcspy.graph_problems.traveling_salesman import ConicTravelingSalesmanProblem
from gcspy.graph_problems.from_ilp import ConicGraphProblemFromILP

# TODO: add support for undirected graphs.

class Graph:
    """
    Base class that contains the method that are common to GraphOfConicPrograms
    and GraphOfConvexPrograms.
    """

    def __init__(self):
        self.vertices = []
        self.edges = []

    def has_vertex(self, name):
        return name in [vertex.name for vertex in self.vertices]
    
    def has_edge(self, name):
        return name in [edge.name for edge in self.edges]
        
    def add_vertex(self, name):
        if self.has_vertex(name):
            raise ValueError(f"Vertex with name {name} is aleady defined.")
        return self._add_vertex(name)

    def add_edge(self, tail, head):
        if not self.has_vertex(tail.name):
            raise ValueError(f"Vertex with name {tail.name} is not defined.")
        if not self.has_vertex(head.name):
            raise ValueError(f"Vertex with name {head.name} is not defined.")
        name = (tail.name, head.name)
        if self.has_edge(name):
            raise ValueError(f"Edge with name {name} is aleady defined.")
        return self._add_edge(tail, head)

    def _add_vertex(self, name):
        """
        This method must be overwritte by the derived class.
        """
        raise NotImplementedError

    def _add_edge(self, tail, head):
        """
        This method must be overwritte by the derived class.
        """
        raise NotImplementedError

    def get_vertex(self, name):
        for vertex in self.vertices:
            if vertex.name == name:
                return vertex
        raise ValueError(f"There is no vertex with name {name}.")

    def get_edge(self, tail_name, head_name):
        name = (tail_name, head_name)
        for edge in self.edges:
            if edge.name == name:
                return edge
        raise ValueError(f"There is no edge with name {name}.")

    def vertex_index(self, vertex):
        return self.vertices.index(vertex)

    def edge_index(self, edge):
        return self.edges.index(edge)

    def incoming_edges(self, v):
        if isinstance(v, Iterable):
            return [e for e in self.edges if e.head in v and e.tail not in v]
        else:
            return [e for e in self.edges if e.head == v]
        
    def outgoing_edges(self, v):
        if isinstance(v, Iterable):
            return [e for e in self.edges if e.tail in v and e.head not in v]
        else:
            return [e for e in self.edges if e.tail == v]
        
    def incident_edges(self, v):
        return self.incoming_edges(v) + self.outgoing_edges(v)

    def incoming_edge_indices(self, v):
        if isinstance(v, Iterable):
            return [k for k, e in enumerate(self.edges) if e.head in v and e.tail not in v]
        else:
            return [k for k, e in enumerate(self.edges) if e.head == v]

    def outgoing_edge_indices(self, v):
        if isinstance(v, Iterable):
            return [k for k, e in enumerate(self.edges) if e.tail in v and e.head not in v]
        else:
            return [k for k, e in enumerate(self.edges) if e.tail == v]
        
    def incident_edge_indices(self, v):
        return self.incoming_edge_indices(v) + self.outgoing_edge_indices(v)

    def num_vertices(self):
        return len(self.vertices)

    def num_edges(self):
        return len(self.edges)
    
    def vertex_binaries(self):
        return cp.hstack([vertex.y for vertex in self.vertices])

    def edge_binaries(self):
        return cp.hstack([edge.y for edge in self.edges])
    
    def add_disjoint_subgraph(self, graph):
        if type(graph) != type(self):
            raise ValueError(
                f"Type mismatch: type(graph) = {type(graph)}, type(self) = {type(self)}. "
                "The two graphs must be of the same type."
                )
        self.vertices += graph.vertices
        self.edges += graph.edges
    
    def graphviz(self):
        from gcspy.plot_utils import graphviz_graph
        return graphviz_graph(self)

class GraphOfConicPrograms(Graph):

    def __init__(self):
        self.vertices = []
        self.edges = []
        
    def _add_vertex(self, name):
        vertex = ConicVertex(name)
        self.vertices.append(vertex)
        return vertex

    def _add_edge(self, tail, head):
        edge = ConicEdge(tail, head)
        self.edges.append(edge)
        return edge

class GraphOfConvexPrograms(Graph):

    def __init__(self):
        super().__init__()
        
    def _add_vertex(self, name):
        vertex = ConvexVertex(name)
        self.vertices.append(vertex)
        return vertex

    def _add_edge(self, tail, head):
        edge = ConvexEdge(tail, head)
        self.edges.append(edge)
        return edge
    
    def vertex_binaries(self):
        return np.array([vertex.binary_variable for vertex in self.vertices])

    def edge_binaries(self):
        return np.array([edge.binary_variable for edge in self.edges])

    def to_conic(self):
        conic_graph = GraphOfConicPrograms()
        for vertex in self.vertices:
            conic_vertex = vertex.to_conic()
            conic_graph.vertices.append(conic_vertex)
        for edge in self.edges:
            conic_tail = conic_graph.get_vertex(edge.tail.name)
            conic_head = conic_graph.get_vertex(edge.head.name)
            conic_edge = edge.to_conic(conic_tail, conic_head)
            conic_graph.edges.append(conic_edge)
        return conic_graph
    
    def set_variable_values_from_conic_graph(self, conic_graph, xv, yv, xe, ye):

        # set value of vertex variables
        for convex_vertex, x, y in zip(self.vertices, xv, yv):
            convex_vertex.binary_variable.value = y
            conic_vertex = conic_graph.get_vertex(convex_vertex.name)
            for convex_variable in convex_vertex.variables:
                if x is None:
                    convex_variable.value = None
                else:
                    convex_variable.value = conic_vertex.get_convex_variable_value(convex_variable, x)

        # set value of edge variables
        for convex_edge, x, y in zip(self.edges, xe, ye):
            convex_edge.binary_variable.value = y
            conic_edge = conic_graph.get_edge(*convex_edge.name)
            for convex_variable in convex_edge.variables:
                if x is None:
                    convex_variable.value = None
                else:
                    convex_variable.value = conic_edge.get_convex_variable_value(convex_variable, x)
    
    def solve_shortest_path(self, source, target, binary=True, *args, **kwargs):
        conic_graph = self.to_conic()
        conic_problem = ConicShortestPathProblem(conic_graph, source.name, target.name, binary)
        prob, xv, yv, xe, ye = conic_problem.solve(*args, **kwargs)
        self.set_variable_values_from_conic_graph(conic_graph, xv, yv, xe, ye)
        return prob
    
    def solve_traveling_salesman(self, subtour_elimination=True, binary=True, *args, **kwargs):
        conic_graph = self.to_conic()
        conic_problem = ConicTravelingSalesmanProblem(conic_graph, subtour_elimination, binary)
        prob, xv, yv, xe, ye = conic_problem.solve(*args, **kwargs)
        self.set_variable_values_from_conic_graph(conic_graph, xv, yv, xe, ye)
        return prob
    
    def solve_facility_location(self, binary=True, *args, **kwargs):
        conic_graph = self.to_conic()
        conic_problem = ConicFacilityLocationProblem(conic_graph, binary)
        prob, xv, yv, xe, ye = conic_problem.solve(*args, **kwargs)
        self.set_variable_values_from_conic_graph(conic_graph, xv, yv, xe, ye)
        return prob
    
    def solve_spanning_tree(self, root, subtour_elimination=True, binary=True, *args, **kwargs):
        conic_graph = self.to_conic()
        conic_problem = ConicSpanningTreeProblem(conic_graph, root.name, subtour_elimination, binary)
        prob, xv, yv, xe, ye = conic_problem.solve(*args, **kwargs)
        self.set_variable_values_from_conic_graph(conic_graph, xv, yv, xe, ye)
        return prob
    
    def solve_from_ilp(self, ilp_constraints, binary=True, *args, **kwargs):
        convex_yv = self.vertex_binaries()
        convex_ye = self.edge_binaries()
        conic_graph = self.to_conic()
        conic_problem = ConicGraphProblemFromILP(conic_graph, convex_yv, convex_ye, ilp_constraints, binary)
        prob, xv, yv, xe, ye = conic_problem.solve(*args, **kwargs)
        self.set_variable_values_from_conic_graph(conic_graph, xv, yv, xe, ye)
        return prob

    def plot_2d(self, **kwargs):
        from gcspy.plot_utils import plot_2d_graph
        return plot_2d_graph(self, **kwargs)

    def plot_2d_subgraph(self):
        from gcspy.plot_utils import plot_2d_subgraph
        return plot_2d_subgraph(self)
