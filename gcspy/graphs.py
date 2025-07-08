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

    def incoming_edge_indices(self, vertex):
        if isinstance(vertex, Iterable):
            return [k for k, edge in enumerate(self.edges) if edge.head in vertex and edge.tail not in vertex]
        else:
            return [k for k, edge in enumerate(self.edges) if edge.head == vertex]

    def outgoing_edge_indices(self, vertex):
        if isinstance(vertex, Iterable):
            return [k for k, edge in enumerate(self.edges) if edge.tail in vertex and edge.head not in vertex]
        else:
            return [k for k, edge in enumerate(self.edges) if edge.tail == vertex]
        
    def incident_edge_indices(self, vertex):
        return self.incoming_edge_indices(vertex) + self.outgoing_edge_indices(vertex)

    def incoming_edges(self, vertex):
        return [self.edges[k] for k in self.incoming_edge_indices(vertex)]
        
    def outgoing_edges(self, vertex):
        return [self.edges[k] for k in self.outgoing_edge_indices(vertex)]
        
    def incident_edges(self, vertex):
        return [self.edges[k] for k in self.incident_edge_indices(vertex)]

    def num_vertices(self):
        return len(self.vertices)

    def num_edges(self):
        return len(self.edges)

    def add_disjoint_subgraph(self, graph):
        if type(graph) != type(self):
            raise ValueError(
                f"Type mismatch: type(graph) = {type(graph)}, type(self) = {type(self)}. "
                "The two graphs must be of the same type.")
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

        # initialize empty conic graph
        conic_graph = GraphOfConicPrograms()

        # add one vertex at the time
        for vertex in self.vertices:
            conic_vertex = vertex.to_conic()
            conic_graph.vertices.append(conic_vertex)

        # add one edge at the time
        for edge in self.edges:
            conic_tail = conic_graph.get_vertex(edge.tail.name)
            conic_head = conic_graph.get_vertex(edge.head.name)
            conic_edge = edge.to_conic(conic_tail, conic_head)
            conic_graph.edges.append(conic_edge)

        return conic_graph

    def _solve_graph_problem(self, conic_graph, conic_problem, *args, **kwargs):

        # solve problem
        prob, xv, yv, xe, ye = conic_problem.solve(*args, **kwargs)

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

        return prob
    
    def solve_shortest_path(self, source, target, binary=True, *args, **kwargs):
        conic_graph = self.to_conic()
        conic_problem = ConicShortestPathProblem(conic_graph, source.name, target.name, binary)
        return self._solve_graph_problem(conic_graph, conic_problem, *args, **kwargs)
    
    def solve_traveling_salesman(self, subtour_elimination=True, binary=True, *args, **kwargs):
        conic_graph = self.to_conic()
        conic_problem = ConicTravelingSalesmanProblem(conic_graph, subtour_elimination, binary)
        return self._solve_graph_problem(conic_graph, conic_problem, *args, **kwargs)
    
    def solve_facility_location(self, binary=True, *args, **kwargs):
        conic_graph = self.to_conic()
        conic_problem = ConicFacilityLocationProblem(conic_graph, binary)
        return self._solve_graph_problem(conic_graph, conic_problem, *args, **kwargs)
    
    def solve_spanning_tree(self, root, subtour_elimination=True, binary=True, *args, **kwargs):
        conic_graph = self.to_conic()
        conic_problem = ConicSpanningTreeProblem(conic_graph, root.name, subtour_elimination, binary)
        return self._solve_graph_problem(conic_graph, conic_problem, *args, **kwargs)
    
    def solve_from_ilp(self, ilp_constraints, binary=True, *args, **kwargs):
        convex_yv = self.vertex_binaries()
        convex_ye = self.edge_binaries()
        conic_graph = self.to_conic()
        conic_problem = ConicGraphProblemFromILP(conic_graph, convex_yv, convex_ye, ilp_constraints, binary)
        return self._solve_graph_problem(conic_graph, conic_problem, *args, **kwargs)

    def plot_2d(self, **kwargs):
        from gcspy.plot_utils import plot_2d_graph
        return plot_2d_graph(self, **kwargs)

    def plot_2d_solution(self):
        from gcspy.plot_utils import plot_2d_solution
        return plot_2d_solution(self)
