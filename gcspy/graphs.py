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

    def incoming_edge_indices(self, vertices):
        """
        Return indices of edges that are incoming to `vertices` (i.e., edges
        whose head is in `vertices` but tail is not).
        """
        if not isinstance(vertices, Iterable):
            vertices = [vertices]
        def is_incoming(edge, vertices):
            return edge.tail not in vertices and edge.head in vertices
        return [k for k, edge in enumerate(self.edges) if is_incoming(edge, vertices)]

    def outgoing_edge_indices(self, vertices):
        """
        Return indices of edges that are outgoing from `vertices` (i.e., edges
        whose tail is in `vertices` but head is not).
        """
        if not isinstance(vertices, Iterable):
            vertices = [vertices]
        def is_outgoing(edge, vertices):
            return edge.tail in vertices and edge.head not in vertices
        return [k for k, edge in enumerate(self.edges) if is_outgoing(edge, vertices)]

    def incident_edge_indices(self, vertices):
        """
        Return indices of edges that are incident with `vertices` (either
        incoming or outgoing).
        """
        return self.incoming_edge_indices(vertices) + self.outgoing_edge_indices(vertices)

    def incoming_edges(self, vertices):
        """
        Return edges that are incoming to `vertices` (i.e., edges whose head is
        in `vertices` but tail is not).
        """
        return [self.edges[k] for k in self.incoming_edge_indices(vertices)]
        
    def outgoing_edges(self, vertices):
        """
        Return edges that are outgoing from `vertices` (i.e., edges whose tail
        is in `vertices` but head is not).
        """
        return [self.edges[k] for k in self.outgoing_edge_indices(vertices)]
        
    def incident_edges(self, vertices):
        """
        Return edges that are incident with `vertices` (either incoming or
        outgoing).
        """
        return [self.edges[k] for k in self.incident_edge_indices(vertices)]

    def num_vertices(self):
        """
        Return number of vertices in the graph.
        """
        return len(self.vertices)

    def num_edges(self):
        """
        Return number of edges in the graph.
        """
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

        # solve problem in conic form
        prob, xv, yv, xe, ye = conic_problem.solve(*args, **kwargs)

        # set value of vertex variables for convex program
        for convex_vertex, x, y in zip(self.vertices, xv, yv):
            convex_vertex.binary_variable.value = y
            if x is None:
                for variable in convex_vertex.variables:
                    variable.value = None
            else:
                conic_vertex = conic_graph.get_vertex(convex_vertex.name)
                for variable in convex_vertex.variables:
                    variable.value = conic_vertex.get_convex_variable_value(variable, x)

        # set value of edge variables for convex program
        for convex_edge, x, y in zip(self.edges, xe, ye):
            convex_edge.binary_variable.value = y
            if x is None:
                for variable in convex_edge.variables:
                    variable.value = None
            else:
                conic_edge = conic_graph.get_edge(*convex_edge.name)
                x_tail = xv[self.vertex_index(convex_edge.tail)]
                x_head = xv[self.vertex_index(convex_edge.head)]
                x_extended = np.concatenate((x_tail, x_head, x))
                for variable in convex_edge.variables:
                    variable.value = conic_edge.get_convex_variable_value(variable, x_extended)

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

    def solve_convex_restriction(self, vertex_indices, edge_indices):
        """
        Solves convex program obtained by discarding all the vertices and edges
        that are not in the given lists.
        """

        # check the given vertices and edges for a valid subgraph
        for k in edge_indices:
            edge = self.edges[k]
            i = self.vertex_index(edge.tail)
            j = self.vertex_index(edge.head)
            if i not in vertex_indices or j not in vertex_indices:
                raise ValueError('Given indices do not form a subgraph.')

        # assemble convex program
        cost = 0
        constraints = []
        for i in vertex_indices:
            vertex = self.vertices[i]
            cost += vertex.cost
            constraints.extend(vertex.constraints)
        for k in edge_indices:
            edge = self.edges[k]
            cost += edge.cost
            constraints.extend(edge.constraints)

        # solve convex program
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve()

        # set vertex variable values
        for i, vertex in enumerate(self.vertices):
            if i in vertex_indices:
                vertex.binary_variable.value = 1
            else:
                vertex.binary_variable.value = None
                for variable in vertex.variables:
                    variable.value = None

        # set edgevariable values
        for k, edge in enumerate(self.edges):
            if k in edge_indices:
                edge.binary_variable.value = 1
            else:
                edge.binary_variable.value = 0
                for variable in edge.variables:
                    variable.value = None

        return prob

    def plot_2d(self, **kwargs):
        from gcspy.plot_utils import plot_2d_graph
        return plot_2d_graph(self, **kwargs)

    def plot_2d_solution(self):
        from gcspy.plot_utils import plot_2d_solution
        return plot_2d_solution(self)
