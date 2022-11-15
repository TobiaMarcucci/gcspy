import numpy as np
import cvxpy as cp
import networkx as nx
from graphviz import Digraph

import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Vertex:

    def __init__(self, name=None):
        
        self.name = name
        self.variables = []
        self.constraints = []

    def add_variable(self, size):

        x = cp.Variable(size)
        self.variables.append(x)

        return x
    
    def add_constraint(self, constraint):

        self.constraints.append(constraint)

    def get_feasible_point(self):

        prob = cp.Problem(cp.Minimize(0), self.constraints)
        prob.solve()

        return [x.value for x in self.variables]

    def discretize(self, n=30):

        if len(self.variables) != 1 or self.variables[0].size != 2:
            raise ValueError('Plotting is supported only for 2D sets.')

        x = self.variables[0]
        c = cp.Parameter(2)
        prob = cp.Problem(cp.Maximize(c @ x), self.constraints)

        vertices = np.zeros((n, 2))
        for i, angle in enumerate(np.linspace(0, 2 * np.pi, n)):
            c.value = np.array([np.cos(angle), np.sin(angle)])            
            prob.solve(warm_start=True)
            vertices[i] = x.value

        return vertices

    def plot(self, point=None, n=30, **kwargs):

        options = {'fc': 'lightcyan', 'ec': 'black'}
        options.update(kwargs)
        vertices = self.discretize(n)
        plt.fill(*vertices.T, **options, zorder=0)
        if point is not None:
            plt.scatter(*point[0], fc='w', ec='k', zorder=3)

class Edge:

    def __init__(self, u, v):
        
        self.u = u
        self.v = v
        self.name = (self.u.name, self.v.name)
        self.length = 0
        self.constraints = []

    def add_length(self, length):

        self.length += length
    
    def add_constraint(self, constraint):

        self.constraints.append(constraint)

    def plot(self, tail=None, head=None, **kwargs):

        for variables in [self.u.variables, self.v.variables]:
            if len(variables) != 1 or variables[0].size != 2:
                raise ValueError('Plotting is supported only for 2D sets.')

        options = {
            'color': 'k',
            'zorder': 2,
            'arrowstyle': '->, head_width=3, head_length=8',
            'connectionstyle': "arc3,rad=.3"
            }
        options.update(kwargs)

        if tail is None:
            tail = self.u.get_feasible_point()
        if head is None:
            head = self.v.get_feasible_point()
        arrow = patches.FancyArrowPatch(tail[0], head[0], **options)
        plt.gca().add_patch(arrow)


class GraphOfConvexSets:

    def __init__(self):

        self.vertices = []
        self.edges = []

    def add_vertex(self, name=None):

        v = Vertex(name)
        self.vertices.append(v)

        return v

    def get_vertex(self, name):

        for v in self.vertices:
            if v.name == name:

                return v

    def add_edge(self, u, v):

        e = Edge(u, v)
        self.edges.append(e)

        return e

    def get_edge(self, name_u, name_v):

        for e in self.edges:
            if e.u.name == name_u and e.v.name == name_v:
                
                return e

    def in_edges(self, v):

        return [e for e in self.edges if e.v == v]

    def out_edges(self, u):

        return [e for e in self.edges if e.u == u]

    def has_cycles(self, s):

        def depth_first(v, visited, cyclic):
            if not cyclic:
                visited.add(v)
                out_edges = self.out_edges(v)
                for e in out_edges:
                    if e.v in visited:
                        cyclic = True
                        return
                for e in out_edges:
                    depth_first(e.v, visited, cyclic)

        visited = set()
        cyclic = False
        depth_first(s, visited, cyclic)

        return cyclic

        # graph = nx.DiGraph()
        # graph.add_nodes_from(self.vertices)
        # graph.add_edges_from([(e.u, e.v) for e in self.edges])

        # return not nx.is_directed_acyclic_graph(graph)

    def graphviz(self, vertex_labels=None, edge_labels=None):

        if vertex_labels is None:
            vertex_labels = [str(v.name) for v in self.vertices]
        if edge_labels is None:
            edge_labels = [''] * len(self.edges)

        G = Digraph()
        for v, l in zip(self.vertices, vertex_labels):
            G.node(str(v), l)
        for e, l in zip(self.edges, edge_labels):
            G.edge(str(e.u), str(e.v), l)

        return G
