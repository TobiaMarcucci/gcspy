import numpy as np
import cvxpy as cp
import networkx as nx
import gurobipy as gp
from gurobipy import GRB
from itertools import combinations

def create_environment(gurobi_parameters=None):
    env = gp.Env()
    gurobi_parameters = dict(gurobi_parameters or {})
    gurobi_parameters.setdefault("OutputFlag", 0)
    for key, value in gurobi_parameters.items():
        env.setParam(key, value)
    return env

def constrain_in_cone(model, z, K):

    # Linear constraints.
    if K == cp.Zero:
        model.addConstr(z == 0)
    elif K == cp.NonNeg:
        model.addConstr(z >= 0)
    elif K == cp.NonPos:
        model.addConstr(z <= 0)

    # Second order cone constraint.
    elif K == cp.SOC:
        s = model.addVar() # Nonnegative slack variable.
        model.addConstr(z[0] == s)
        model.addConstr(z[1:] @ z[1:] <= s ** 2) # Convex for gurobi.

    # There are no other constraints we can support for MICPs.
    else:
        raise NotImplementedError

def constraint_homogenization(model, prog, x, y):
    z = prog.A @ x + prog.b * y
    start = 0
    for cone_type, cone_size in prog.K:
        stop = start + cone_size
        constrain_in_cone(model, z[start:stop], cone_type)
        start = stop

def edge_cost_homogenization(edge, xv, xw, xe, y):
    edge._check_vector_sizes(xv, xw, xe)
    x = gp.concatenate((xv, xw, xe))
    return edge.c @ x + edge.d * y
    
def edge_constraint_homogenization(model, edge, xv, xw, xe, y):
    edge._check_vector_sizes(xv, xw, xe)
    x = gp.concatenate((xv, xw, xe))
    constraint_homogenization(model, edge, x, y)

def define_variables(model, conic_graph, binary):

    # Binary variables.
    vtype = GRB.BINARY if binary else GRB.CONTINUOUS
    ye = model.addMVar(conic_graph.num_edges(), vtype=vtype)

    # Function that allows adding variables of zero size.
    add_var = lambda size: model.addMVar(size, lb=-np.inf) if size > 0 else np.array([])
    
    # Auxiliary continuous varibales.
    zv = np.array([add_var(vertex.size) for vertex in conic_graph.vertices])
    ze = np.array([add_var(edge.slack_size) for edge in conic_graph.edges])
    ze_tail = np.array([add_var(edge.tail.size) for edge in conic_graph.edges])
    ze_head = np.array([add_var(edge.head.size) for edge in conic_graph.edges])

    return zv, ye, ze, ze_tail, ze_head

def enforce_edge_programs(model, conic_graph, ye, ze, ze_tail, ze_head):

    # Edge costs and constraints.
    cost = 0
    for k, edge in enumerate(conic_graph.edges):
        cost += edge_cost_homogenization(edge, ze_tail[k], ze_head[k], ze[k], ye[k])
        edge_constraint_homogenization(model, edge, ze_tail[k], ze_head[k], ze[k], ye[k])
        constraint_homogenization(model, edge.tail, ze_tail[k], ye[k])
        constraint_homogenization(model, edge.head, ze_head[k], ye[k])

    return cost

def get_solution(conic_graph, zv, ye, ze, tol):

    # Set vertex variable values.
    for vertex, z in zip(conic_graph.vertices, zv):
        vertex.binary_variable.value = 1
        vertex.x.value = z.X

    # Set edge variable values.
    for edge, y, z in zip(conic_graph.edges, ye, ze):
        edge.binary_variable.value = y.X
        z_value = z.X if z.size > 0 else np.array([])
        if y.X is not None and y.X > tol:
            edge.x.value = np.concatenate((
                edge.tail.x.value,
                edge.head.x.value,
                z_value / y.X))
            
class Callback:
    """
    Usable for the TSP and the MSTP.
    """

    def __init__(self, conic_graph, ye):
        self.conic_graph = conic_graph
        self.ye = ye

    def __call__(self, model, where):
        if where == GRB.Callback.MIPSOL:
            ye = model.cbGetSolution(self.ye)
            edges = [self.conic_graph.edges[k] for k, y in enumerate(ye) if y > 0.5]
            tour = self.shortest_subtour(edges)
            if tour and len(tour) < self.conic_graph.num_vertices():
                self.cut(model, tour)

    def shortest_subtour(self, edges):
        if self.conic_graph.directed:
            G = nx.DiGraph()
        else:
            G = nx.Graph()
        G.add_edges_from([(e.tail, e.head) for e in edges])
        if self.conic_graph.directed:
            tours = list(nx.simple_cycles(G))
            tours.sort(key=len)
        else:
            tours = list(nx.simple_cycles(G, nx.girth(G)))
        if tours:
            return tours[0]
            
    # def shortest_subtour(self, edges):
    #     """
    #     The edges here are only the ones that have binary equal to one. It is
    #     assumed there is exactly one incoming edge and one outgoing edge for
    #     every vertex represented in the edge list.
    #     """

    #     # Create a mapping from each vertex to its neighbors. Do not use the
    #     # neighbors method provided by the graph since it would also add
    #     # neighbors connected by edges with binary equal to zero.
    #     vertex_neighbors = {}
    #     for edge in edges:
    #         vertex_neighbors.setdefault(edge.tail, []).append(edge.head)
    #         if not self.conic_graph.directed:
    #             vertex_neighbors.setdefault(edge.head, []).append(edge.tail)

    #     # Follow edges to find cycles. Each time a new cycle is found, keep track
    #     # of the shortest cycle found so far and restart from an unvisited vertex.
    #     unvisited = set(vertex_neighbors)
    #     shortest = None
    #     while unvisited:
    #         cycle = []
    #         neighbors = list(unvisited)
    #         while neighbors:
    #             current = neighbors.pop()
    #             cycle.append(current)
    #             unvisited.remove(current)
    #             neighbors = [vertex for vertex in vertex_neighbors[current] if vertex in unvisited]
    #         if shortest is None or len(cycle) < len(shortest):
    #             shortest = cycle

    #     return shortest

def subtour_elimination_constraints(model, conic_graph, ye):
    """
    Subtour elimination constraints for all subsets of vertices.
    """
    start = 2 if conic_graph.directed else 3
    for n_vertices in range(start, conic_graph.num_vertices() - 1):
        for vertices in combinations(conic_graph.vertices, n_vertices):
            ind = conic_graph.induced_edge_indices(vertices)
            model.addConstr(sum(ye[ind]) <= n_vertices - 1)

class SubtourEliminationCallback(Callback):

    def cut(self, model, tour):
        ind = self.conic_graph.induced_edge_indices(tour)
        model.cbLazy(sum(self.ye[ind]) <= len(tour) - 1)

class CutsetCallback(Callback):

    def cut(self, model, tour):
        inc = self.conic_graph.incoming_edge_indices(tour)
        model.cbLazy(sum(self.ye[inc]) >= 1)