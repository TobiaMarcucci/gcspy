import gurobipy as gp
from gurobipy import GRB
from itertools import combinations
from gcspy.graph_problems.utils_gurobipy import define_variables, enforce_edge_programs, cost_homogenization, constraint_homogenization, get_solution

def traveling_salesman_gurobipy(conic_graph, lazy_constraints, binary, tol, gurobi_parameters=None):

    # Create environment.
    env = gp.Env()
    gurobi_parameters = dict(gurobi_parameters or {})
    gurobi_parameters.setdefault("OutputFlag", 0)
    for key, value in gurobi_parameters.items():
        env.setParam(key, value)

    # Inialize model.
    model = gp.Model(env=env)

    # Define variables.
    zv, ye, ze, ze_tail, ze_head = define_variables(model, conic_graph, binary)

    # Edge costs and constraints.
    cost = enforce_edge_programs(model, conic_graph, ye, ze, ze_tail, ze_head)

    # Vertex costs and constraints.
    for i, vertex in enumerate(conic_graph.vertices):
        cost += cost_homogenization(vertex, zv[i], 1)

        # Directed graphs.
        if conic_graph.directed:
            inc = conic_graph.incoming_edge_indices(vertex)
            out = conic_graph.outgoing_edge_indices(vertex)
            model.addConstr(sum(ye[inc]) == 1)
            model.addConstr(sum(ye[out]) == 1)
            model.addConstr(sum(ze_head[inc]) == zv[i])
            model.addConstr(sum(ze_tail[out]) == zv[i])
            
        # Undirected graphs graphs.
        else:
            incident = conic_graph.incident_edge_indices(vertex)
            inc = [k for k in incident if conic_graph.edges[k].head == vertex]
            out = [k for k in incident if conic_graph.edges[k].tail == vertex]
            model.addConstr(sum(ye[incident]) == 2)
            model.addConstr(sum(ze_head[inc]) + sum(ze_tail[out]) == 2 * zv[i])
            for k in inc:
                constraint_homogenization(model, vertex, zv[i] - ze_head[k], 1 - ye[k])
            for k in out:
                constraint_homogenization(model, vertex, zv[i] - ze_tail[k], 1 - ye[k])

    # Set objective.
    model.setObjective(cost, GRB.MINIMIZE)

    # Solve with subtour elimination as constraint.
    if lazy_constraints:
        model.Params.LazyConstraints = 1
        callback = Callback(conic_graph, ye)
        model.optimize(callback)

    # Exponentially many subtour elimination constraints.
    else:
        start = 2 if conic_graph.directed else 3
        for n_vertices in range(start, conic_graph.num_vertices() - 1):
            for vertices in combinations(conic_graph.vertices, n_vertices):
                ind = conic_graph.induced_edge_indices(vertices)
                model.addConstr(sum(ye[ind]) <= n_vertices - 1)
        model.optimize()

    # Set value of vertex binaries.
    if model.status == 2:
        get_solution(conic_graph, zv, ye, ze, tol)

    return model

class Callback:
    """
    Adapted from https://docs.gurobi.com/projects/examples/en/current/examples/python/tsp.html.
    """

    def __init__(self, conic_graph, ye):
        self.conic_graph = conic_graph
        self.ye = ye

    def __call__(self, model, where):
        if where == GRB.Callback.MIPSOL:
            ye = model.cbGetSolution(self.ye)
            edges = [self.conic_graph.edges[k] for k, y in enumerate(ye) if y > 0.5]
            tour = self.shortest_subtour(edges)
            if len(tour) < self.conic_graph.num_vertices():
                self.cut(model, tour)
                
    def cut(self, model, tour):
        induced_edges = [k for k, edge in enumerate(self.conic_graph.edges) if edge.tail in tour and edge.head in tour]
        model.cbLazy(gp.quicksum(self.ye[k] for k in induced_edges) <= len(tour) - 1)

    def shortest_subtour(self, edges):
        """
        The edges here are only the ones that have binary equal to one.
        """

        # Create a mapping from each vertex to its neighbors. Do not use the
        # neighbors method provided by the graph since it would also add
        # neighbors connected by edges with binary equal to zero.
        vertex_neighbors = {}
        for edge in edges:
            vertex_neighbors.setdefault(edge.tail, []).append(edge.head)
            if not self.conic_graph.directed:
                vertex_neighbors.setdefault(edge.head, []).append(edge.tail)

        # Follow edges to find cycles. Each time a new cycle is found, keep track
        # of the shortest cycle found so far and restart from an unvisited vertex.
        unvisited = set(vertex_neighbors)
        shortest = None
        while unvisited:
            cycle = []
            neighbors = list(unvisited)
            while neighbors:
                current = neighbors.pop()
                cycle.append(current)
                unvisited.remove(current)
                neighbors = [vertex for vertex in vertex_neighbors[current] if vertex in unvisited]
            if shortest is None or len(cycle) < len(shortest):
                shortest = cycle

        return shortest
