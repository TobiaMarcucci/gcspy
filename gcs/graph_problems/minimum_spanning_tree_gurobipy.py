from itertools import combinations
try:
    import gurobipy as gp
    from gurobipy import GRB
    from gcsopt.graph_problems.utils_gurobipy import (create_environment, define_variables,
        enforce_edge_programs, constraint_homogenization, get_solution,
        SubtourEliminationCallback, CutsetCallback, subtour_elimination_constraints)
    has_gurobi = True
except ModuleNotFoundError:
    has_gurobi = False

def undirected_minimum_spanning_tree_gurobipy(conic_graph, lazy_constraints, binary, tol, gurobi_parameters=None):

    # Check that gurobipy is installed and graph is undirected.
    if not has_gurobi:
        raise ImportError("Gurobi is not installed. Install gurobipy to use this function.")
    if conic_graph.directed:
        raise ValueError("Function applicable only to undirected graphs.")

    # Inialize model.
    env = create_environment(gurobi_parameters)
    model = gp.Model(env=env)

    # Define variables.
    zv, ye, ze, ze_tail, ze_head = define_variables(model, conic_graph, binary)

    # Edge costs and constraints.
    cost = enforce_edge_programs(model, conic_graph, ye, ze, ze_tail, ze_head)

    # Number of edges in the tree.
    model.addConstr(sum(ye) == conic_graph.num_vertices() - 1)

    # Vertex costs.
    for i, vertex in enumerate(conic_graph.vertices):
        cost += vertex.cost_homogenization(zv[i], 1)

        # Cutset constraints for one vertex only.
        incident = conic_graph.incident_edge_indices(vertex)
        inc = [k for k in incident if conic_graph.edges[k].head == vertex]
        out = [k for k in incident if conic_graph.edges[k].tail == vertex]
        constraint_homogenization(model, vertex,
            sum(ze_head[inc]) + sum(ze_tail[out]) - zv[i],
            sum(ye[incident]) - 1)
        
        # Constraints implied by ye <= 1.
        for k in inc:
            constraint_homogenization(model, vertex, zv[i] - ze_head[k], 1 - ye[k])
        for k in out:
            constraint_homogenization(model, vertex, zv[i] - ze_tail[k], 1 - ye[k])

    # Set objective.
    model.setObjective(cost, GRB.MINIMIZE)

    # Solve with lazy constraints.
    if lazy_constraints:
        model.Params.LazyConstraints = 1
        callback = SubtourEliminationCallback(conic_graph, ye)
        model.optimize(callback)

    # Exponentially many subtour elimination constraints.
    else:
        subtour_elimination_constraints(model, conic_graph, ye)
        model.optimize()

    # Set value of vertex binaries.
    if model.status == 2:
        get_solution(conic_graph, zv, ye, ze, tol)

    return model

def directed_minimum_spanning_tree_gurobipy(conic_graph, conic_root, lazy_constraints, binary, tol, gurobi_parameters=None):

    # Check that gurobipy is installed and graph is directed.
    if not has_gurobi:
        raise ImportError("Gurobi is not installed. Install gurobipy to use this function.")
    if not conic_graph.directed:
        raise ValueError("Function applicable only to directed graphs.")

    # Inialize model.
    env = create_environment(gurobi_parameters)
    model = gp.Model(env=env)

    # Define variables.
    zv, ye, ze, ze_tail, ze_head = define_variables(model, conic_graph, binary)

    # Edge costs and constraints.
    cost = enforce_edge_programs(model, conic_graph, ye, ze, ze_tail, ze_head)

    # # Number of edges in the tree.
    # model.addConstr(sum(ye) == conic_graph.num_vertices() - 1)

    # Vertex costs.
    for i, vertex in enumerate(conic_graph.vertices):
        cost += vertex.cost_homogenization(zv[i], 1)

        # Constraints on incoming edges.
        inc = conic_graph.incoming_edge_indices(vertex)
        if vertex == conic_root:
            constraint_homogenization(model, vertex, zv[i], 1)
            model.addConstrs((ye[k] == 0 for k in inc))
            model.addConstrs((ze_head[k] == 0 for k in inc))
        else:
            model.addConstr(sum(ye[inc]) == 1)
            model.addConstr(sum(ze_head[inc]) == zv[i])

        # Constraints on outgoing edges.
        for k in conic_graph.outgoing_edge_indices(vertex):
            constraint_homogenization(model, vertex, zv[i] - ze_tail[k], 1 - ye[k])

    # Set objective.
    model.setObjective(cost, GRB.MINIMIZE)

    # Solve with lazy constraints.
    # if lazy_constraints:
    if lazy_constraints:
        model.Params.LazyConstraints = 1
        callback = CutsetCallback(conic_graph, ye)
        model.optimize(callback)

    # Exponentially many cutset constraints.
    else:
        i = conic_graph.vertex_index(conic_root)
        subvertices = conic_graph.vertices[:i] + conic_graph.vertices[i+1:]
        for subtour_size in range(2, conic_graph.num_vertices()):
            for vertices in combinations(subvertices, subtour_size):
                inc = conic_graph.incoming_edge_indices(vertices)
                model.addConstr(sum(ye[inc]) >= 1)
        model.optimize()

    # Set value of vertex binaries.
    if model.status == 2:
        get_solution(conic_graph, zv, ye, ze, tol)

    return model
    
