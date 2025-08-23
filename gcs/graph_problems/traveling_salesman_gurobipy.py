try:
    import gurobipy as gp
    from gurobipy import GRB
    from gcsopt.graph_problems.utils_gurobipy import (create_environment, define_variables,
        enforce_edge_programs, constraint_homogenization,
        get_solution, SubtourEliminationCallback, subtour_elimination_constraints)
    has_gurobi = True
except ModuleNotFoundError:
    has_gurobi = False

def traveling_salesman_gurobipy(conic_graph, lazy_constraints, binary, tol, gurobi_parameters=None):
    if not has_gurobi:
        raise ImportError("Gurobi is not installed. Install gurobipy to use this method.")

    # Inialize model.
    env = create_environment(gurobi_parameters)
    model = gp.Model(env=env)

    # Define variables.
    zv, ye, ze, ze_tail, ze_head = define_variables(model, conic_graph, binary)

    # Edge costs and constraints.
    cost = enforce_edge_programs(model, conic_graph, ye, ze, ze_tail, ze_head)

    # Vertex costs and constraints.
    for i, vertex in enumerate(conic_graph.vertices):
        cost += vertex.cost_homogenization(zv[i], 1)

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
