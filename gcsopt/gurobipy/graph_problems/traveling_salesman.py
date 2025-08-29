import gurobipy as gp
from gurobipy import GRB
from gcsopt.gurobipy.graph_problems.utils import (create_environment,
    define_variables, enforce_edge_programs, constraint_homogenization,
    set_solution, SubtourEliminationCallback, subtour_elimination_constraints)

def traveling_salesman_conic(conic_graph, lazy_constraints, binary, tol, gurobi_parameters=None, save_bounds=False):

    # Inialize model.
    env = create_environment(gurobi_parameters)
    model = gp.Model(env=env)

    # Define variables.
    yv, zv, ye, ze, ze_tail, ze_head = define_variables(model, conic_graph, binary)

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
        callback = SubtourEliminationCallback(conic_graph, ye, save_bounds)
        model.optimize(callback)
        set_solution(model, conic_graph, yv, zv, ye, ze, tol, callback)

    # Exponentially many subtour elimination constraints.
    else:
        subtour_elimination_constraints(model, conic_graph, ye)
        model.optimize()
        set_solution(model, conic_graph, yv, zv, ye, ze, tol)

def traveling_salesman(convex_graph, lazy_constraints=True, binary=True, tol=1e-4, gurobi_parameters=None, save_bounds=False):
        conic_graph = convex_graph.to_conic()
        traveling_salesman_conic(conic_graph, lazy_constraints, binary, tol, gurobi_parameters, save_bounds)
        convex_graph._set_solution(conic_graph)