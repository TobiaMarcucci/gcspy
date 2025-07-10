import numpy as np
import matplotlib.pyplot as plt
from maze import maze, graph, source, target

# solve problem
prob = graph.solve_shortest_path(source, target, binary=False)
print("Convex relaxation status:", prob.status)
print("Optimal value of convex relaxation:", prob.value)

# natural rounding strategy for shortest path problem is randomized depth first
# search with backtracking
def randomized_dfs(graph, source, target, edge_probabilities):

    # initialize path and set of visited vertices
    path = [source]
    visited = []

    # repeat until target is reached
    while path:
        if path[-1] == target:
            return path

        # collect neighbors and probabilities
        neighbors = []
        probabilities = []
        for k in graph.outgoing_edge_indices(path[-1]):
            neighbor = graph.edges[k].head
            probability = edge_probabilities[k]
            if neighbor not in path + visited and probability > 0:
                neighbors.append(neighbor)
                probabilities.append(probability)

        # explore a random neighbor
        if neighbors:
            probabilities = np.array(probabilities) / sum(probabilities)
            neighbor = np.random.choice(neighbors, p=probabilities)
            path.append(neighbor)
            
        # backtrack and prevent revisit of same vertex
        else:
            visited.append(path.pop())

    # path not found
    return None

# round solution of convex relaxation
edge_probabilities = graph.edge_binary_values()
path = randomized_dfs(graph, source, target, edge_probabilities)
path_edges = [graph.get_edge(tail.name, head.name) for tail, head in zip(path[:-1], path[1:])]

# solve convex restriction for the given path
# is this optimal value is equal to the one of the convex relaxation then we
# have a certificate of global optimality for our trajectory
# (to make the relaxation looser and the problem more challening consider
# replacing the vertex cost vertex.add_cost(cp.norm2(x[1] - x[0])) with
# vertex.add_cost(cp.sum_squares(x[1] - x[0])) in maze.py)
prob = graph.solve_convex_restriction(path, path_edges)
print("Convex restriction status:", prob.status)
print("Optimal value of convex restriction:", prob.value)

# plot optimal trajectory
plt.figure()
maze.plot()
for vertex in graph.vertices:
    if np.isclose(vertex.binary_variable.value, 1):
        plt.plot(*vertex.variables[0].value.T, 'b--')
plt.show()