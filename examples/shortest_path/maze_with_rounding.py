import numpy as np
import matplotlib.pyplot as plt
from maze import maze, graph, source, target
from gcspy.graph_problems.rounding.shortest_path import randomized_dfs

# solve problem with mixed integer programming
mip = graph.solve_shortest_path(source, target)
print("Mixed-integer problem status:", mip.status)
print("Mixed-integer problem value:", mip.value)

# Solve problem using convex relaxation plus rounding. A natural rounding
# strategy for shortest-path problem is randomized depth-first search with
# backtracking. The function randomized_dfs allows to run multiple depth-first
# searches and keep the best solution found.

# fix arguments in randomized depth first search
num_paths = 5 # number of distinct random paths to be evaluated
max_trials = 100 # maximum number of trials to find num_paths distinct paths
def rounding_fn(graph, source, target):
    return randomized_dfs(graph, source, target, num_paths, max_trials)

# solve problem with rounding: if the optimal value of the restriction is equal
# to the optimal value of the relaxation we have a certificate of global
# optimality for our trajectory
relaxation, restriction = graph.solve_shortest_path_with_rounding(source, target, rounding_fn)
print("Convex relaxation status:", relaxation.status)
print("Convex restriction status:", restriction.status)
print("Convex relaxation value:", relaxation.value)
print("Convex restriction value:", restriction.value)
# (to make the relaxation looser and the problem more challening consider
# replacing the vertex cost vertex.add_cost(cp.norm2(x[1] - x[0])) with
# vertex.add_cost(cp.sum_squares(x[1] - x[0])) in the file maze.py)

# plot optimal trajectory
plt.figure()
maze.plot()
for vertex in graph.vertices:
    if np.isclose(vertex.binary_variable.value, 1):
        plt.plot(*vertex.variables[0].value.T, 'b--')
plt.show()