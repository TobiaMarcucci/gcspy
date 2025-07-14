import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from maze_utils import Maze
from gcspy import GraphOfConvexSets

# create maze
maze_side = 12
knock_downs = 10
random_seed = 0
maze = Maze(maze_side, maze_side, random_seed)
maze.knock_down_walls(knock_downs)

# initialize empty graph
graph = GraphOfConvexSets()

# start and goal points
start = np.array([0.5, 0])
goal = np.array([maze_side - 0.5, maze_side])

# add vertices
for i in range(maze_side):
    for j in range(maze_side):
        vertex = graph.add_vertex((i, j))

        # trajectory start and end point within cell (i, j)
        x = vertex.add_variable((2, 2))

        # minimize distance traveled within cell (i, j)
        vertex.add_cost(cp.norm2(x[1] - x[0]))

        # constrain trajectory segment in cell (i, j)
        l = np.array([i, j])
        u = l + 1
        vertex.add_constraints([x[0] >= l, x[0] <= u])
        vertex.add_constraints([x[1] >= l, x[1] <= u])

        # fix start and goal points
        if all(l == 0):
            vertex.add_constraint(x[0] == start)
        elif all(u == maze_side):
            vertex.add_constraint(x[1] == goal)

# add edges between communicating cells
for i in range(maze_side):
    for j in range(maze_side):
        cell = maze.get_cell(i, j)
        tail = graph.get_vertex((i, j))
        for direction, d in maze.directions.items():
            if not cell.walls[direction]:
                head = graph.get_vertex((i + d[0], j + d[1]))
                edge = graph.add_edge(tail, head)

                # enforce trajectory continuity
                end_tail = tail.variables[0][1]
                start_head = head.variables[0][0]
                edge.add_constraint(end_tail == start_head) 

# select source and target vertices
source = graph.get_vertex((0, 0))
target = graph.get_vertex((maze_side - 1, maze_side - 1))

# run followin code only if this file is executed directly, and not when it is
# imported by other files
if __name__ == "__main__":

    # solve problem
    prob = graph.solve_shortest_path(source, target)
    print("Problem status:", prob.status)
    print("Optimal value:", prob.value)

    # plot optimal trajectory
    plt.figure()
    maze.plot()
    for vertex in graph.vertices:
        if np.isclose(vertex.binary_variable.value, 1):
            plt.plot(*vertex.variables[0].value.T, 'b--')
    plt.show()