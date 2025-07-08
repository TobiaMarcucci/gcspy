import numpy as np
import cvxpy as cp
from maze_utils import Maze
from gcspy import GraphOfConvexPrograms

# create maze
maze_side = 10
knock_downs = 3
maze = Maze(maze_side, maze_side)
maze.knock_down_walls(knock_downs)

# initialize empty graph
graph = GraphOfConvexPrograms()

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
        vertex.add_constraints([x[0] >= [i, j], x[0] <= [i + 1, j + 1]])
        vertex.add_constraints([x[1] >= [i, j], x[1] <= [i + 1, j + 1]])

        # fix start and goal points
        if i == 0 and j == 0:
            vertex.add_constraint(x[0] == start)
        elif i == maze_side - 1 and j == maze_side - 1:
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

# solve shortest path problem
s = graph.get_vertex((0, 0))
t = graph.get_vertex((maze_side - 1, maze_side - 1))
prob = graph.solve_shortest_path(s, t)
print("Problem status:", prob.status)
print("Optimal value:", prob.value)

# plot optimal trajectory
import matplotlib.pyplot as plt
plt.figure()
maze.plot()
for vertex in graph.vertices:
    if vertex.binary_variable.value and vertex.binary_variable.value > 0.5:
        plt.plot(*vertex.variables[0].value.T, c='b')
plt.show()