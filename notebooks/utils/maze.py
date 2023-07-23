import matplotlib.pyplot as plt
import random as rd


class Cell:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.walls = {"N": True, "S": True, "E": True, "W": True}

    def has_all_walls(self):
        return all(self.walls.values())

    def knock_down_wall(self, wall):
        self.walls[wall] = False


class Maze:
    directions = {"W": (-1, 0), "E": (1, 0), "S": (0, -1), "N": (0, 1)}

    def __init__(self, nx, ny):
        self.nx = nx
        self.ny = ny
        self.cells = [[Cell(x, y) for y in range(ny)] for x in range(nx)]

    def get_cell(self, x, y):
        return self.cells[x][y]
    
    def plot(self):
        plt.gca().axis('off')
        plt.plot([0, self.nx - 1], [self.ny, self.ny], c='k')
        plt.plot([self.nx, self.nx], [0, self.ny], c='k')
        for x in range(self.nx):
            for y in range(self.ny):
                if self.get_cell(x, y).walls['S'] and (x != 0 or y != 0):
                    plt.plot([x, x + 1], [y, y], c='k')
                if self.get_cell(x, y).walls['W']:
                    plt.plot([x, x], [y, y + 1], c='k')
        
    def unexplored_neighbors(self, cell):
        neighbours = []
        for direction, (dx, dy) in self.directions.items():
            x2 = cell.x + dx
            y2 = cell.y + dy
            if (0 <= x2 < self.nx) and (0 <= y2 < self.ny):
                neighbour = self.get_cell(x2, y2)
                if neighbour.has_all_walls():
                    neighbours.append((direction, neighbour))
        return neighbours

    def make_maze(self, seed=0):
        rd.seed(seed)
        n = self.nx * self.ny
        cell_stack = [self.get_cell(0, 0)]
        while len(cell_stack) > 0:
            neighbours = self.unexplored_neighbors(cell_stack[-1])
            if not neighbours:
                cell_stack.pop()
            else:
                direction, next_cell = rd.choice(neighbours)
                self.knock_down_wall(cell_stack[-1], direction)
                cell_stack.append(next_cell)
            
    def knock_down_wall(self, cell, wall):
        cell.knock_down_wall(wall)
        dx, dy = self.directions[wall]
        neighbor = self.get_cell(cell.x + dx, cell.y + dy)
        neighbor_wall = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}[wall]
        neighbor.knock_down_wall(neighbor_wall)

    def knock_down_walls(self, n, seed=0):
        rd.seed(seed)
        knock_downs = 0
        while knock_downs < n:
            x = rd.randint(1, self.nx - 2)
            y = rd.randint(1, self.ny - 2)
            cell = self.get_cell(x, y)
            walls = [wall for wall, has_wall in cell.walls.items() if has_wall]
            if len(walls) > 0:
                wall = rd.choice(walls)
                self.knock_down_wall(cell, wall)
                knock_downs += 1
