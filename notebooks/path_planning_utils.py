import numpy as np
import cvxpy as cp
import networkx as nx
import matplotlib.pyplot as plt

from copy import copy
from matplotlib.patches import Rectangle

class Box:

    def __init__(self, bot, top):

        self.bot = bot
        self.top = top
        self.center = (bot + top) / 2

    def plot(self, fc='lightcyan', ec='k', **kwargs):

        diag = self.top - self.bot
        rect = Rectangle(self.bot, *diag, fc=fc, ec=ec, **kwargs)
        plt.gca().add_patch(rect)

def generate_boxes(n, sides, seed=0):

    np.random.seed(seed)
    boxes = {}
    for i in range(n):
        for j in range(n):
            center = np.array([i, j])
            shuffled_sides = copy(sides)
            np.random.shuffle(shuffled_sides)
            halfdiag = np.diag(shuffled_sides).dot(np.random.rand(2))
            boxes[i,j] = Box(center - halfdiag, center + halfdiag)

    return boxes

def intersect(boxes, sides):

    inters = {}
    reach = int(2 * max(sides))
    n = int(len(boxes) ** .5)
    min_reach = lambda i: max(i - reach, 0)
    max_reach = lambda i: min(i + reach + 1, n)

    for i in range(n):
        i_range = range(i, max_reach(i))
        for j in range(n):
            j_range = range(min_reach(j), max_reach(j))
            for ii in i_range:
                for jj in j_range:
                    if ii > i or jj > j:
                        bot = np.maximum(boxes[i,j].bot, boxes[ii,jj].bot)
                        top = np.minimum(boxes[i,j].top, boxes[ii,jj].top)
                        if all(bot <= top):
                            inters[(i,j),(ii,jj)] = Box(bot, top)

    return inters

def line_graph(boxes, inters, s, t):

    G_inter = nx.Graph()
    G_inter.add_nodes_from(boxes.keys())
    G_inter.add_edges_from(inters.keys())

    G_line = nx.line_graph(G_inter)
    for e in G_line.edges:
        cu = inters[e[0]].center
        cv = inters[e[1]].center
        G_line.edges[e]['weight'] = np.linalg.norm(cv - cu)

    G = G_line.to_directed()
    G.add_node(s)
    G.add_node(t)
    n = int(len(boxes) ** .5)
    for v in G.nodes:
        if v not in [s, t]:
            if v[0] == (0, 0):
                G.add_edge(s, v)
            if v[1] == (n -1, n - 1):
                G.add_edge(v, t)

    return G

def optimize_path(path, inters):

    variables = {}
    constraints = []
    cost = 0
    x_prev = None
    for v in path[1:-1]:
        variables[v] = cp.Variable(2)
        constraints.append(variables[v] >= inters[v].bot)
        constraints.append(variables[v] <= inters[v].top)
        if x_prev is not None:
            cost += cp.norm(variables[v] - x_prev, 2)
        x_prev = variables[v]

    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve()

    return np.array([x.value for x in variables.values()])

def plot(boxes, label=False, gap=.2, **kwargs):

    bot = np.full(2, np.inf)
    top = np.full(2, - np.inf)
    for k, box in boxes.items():
        box.plot(**kwargs)
        bot = np.minimum(bot, box.bot)
        top = np.maximum(top, box.top)
        if label:
            plt.text(*box.center, str(k), ha='center', va='center')
    plt.xlim([bot[0] - gap, top[0] + gap])
    plt.ylim([bot[1] - gap, top[1] + gap])
