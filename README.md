# GCSOPT

Python library to solve optimization problems in Graphs of Convex Sets (GCS).
For a detailed description of the algorithms implemented implemented in this library see the PhD thesis [Graphs of Convex Sets with Applications to Optimal Control and Motion Planning
](https://dspace.mit.edu/handle/1721.1/156598?show=full).
(Please note that the library recently changed name, and in the thesis it is called `gcspy`.)

## Main features

- Uses the syntax of [CVXPY](https://www.cvxpy.org) for describing convex sets and convex functions.
- Provides a simple interface for assembling your graphs.
- Interface with state-of-the-art solvers via [CVXPY](https://www.cvxpy.org/).

## Installation

You can install the latest release from [PyPI](https://pypi.org/project/gcsopt/):
```bash
pip install gcsopt
```

To install from source:
```bash
git clone https://github.com/TobiaMarcucci/gcsopt.git
cd gcsopt
pip install .
```


## Example
Here is a minimal example of how to use gcsopt:
```python
import cvxpy as cp
from gcsopt import GraphOfConvexSets

# Initialize empty directed graph.
G = GraphOfConvexSets(directed=True)

# Add source vertex with circular set.
s = G.add_vertex("s")
xs = s.add_variable(2)
cs = [-2, 0] # Center of the source circle.
s.add_constraint(cp.norm2(xs - cs) <= 1)

# Add target vertex with circular set.
t = G.add_vertex("t")
xt = t.add_variable(2)
ct = [2, 0] # Center of the target circle.
t.add_constraint(cp.norm2(xt - ct) <= 1)

# Add edge from source to target.
e = G.add_edge(s, t)
e.add_cost(cp.sum_squares(xt - xs))

# Solve shortest path problem from source to target.
G.solve_shortest_path(s, t)
print("Problem status:", G.status)
print("Optimal value:", G.value)
print("Optimal solution:")
print("xs =", xs.value)
print("xt =",xt.value)
```

The otput of this script is:
```bash
Problem status: optimal
Optimal value: 4.0
Optimal solution:
xs = [-1.  0.]
xt = [1. 0.]
```

## License
This project is licensed under the MIT License.

## Author
Developed and maintained by Tobia Marcucci.
