from setuptools import setup, find_packages

setup(
    name='gcspy',
    version='0.1.0',
    description="Library based on CVXPY to solve optimization problems in Graphs of Convex Sets (GCS). The techniques implemented here are based on the paper Shortest Paths in Graphs of Convex Sets by Tobia Marcucci, Jack Umenberger, Pablo A. Parrilo, and Russ Tedrake.",
    author="Tobia Marcucci",
    packages=find_packages(),
    install_requires=[
        "cvxpy >= 1.5"
    ],
)