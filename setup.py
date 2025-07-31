from setuptools import setup, find_packages

setup(
    name='gcspy',
    version='0.1.0',
    description="Library for solving optimization problems over Graphs of Convex Sets (GCS).",
    author="Tobia Marcucci",
    packages=find_packages(),
    install_requires=[
        "cvxpy >= 1.5"
    ],
)
