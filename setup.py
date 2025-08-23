from setuptools import setup, find_packages

setup(
    name="gcsopt",
    version="0.1.0",
    description="Library for solving optimization problems over Graphs of Convex Sets (GCS).",
    author="Tobia Marcucci",
    author_email='marcucci@ucsb.edu',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        "numpy",
        "cvxpy >= 1.5",
        "pytest",
    ],
)
