from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gcsopt",
    version="0.1.1",
    description="Library for solving optimization problems over Graphs of Convex Sets (GCS).",
    long_description=long_description,
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
