from setuptools import setup, find_packages

setup(
    name="munis",
    version="1.0.0",
    author="jwohlwend@csail.mit.edu",
    description="Munis",
    packages=find_packages(exclude=["models", "eval"]),
)
