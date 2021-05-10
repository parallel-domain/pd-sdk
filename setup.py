from setuptools import setup

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="paralleldomain",
    version="0.1dev",
    author="Nisse Knudsen",
    author_email="nisse@paralleldomain.com",
    packages=["paralleldomain"],
    python_requires=">=3.7",
    long_description="Python SDK for ParallelDomain Datasets",
    install_requires=requirements,
)
