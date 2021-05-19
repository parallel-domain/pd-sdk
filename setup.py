from setuptools import setup

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="paralleldomain",
    version="0.0.2dev",
    author="Nisse Knudsen",
    author_email="nisse@paralleldomain.com",
    packages=["paralleldomain"],
    python_requires=">=3.6",
    long_description="Python SDK for ParallelDomain Datasets",
    install_requires=requirements,
    extras_require={
            "dev": [
                "pytest>=5.3.1",
                "pytest-cov>=2.8.1",
            ]
        },
)
