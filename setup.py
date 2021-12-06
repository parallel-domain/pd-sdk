from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="paralleldomain",
    version="0.5.3",
    author=", ".join(["Nisse Knudsen", "Phillip Thomas", "Lars Pandikow", "Michael Stanley"]),
    author_email=", ".join(
        [
            "nisse@paralleldomain.com",
            "phillip@paralleldomain.com",
            "lars@paralleldomain.com",
            "michael.stanley@paralleldomain.com",
        ]
    ),
    packages=find_packages(exclude=["test_paralleldomain"]),
    package_data={
        "paralleldomain": ["py.typed"],
    },
    python_requires=">=3.6",
    long_description="Python SDK for ParallelDomain Datasets",
    install_requires=requirements,
    include_package_data=True,
    extras_require={
        "dev": [
            "pytest>=5.3.1,<6.0.0",
            "pytest-cov>=2.8.1,<3.0.0",
            "types-ujson",
            "types-cachetools",
            "pre-commit>=2.13.0,<3.0.0",
        ],
    },
    zip_safe=False,
)
