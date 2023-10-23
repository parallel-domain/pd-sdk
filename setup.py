import glob

from setuptools import find_packages, setup

setup(
    name="paralleldomain",
    version="0.16.0",
    author=", ".join(["Nisse Knudsen", "Phillip Thomas", "Lars Pandikow", "Michael Stanley"]),
    author_email=", ".join(
        [
            "nisse@paralleldomain.com",
            "phillip@paralleldomain.com",
            "lars@paralleldomain.com",
            "michael.stanley@paralleldomain.com",
        ]
    ),
    # `.scripts` is copied into the package during install via package_dir
    packages=find_packages(exclude=["test_paralleldomain"])
    + ["paralleldomain.scripts.examples", "paralleldomain.scripts.internal"],
    python_requires=">=3.8",
    long_description="Python SDK for Parallel Domain Data Lab and Datasets",
    install_requires=[
        "Pillow>=6.2.1,<11.0.0",
        "awscli>=1.0.0,<2.0.0",
        "bezier>=2023.7.28",
        "cachetools>=4.2.2,<5.0.0",
        "cgroupspy>=0.2.1,<1.0.0",
        "coloredlogs>=15.0.1,<16.0.0",
        "dataclasses_json>=0.5.3,<1.0.0",
        "deprecation>=2.1.0,<3.0.0",
        "humanize>=3.10.0,<5.0.0",
        "igraph>=0.9.8,<1.0.0",
        "imagesize>=1.3.0,<1.4.0",
        "iso8601>=0.1.16,<1.0.0",
        "more-itertools>=8.11.0,<9.0.0",
        "numpy>=1.19,<2.0",
        "opencv-python-headless>=4.5.3.56,<5.0.0.0",
        "opensimplex>=0.4.4,<1.0.0",
        "protobuf>=3.20.1,<4.0.0",
        "pypeln>=0.4.9,<1.0.0",
        "pyquaternion>=0.9.9,<1.0.0",
        "s3path>=0.4.1,<1.0.0",
        "tqdm>=4.55.3,<5.0.0",
        "transforms3d>=0.3.1,<1.0.0",
        "ujson>=5.1.0,<6.0.0",
    ],
    include_package_data=True,
    package_dir={
        "paralleldomain.scripts.examples": "examples",
        "paralleldomain.scripts.internal": "internal",
    },
    data_files=[
        (".", glob.glob("paralleldomain/decoding/waymo_open_dataset/pre_calculated/**")),
    ],
    extras_require={
        "data_lab": [
            "step-sdk @ git+https://github.com/parallel-domain/step-sdk.git@v2.7.0",
            "rerun-sdk>=0.9.1,<0.10.0",
            "py7zr>=0.20.5,<1.0.0",
        ],
        "statistics": [
            "filelock>=3.0.0,<4.0.0",  # no dependencies
            "pandas>=1.3.5,<2.0.0",  # depends on pytz, tzdata, six, numpy, python-dateutil
            "watchdog>=3.0.0,<4.0.0",  # no dependencies
        ],
        "visualization": [
            "rerun-sdk>=0.9.1,<0.10.0",
            "imgui[glfw]>=2.0",
            "filelock>=3.0.0,<4.0.0",  # no dependencies
            "pandas>=1.3.5,<2.0.0",  # depends on pytz, tzdata, six, numpy, python-dateutil
            "watchdog>=3.0.0,<4.0.0",  # no dependencies
        ],
        "dash": [
            "dash-core-components>=2.0.0",
            "dash>=2.9.3",
            "imgui[glfw]>=2.0",
            "jupyter-dash>=0.4.2",
            "nbformat>=5.0.0",
        ],
        "dev": [
            "black==22.6.0",
            "git-filter-repo>=2.34.0,<3.0.0",
            "mypy>=1.3.0",
            "pre-commit>=2.13.0,<3.0.0",
            "pytest-cov>=2.12.1,<3.0.0",
            "pytest>=7.2.2,<8.0.0",
            "ruff>=0.0.280",
            "types-cachetools",
            "types-ujson",
            # These are required for pytest tests/yaml
            "rpyc>=5.3.1",
            "pytest-subtests>=0.11.0",
            "scipy>=1.10.1",
            # These are required for downloading maps if you run pytest tests/yaml with a local instance
            "records>=0.5.3",
            "logzero>=1.7.0",
            "prettytable>=3.9.0",
            "google-cloud-storage~=1.44.0",
            # Required for upload in jenkins
            "elasticsearch==7.13.4",
        ],
        "remote": [
            "ray==2.7.0",
        ],
    },
    zip_safe=False,
    entry_points={
        "console_scripts": ["pd=paralleldomain.cli:pd_cli"],
    },
)
