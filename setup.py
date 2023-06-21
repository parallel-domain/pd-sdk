import glob

from setuptools import find_packages, setup

setup(
    name="paralleldomain",
    version="0.11.2",
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
    python_requires=">=3.8",
    long_description="Python SDK for Parallel Domain Datasets",
    install_requires=[
        "awscli>=1.0.0,<2.0.0",
        "coloredlogs>=15.0.1,<16.0.0",
        "cachetools>=4.2.2,<5.0.0",
        "cgroupspy>=0.2.1,<1.0.0",
        "dataclasses_json>=0.5.3,<1.0.0",
        "humanize>=3.10.0,<5.0.0",
        "iso8601>=0.1.16,<1.0.0",
        "igraph>=0.9.8,<1.0.0",
        "more-itertools>=8.11.0,<9.0.0",
        "numpy>=1.19,<2.0",
        "opencv-python-headless>=4.5.3.56,<5.0.0.0",
        "protobuf>=3.20.1,<4.0.0",
        "pyquaternion>=0.9.9,<1.0.0",
        "transforms3d>=0.3.1,<1.0.0",
        "typing-extensions>=3.6.6,<5.0.0.0",
        "s3path>=0.4.1,<1.0.0",
        "ujson>=5.1.0,<6.0.0",
        "imagesize>=1.3.0,<1.4.0",
        "pypeln>=0.4.9,<1.0.0",
        "tqdm>=4.55.3,<5.0.0",
        "Pillow>=6.2.1,<10.0.0",
    ],
    include_package_data=True,
    data_files=glob.glob("paralleldomain/decoding/waymo_open_dataset/pre_calculated/**"),
    extras_require={
        "data_lab": [
            "step-sdk @ git+https://github.com/parallel-domain/step-sdk.git",
            "opencv-python>=4.5.3.56,<5.0.0.0",
        ],
        "visualization": ["opencv-python>=4.5.3.56,<5.0.0.0"],
        "dev": [
            "git-filter-repo>=2.34.0,<3.0.0",
            "pytest>=7.2.2,<8.0.0",
            "pytest-cov>=2.12.1,<3.0.0",
            "types-ujson",
            "types-cachetools",
            "pre-commit>=2.13.0,<3.0.0",
        ],
    },
    zip_safe=False,
    entry_points={"console_scripts": ["pd-credentials-setup=paralleldomain.generation.credentials:main"]},
)
