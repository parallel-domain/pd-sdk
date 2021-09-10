# Parallel Domain SDK

## Introduction

The Parallel Domain SDK (or short: PD SDK) allows the community to access Parallel Domain's synthetic data as Python objects.

The PD SDK can decode different data formats into its Python objects, including [Dataset Governance Policy (DGP)](https://github.com/TRI-ML/dgp/blob/master/dgp/proto/README.md) format.
Currently, local file system and s3 buckets are supported as dataset locations for decoding.

## Quick Start

To use the PD SDK, you can simply install it using `pip` in your Python environment. Activate your Python environment before-hand, if wanted.


```bash
# Clone latest PD SDK release
$ git clone git@github.com:parallel-domain/pd-sdk.git

# Change directory
$ cd pd-sdk

# Optional: Parallelize build process for dependencies using gcc, e.g., `opencv-python-headless`
$ export MAKEFLAGS="-j$(nproc)"

# Install PD SDK from local clone
$ pip install .
```

---

**Supported Python Versions:**

* Python3.6
* Python3.7
* Python3.8
* Python3.9

---

### Developers

If you are looking to extend or debug the PD SDK, use the following install procedure.

```bash
# Clone latest PD SDK release
$ git clone git@github.com:parallel-domain/pd-sdk.git

# Change directory
$ cd pd-sdk

# Optional: Parallelize build process for dependencies using gcc, e.g., `opencv-python-headless`
$ export MAKEFLAGS="-j$(nproc)"

# Install PD SDK from local clone with developer dependencies
$ pip install -e .[dev]
```

To make code contributions more stream-lined, we provide local [pre-commit hooks](https://pre-commit.com/) that check / adapt to the PD SDK style guidelines automatically on commit.
It is optional to set those up, but it helps to keep all code in the same style.

```bash
# within local PD SDK directory
$ pre-commit install
```

After setting up pre-commit hooks, every file in a commit will be checked by a number of hooks. If any of the hooks fail, the commit is rejected. Fortunately, most of the hooks are auto-correcting, so they edit
the files per the style guidelines and you can just commit again. Only `flake8` lint errors can just be solved by human editing.

#### Tests

Tests can be executed using `pytest` and providing a location to a DGP dataset which should be used during the test run.
```bash
# within local PD SDK directory - set env variable DATASET_PATH to a local or s3 location.
$ DATASET_PATH=/data/test_dataset pytest .
```

#### Documentation

You can find more details about adding to and building PD SDK's documentation in `docs/README.md`

## Documentation

### Tutorials

There are several tutorials available covering common use cases. Those can be found under [Documentation -> Tutorials](https://parallel-domain.github.io/pd-sdk/).
In case you are missing an important tutorial, feel free to request it via a Github Issue or create a PR, in case you have written one already yourself.

### API Reference

Public classes / methods / properties are annotated with Docstrings. The compiled API Reference can be found under [Documentation -> API Reference](https://parallel-domain.github.io/pd-sdk/)
