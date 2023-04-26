# Parallel Domain SDK
 
## Introduction

The Parallel Domain SDK (or short: PD SDK) allows the community to access Parallel Domain's synthetic data as Python objects.

The PD SDK can also decode different data formats into its common Python object representation (more public dataset formats will be supported in the future):
- [Dataset Governance Policy (DGP)](https://github.com/TRI-ML/dgp/blob/master/dgp/proto/README.md)
- [Cityscapes](https://www.cityscapes-dataset.com/dataset-overview/)
- [NuImages](https://www.nuscenes.org/nuimages)
- [NuScenes](https://www.nuscenes.org/nuscenes)
- [Flying Chairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html)
- [Flying Things](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
- [GTA5 / Playing for Data](https://download.visinf.tu-darmstadt.de/data/from_games/)
- [KITTI Optical Flow](https://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=flow)

Currently, local file system and s3 buckets are supported as dataset locations for decoding. See [AnyPath](https://parallel-domain.github.io/pd-sdk/tutorial/any_path/index.html) documentation for our file system abstraction.

PD SDK is designed to serve the following use cases:

- load data in ML data pipelines from local or cloud storage directly into memory.
- encode data into different dataset formats. Currently, it's possible to convert into DGP and DGPv1 format.
- generate data in PD's Data Lab.

More details on the [**Architecture**](paralleldomain/ARCHITECTURE.md) of PD SDK can be found [here](paralleldomain/ARCHITECTURE.md).

### Example: Load and visualize data
To run `show_sensor_frame` you need to install the `visualization` dependencies with one of the methods described in [Installation](#installation).

Next, you can decode a dataset located at a given local or S3 path using the `decode_dataset` method.
To quickly access all sensor frames in a dataset, use the `sensor_frame_pipeline` method for that dataset. This method yields all the sensor frames in order, so the outer loop iterates through scenes, followed by frames, and finally sensors within each frame.

```python
from paralleldomain.decoding.helper import decode_dataset
from paralleldomain.model.annotation import AnnotationTypes
from paralleldomain.visualization.sensor_frame_viewer import show_sensor_frame

pd_dataset = decode_dataset(dataset_path="s3://bucket/with/dgp/dataset", dataset_format="dgp")

for sensor_frame, frame, scene in pd_dataset.sensor_frame_pipeline():
    show_sensor_frame(
        sensor_frame=camera_frame,
        annotations_to_show=[AnnotationTypes.BoundingBoxes2D],
    )
```

For more examples make sure to check out our [Documentation](#documentation).

## Documentation

### Tutorials

There are several tutorials available covering common use cases. Those can be found under [Documentation -> Tutorials](https://parallel-domain.github.io/pd-sdk/).
In case you are missing an important tutorial, feel free to request it via a GitHub Issue or create a PR, in case you have written one already yourself.

### API Reference

Public classes / methods / properties are annotated with Docstrings. The compiled API Reference can be found under [Documentation -> API Reference](https://parallel-domain.github.io/pd-sdk/)


## Installation
Supported Python Versions: **3.8**, **3.9**, **3.10**, **3.11**

Choose one of the following installation methods for the Python package from GitHub based on your use case:

### Quick Installation
For users who just want to use the library without editing it, this method is recommended. It quickly installs the package from GitHub, allowing you to access its functionalities without modifying the source code. Run the following command:

```bash
pip install "paralleldomain @ git+https://github.com/parallel-domain/pd-sdk.git@main#egg=paralleldomain"
```

### Developer Setup

This method is suitable for developers who need to modify the source code, contribute to the project, or parallelize the build process. With this setup, you can make changes to the library, test new features, and experiment with the codebase. Follow these steps:

```bash
# Clone latest PD SDK release
git clone https://github.com/parallel-domain/pd-sdk.git

# Change directory
cd pd-sdk

# Optional: Parallelize build process for dependencies using gcc, e.g., `opencv-python-headless`
export MAKEFLAGS="-j$(nproc)"

# Install PD SDK from local clone
pip install .
```

This method allows you to directly work on the library's source code, which is not possible with the quick installation method.

### Install Extras

These optional extras can be installed to enhance the functionality of PD SDK based on your specific needs:

- `data_lab`: This extra includes dependencies for Data Lab, PD's synthetic data generation platform (includes `visualization`).

- `visualization`: Install this extra to include `opencv` with GUI components, helpful to visualize data.

- `dev`: The development extra contains dependencies for developers, such as testing tools, pre-commit hooks, and other utilities that assist in maintaining code quality and ensuring a smooth development process. This is recommended for users who plan to contribute to the project or work extensively with the source code.


## Testing
Before running `pytest` you need to make sure to have its package installed. To do this, add the `dev` install extra as described in [Install Extras](#install-extras).

Go to the root folder of your pd-sdk repo and run:
```bash
pytest test_paralleldomain
```

If you'd like to run tests for Data Lab, make sure that your `PD_CLIENT_CREDENTIALS_PATH_ENV`, `PD_CLIENT_STEP_API_KEY_ENV` and `PD_CLIENT_ORG_ENV` environment variables are set, and you have the `data_lab` install extra set up.
Otherwise, those tests will be skipped. You can find more details on how to set those in [Data Lab Quickstart](https://app.paralleldomain.com/docs/latest/data-lab-quickstart)
