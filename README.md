# Parallel Domain SDK
 [**Documentation**](https://parallel-domain.github.io/pd-sdk/index.html) | [**Documentation -> Tutorials**](https://parallel-domain.github.io/pd-sdk/tutorial/general/index.html) | [**Documentation -> API Reference**](https://parallel-domain.github.io/pd-sdk/api/dataset.html)

Supported Python Versions: **3.7** | **3.8** | **3.9**

## Install
For more detailed instructions see [here](INSTALL.md). For default installation run:

```bash
pip install "paralleldomain @ git+ssh://git@github.com/parallel-domain/pd-sdk-internal@main#egg=paralleldomain"
```


## Introduction

The Parallel Domain SDK (or short: PD SDK) allows the community to access Parallel Domain's synthetic data as Python objects.

The PD SDK can also decode different data formats into its common Python object represenation (more public dataset formats will be supported in the future):
- [Dataset Governance Policy (DGP)](https://github.com/TRI-ML/dgp/blob/master/dgp/proto/README.md)
- [CityScapes](https://www.cityscapes-dataset.com/dataset-overview/)
- [NuImages](https://www.nuscenes.org/nuimages)
- [NuScenes](https://www.nuscenes.org/nuscenes)
- [Flying Chairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html)
- [Flying Things](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
- [GTA5 / Playing for Data](https://download.visinf.tu-darmstadt.de/data/from_games/)
- [KITTI Optical Flow](https://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=flow)

Currently, local file system and s3 buckets are supported as dataset locations for decoding. See [AnyPath](https://parallel-domain.github.io/pd-sdk/tutorial/any_path/index.html) documentation for our file system abstraction.

The two main use cases PD SDK is designed for are:

- to load data in ML data pipelines from local or cloud storage directly into RAM.
- to encode data into different dataset formats. Currently it's possible to convert into DGP and DGPv1 format.

More details on the [**Architecture**](paralleldomain/ARCHITECTURE.md) of PD SDK can be found [here](paralleldomain/ARCHITECTURE.md).

### Example: Load and visualize data
To run ``show_sensor_frame`` you need to install the visualization dependencies with (See our [Istnall Instruvtions](INSTALL.md) for more install options):

```bash
pip install "paralleldomain[visualization] @ git+ssh://git@github.com/parallel-domain/pd-sdk-internal@main#egg=paralleldomain"
```
Then you can decode a dataset at a given local or s3 path using the ``decode_dataset`` method.
To have quick access to all sensor frames in a dataset you can use the ``sensor_frame_pipeline`` method of a dataset.
It will yield all the sensor frames of a dataset in order. So the outer loop will be scenes, then frames and then
sensors in each frame.

``` python
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

For more examples make sure to check out our [**Tutorials**](https://parallel-domain.github.io/pd-sdk/tutorial/general/index.html)!

## Documentation

### Tutorials

There are several tutorials available covering common use cases. Those can be found under [Documentation -> Tutorials](https://parallel-domain.github.io/pd-sdk/).
In case you are missing an important tutorial, feel free to request it via a Github Issue or create a PR, in case you have written one already yourself.

### API Reference

Public classes / methods / properties are annotated with Docstrings. The compiled API Reference can be found under [Documentation -> API Reference](https://parallel-domain.github.io/pd-sdk/)


## Testing
Before running `pytest` you need to make sure to have its package installed. If you followed the Developer Setup, `pytest` should be already available.
If you haven't followed the Developer Setup or are unsure, just run:

For OS X / Linux users:
```bash
pip install -e ".[dev,data_lab]"
```

For Windows users:
```powershell
pip install -e .[dev,data_lab]
```

Go to the root folder of your pd-sdk repo and run:
```bash
pytest test_paralleldomain
```

If you'd like to run tests for Data Lab, make sure that your PD_CLIENT_ORG_ENV and PD_CLIENT_STEP_API_KEY_ENV are set.
Otherwise those tests will be skipped.

For OS X / Linux users:
```bash
export PD_CLIENT_STEP_API_KEY_ENV="my api key"
export PD_CLIENT_ORG_ENV="paralleldomain"
```

For Windows users:
```powershell
$env:PD_CLIENT_STEP_API_KEY_ENV="my api key"
$env:PD_CLIENT_ORG_ENV="paralleldomain"
```

You can run just Data Lab related test by running:
```bash
pytest test_paralleldomain/data_lab/
```
