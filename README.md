# Parallel Domain SDK

## Setup

### End-users

Use this procedure if you only want to use the SDK without extending it.

`git clone git@github.com:parallel-domain/pd-sdk.git`

`cd pd-sdk`

`pip install .`

### Developers

Use this procedure if you plan to extend the code base or do adjustments

`git clone git@github.com:parallel-domain/pd-sdk.git`

`cd pd-sdk`

`pip install -e .`

`pre-commit install`

Everytime you commit locally, pre-commit hooks execute on the staged files. These reduces time on CI when creating a PR
for `main` branch. The first run takes a bit longer, but afterwards it should be less than a second.

#### Tests

When changing code, it is recommend to run tests. To do so, set the env
variable `DATASET_PATH=/location/to/sample_dataset` before running the tests.

## Examples

### Load Dataset

Simply load locally or remotely hosted Parallel Domain datasets (DGP format) into convenient Python objects.

```python
from paralleldomain import Dataset
from paralleldomain.decoding.dgp_decoder import DGPDecoder
from paralleldomain.model.annotation import AnnotationTypes

DATASET_PATH = "/content/gdrive/MyDrive/Datasets/public_pdviz_6"
```

We can directly access s3 cloud storage or local copies on hard disk. The SDK finds all available scenes for us and
gives us all the names.

```python
decoder = DGPDecoder(dataset_path=DATASET_PATH)
dataset = Dataset.from_decoder(decoder=decoder)
scene_names = dataset.scene_names
```

Now either iterate over all scenes or pick a specific scene name. For now we will pick the first scene in order.
Afterwards, we will ask to list all sensor names available within the scene, and put them into buckets of lidars and
cameras.

```python
scene_0 = dataset.get_scene(scene_names[0])
lidars = scene_0.lidar_names
cameras = scene_0.camera_names

print("Found LiDARs:", ", ".join(lidars))
print("Found Cameras:", ", ".join(cameras))
```

All frames have also been loaded in order. We can access them by frame index name.

```python
frame_0 = scene_0.get_frame("0")
```

Every frame has the scene sensors attached to them, and we can access each data separately. Images have image.rgba as
the main input (which is the original image as a np.ndarray), but also a virtual point cloud created through the depth
annotation, if available. LiDAR have point_cloud.xyzi as well as other point cloud information like point_cloud.rgb (
photo-colorized points).

### Camera Collage

For camera images, let's create a collage of all RGB images for easier preview in internal systems. We do not care about
final resolution and empty spaces.

```python
import math
import numpy as np
import imageio
from IPython.display import Image  # only for Jupyter notebooks

camera_count = len(cameras)

image_width = 1920  # assuming fixed for now
image_height = 1080

tiles_width = math.ceil(math.sqrt(camera_count))
tiles_height = round(math.sqrt(camera_count))

image_out = np.zeros((tiles_height * 1080, tiles_width * 1920, 3))

for i in range(tiles_width):
    for j in range(tiles_height):
        camera = cameras[i + j]
        camera_sensor = frame_0.get_sensor(camera)  # SDK
        image_out[
            j * image_height : (j + 1) * image_height, i * image_width : (i + 1) * image_width, :
        ] = camera_sensor.image.rgb

imageio.imwrite("collage.jpg", image_out.astype(np.uint8))
display(Image("collage.jpg"))  # only for Jupyter notebooks
```

### Save Full Point Cloud to .pcd

Let's aggregate all lidar sensors' point clouds into one array, and save it as a .pcd file to the hard disk. Also, let's
save each sensor's extrinsic in a json file.

```python
import numpy as np
import open3d as o3d
import json

point_cloud_xyz = np.empty((0, 3))  # Initialize empty `np.ndarray` for storing xyz
point_cloud_rgb = np.empty((0, 3))  # Initialize empty `np.ndarray` for storing rgb
extrinsics = {}

for l in lidars:
    lidar_sensor = frame_0.get_sensor(l)  # Returns sensor object for frame

    extrinsics[l] = {
        "rotation": {
            "qw": lidar_sensor.extrinsic.quaternion.w,
            "qx": lidar_sensor.extrinsic.quaternion.x,
            "qy": lidar_sensor.extrinsic.quaternion.y,
            "qz": lidar_sensor.extrinsic.quaternion.z,
        },
        "translation": {
            "x": lidar_sensor.extrinsic.translation[0],
            "y": lidar_sensor.extrinsic.translation[1],
            "z": lidar_sensor.extrinsic.translation[2],
        },
    }

    pc_vehicle_reference = (lidar_sensor.extrinsic.transformation_matrix @ lidar_sensor.point_cloud.xyz_one.T).T
    pc_vehicle_reference = pc_vehicle_reference[:, :3]

    point_cloud_xyz = np.vstack([point_cloud_xyz, pc_vehicle_reference])  # append to xyz point array
    point_cloud_rgb = np.vstack([point_cloud_rgb, lidar_sensor.point_cloud.rgb])  # append to rgb point array

# Write extrinsic using python's json module
with open("extrinsics.json", "w") as f:
    json.dump(extrinsics, f)

# Use Open3d's pcd-writer for output
# http://www.open3d.org/docs/latest/tutorial/Basic/working_with_numpy.html#From-NumPy-to-open3d.PointCloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud_xyz)
pcd.colors = o3d.utility.Vector3dVector(point_cloud_rgb)
o3d.io.write_point_cloud("points.pcd", pcd)
```

### Access Annotation Styles

On every dataset you can list the available annotation types. Those as objects which can be passed onto a `SensorFrame`
object. Let's get the 3D Bounding Boxes from a LiDAR sensor as well as 2D Optical Flow from a camera image. First we
check if the annotation style are present in the dataset and then return them.

```python
assert AnnotationTypes.BoundingBoxes3D in dataset.available_annotation_types
assert AnnotationTypes.OpticalFlow in dataset.available_annotation_types

boxes3d = frame_0.get_sensor(lidars[0]).get_annotations(AnnotationTypes.BoundingBoxes3D)

for b in boxes3d.boxes[:10]:
    print(b)

optiflow = frame_0.get_sensor(cameras[0]).get_annotations(AnnotationTypes.OpticalFlow)
print(optiflow.vectors.shape)
```

More details about annotation types and their property can be found in `model/annotation.py`.
