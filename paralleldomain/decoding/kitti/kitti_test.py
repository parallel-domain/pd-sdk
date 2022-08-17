from paralleldomain.decoding.helper import decode_dataset
from paralleldomain.model.annotation.optical_flow import OpticalFlow
from paralleldomain.utilities.any_path import AnyPath

decoder_kwargs = dict(
    split_name="training",
    image_folder="image_2",
    occ_optical_flow_folder="flow_occ",
    noc_optical_flow_folder="flow_noc",
    use_non_occluded=False,
    camera_name="default",
)

dataset_path = "s3://pd-internal-ml/flow/KITTI2015"
kitti_train_dataset = decode_dataset(dataset_path=dataset_path, dataset_format="kitti", **decoder_kwargs)

scene_names = kitti_train_dataset.unordered_scene_names
scene = kitti_train_dataset.get_unordered_scene(scene_name=scene_names[0])
print(scene)
# for sf in dataset.get_sensor_frames():
#     print(sf.get_annotations(annotation_type=OpticalFlow))
