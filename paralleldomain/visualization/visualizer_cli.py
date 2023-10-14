import argparse

from paralleldomain.decoding.helper import decode_dataset
from paralleldomain.model.annotation import AnnotationTypes
from paralleldomain.utilities.dataset_transform import DatasetTransformation
from paralleldomain.visualization.model_visualization import show_dataset


def show_dataset_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--dataset_path", type=str, required=True, help="Path to the dataset location (s3 or local)"
    )
    parser.add_argument(
        "-f",
        "--dataset_format",
        type=str,
        default="dgp",
        help="Format of the dataset. Defaults to dgp. Options are all the registered decoders"
        "like data-stream, dgp, dgpv1, nuimages, nuscenes, cityscapes, flying-things, flying-chairs,"
        "kitti, kitti-flow, gta,waymo_open_dataset. Note that some decoders might require additional arguments"
        "check the decoder init for their names. THey can be passed as kwargs to the cli. "
        "E.g: --split_name=training in the waymo_open_dataset case",
    )
    parser.add_argument(
        "--scene_names",
        nargs="+",
        default=None,
        help="Scene names to visualize. If not provided, all scenes will be visualized",
    )
    parser.add_argument(
        "--frame_ids",
        nargs="+",
        default=None,
        help="Frame ids to visualize. If not provided, all frames will be visualized",
    )
    parser.add_argument(
        "--sensor_names",
        nargs="+",
        default=None,
        help="Sensor names to visualize. If not provided, all sensors will be visualized",
    )
    parser.add_argument(
        "--annotations",
        nargs="+",
        default=None,
        help="Annotations to visualize. If not provided, all annotations will be visualized."
        "Note that you just need to pass the name of the AnnotationType. E.g: BoundingBoxes2D, Depth etc.",
    )

    args, unknown = parser.parse_known_args()
    kwargs = dict()
    for arg in unknown:
        if arg.startswith(("-", "--")):
            k, v = arg.split("=")
            kwargs[k.replace("--", "").replace("-", "")] = v

    scene_names = args.scene_names
    frame_ids = args.frame_ids
    sensor_names = args.sensor_names
    annotations = args.annotations

    if annotations is not None:
        annotations = [getattr(AnnotationTypes, name) for name in annotations]

    data_transform = DatasetTransformation(
        annotation_identifiers=annotations,
        sensor_names=sensor_names,
        frame_ids=frame_ids,
        scene_names=scene_names,
    )

    dataset = decode_dataset(dataset_path=args.dataset_path, dataset_format=args.dataset_format, **kwargs)
    dataset = data_transform.apply(dataset)
    show_dataset(dataset)
