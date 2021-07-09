import argparse
import hashlib
import json
import logging
import multiprocessing
import uuid
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from cloudpathlib import CloudPath
from joblib import Parallel, delayed, parallel_backend
from more_itertools import windowed
from PIL import Image

from paralleldomain import Dataset, Scene
from paralleldomain.decoding.dgp.decoder import DGPDecoder
from paralleldomain.encoding.dgp.dtos import (
    AnnotationsBoundingBox2DDTO,
    AnnotationsBoundingBox3DDTO,
    BoundingBox2DBoxDTO,
    BoundingBox2DDTO,
    BoundingBox3DBoxDTO,
    BoundingBox3DDTO,
    CalibrationDTO,
    CalibrationExtrinsicDTO,
    CalibrationIntrinsicDTO,
    DatasetDTO,
    DatasetMetaDTO,
    DatasetSceneSplitDTO,
    OntologyFileDTO,
    OntologyItemColorDTO,
    OntologyItemDTO,
    PoseDTO,
    RotationDTO,
    SceneDataDatum,
    SceneDataDatumImage,
    SceneDataDatumPointCloud,
    SceneDataDatumTypeImage,
    SceneDataDatumTypePointCloud,
    SceneDataDTO,
    SceneDataIdDTO,
    SceneDTO,
    SceneMetadataDTO,
    SceneSampleDTO,
    SceneSampleIdDTO,
    TranslationDTO,
)
from paralleldomain.encoding.encoder import Encoder
from paralleldomain.model.annotation import AnnotationType, AnnotationTypes
from paralleldomain.model.class_mapping import ClassIdMap, ClassMap
from paralleldomain.model.frame import Frame
from paralleldomain.model.sensor import SensorFrame
from paralleldomain.utilities.any_path import AnyPath

logger = logging.getLogger(__name__)


def _json_write(obj: object, fp: Union[Path, CloudPath, str], append_sha1: bool = False) -> str:
    fp = AnyPath(fp)
    fp.parent.mkdir(parents=True, exist_ok=True)

    json_str = json.dumps(obj, indent=2)

    if append_sha1:
        json_str_sha256 = hashlib.sha1(json_str.encode()).hexdigest()
        filename = fp.name.split(".")
        if filename[0] == "":
            filename[0] = json_str_sha256
        else:
            filename[0] = f"{filename[0]}_{json_str_sha256}"

        fp = AnyPath(fp.parent / ".".join(filename))

    with fp.open("w") as json_file:
        json_file.write(json_str)

    return fp.name


def _png_write(obj: object, fp: Union[Path, CloudPath, str]) -> str:
    fp = AnyPath(fp)
    fp.parent.mkdir(parents=True, exist_ok=True)

    with fp.open("wb") as png_file:
        Image.fromarray(obj).save(png_file, "png")

    return fp.name


def _npz_write(npz_kwargs: Dict[str, np.ndarray], fp: Union[Path, CloudPath, str]) -> str:
    fp = AnyPath(fp)
    fp.parent.mkdir(parents=True, exist_ok=True)

    with fp.open("wb") as npz_file:
        np.savez_compressed(npz_file, **npz_kwargs)

    return fp.name


def _attribute_key_dump(obj: object) -> str:
    return str(obj)


def _attribute_value_dump(obj: object) -> str:
    if isinstance(obj, Dict) or isinstance(obj, List):
        return json.dumps(obj, indent=2)
    else:
        return str(obj)


def _vectors_to_rgba(vectors: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [vectors[..., [0]] >> 8, vectors[..., [0]] & 0xFF, vectors[..., [1]] >> 8, vectors[..., [1]] & 0xFF], axis=-1
    ).astype(np.uint8)


"""
def _num_points_in_box(xyz_one: np.ndarray, box: BoundingBox3D) -> np.ndarray:
    to_local_box = box.pose.inverse
    point_in_local_box_coords = (to_local_box @ xyz_one.T).T
    box_size = np.array([box.length, box.width, box.height])
    min_point = -0.5 * box_size
    max_point = 0.5 * box_size
    points_greater_than_min = np.all(point_in_local_box_coords[:, :3] >= min_point, axis=-1)
    points_smaller_than_max = np.all(point_in_local_box_coords[:, :3] <= max_point, axis=-1)
    is_point_in_aa_bb = np.logical_and(points_greater_than_min, points_smaller_than_max)
    return point_in_local_box_coords[is_point_in_aa_bb].shape[0]
"""


# noinspection InsecureHash
class DGPEncoder(Encoder):
    _fisheye_camera_model_map: Dict[str, int] = defaultdict(
        lambda: 2,
        {
            "brown_conrady": 0,
            "fisheye": 1,
        },
    )

    _annotation_type_map: Dict[AnnotationType, str] = {
        AnnotationTypes.BoundingBoxes2D: "0",
        AnnotationTypes.BoundingBoxes3D: "1",
        AnnotationTypes.SemanticSegmentation2D: "2",
        AnnotationTypes.SemanticSegmentation3D: "3",
        AnnotationTypes.InstanceSegmentation2D: "4",
        AnnotationTypes.InstanceSegmentation3D: "5",
        AnnotationTypes.Depth: "6",
        # "7": Annotation,  # Surface Normals 3D
        AnnotationTypes.OpticalFlow: "8",
        # "9": Annotation,  # Motion Vectors 3D aka Scene Flow
        # "10": Annotation,  # Surface normals 2D
    }

    def __init__(
        self,
        dataset: Dataset,
        output_path: AnyPath,
        custom_map: Optional[ClassMap] = None,
        custom_id_map: Optional[ClassIdMap] = None,
        annotation_types: Optional[List[AnnotationType]] = None,
        frame_slice: Optional[slice] = None,
        thread_count: Optional[int] = None,
    ):
        self.dataset = dataset
        self.custom_map = custom_map
        self.custom_id_map = custom_id_map
        self.annotation_types = list(self._annotation_type_map.keys()) if annotation_types is None else annotation_types
        self._dataset_path: Union[Path, CloudPath] = AnyPath(output_path)
        self._frame_slice: slice = slice(None, None, None) if frame_slice is None else frame_slice
        self._thread_count: int = multiprocessing.cpu_count() if thread_count is None else thread_count
        self._scene_paths: List[str] = []

    def finalize(self):
        self._save_dataset_json()

    def encode_dataset(self):
        for s in self.dataset.scene_names:
            self.encode_scene(self.dataset.get_scene(s))

    def encode_scene(self, scene: Scene) -> str:
        scene_data = []
        scene_samples = []

        for sn in scene.sensor_names:
            sensor_frames = [f.get_sensor(sn) for f in scene.frames[self._frame_slice]]
            sensor_data = self._encode_sensor_frames_by_sensor(sensor_frames=sensor_frames, scene_name=scene.name)
            scene_data.append(sensor_data)

        for f in scene.frames[self._frame_slice]:
            sensor_frames = [f.get_sensor(sn) for sn in scene.sensor_names]
            frame_data = self._encode_sensor_frames_by_frame(
                frame=f, sensor_frames=sensor_frames, scene_name=scene.name
            )
            scene_samples.append(frame_data)

        for i, s_sample in enumerate(scene_samples):
            data_keys = [s_data[i].key for s_data in scene_data]
            s_sample.datum_keys = data_keys

        ontology_dict: Dict[str, str] = {}
        for a_type in self.annotation_types:
            for sn in scene.sensor_names:
                sn_frame_0 = scene.get_sensor(sn).get_frame(scene.frame_ids[0])
                if a_type in sn_frame_0.available_annotation_types:
                    a_value = sn_frame_0.get_annotations(a_type)
                    try:
                        a_class_map: ClassMap = a_value.class_map
                    except AttributeError:
                        # "annotations" like OpticalFlow, Depth, Instanced do not know class_map concept
                        a_class_map: ClassMap = ClassMap.from_id_label_dict({})

                    ontology = OntologyFileDTO(
                        items=[
                            OntologyItemDTO(
                                name=cd.name,
                                id=cd.id,
                                isthing=cd.instanced,
                                color=OntologyItemColorDTO(
                                    r=0 if "color" not in cd.meta else cd.meta["color"]["r"],
                                    g=0 if "color" not in cd.meta else cd.meta["color"]["g"],
                                    b=0 if "color" not in cd.meta else cd.meta["color"]["b"],
                                ),
                                supercategory="",
                            )
                            for _, cd in a_class_map.items()
                        ]
                    )

                    relative_path = Path("ontology")
                    filename = ".json"
                    output_path = self._dataset_path / scene.name / relative_path / filename

                    filename = _json_write(ontology.to_dict(), output_path, append_sha1=True)
                    ontology_dict[self._annotation_type_map[a_type]] = filename.split(".")[0]
                    break

        scene_data = [sensor_datum for sensor_data in scene_data for sensor_datum in sensor_data]  # flatten list

        scene_path = self._save_scene_json(
            scene=scene, scene_samples=scene_samples, scene_data=scene_data, ontologies=ontology_dict
        )
        self._scene_paths.append(scene_path)

        return scene_path

    def _encode_sensor_frames_by_frame(self, frame: Frame, sensor_frames: List[SensorFrame], scene_name: str):
        frame_data = SceneSampleDTO(
            id=SceneSampleIdDTO(
                log="", timestamp=frame.date_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"), name="", index=frame.frame_id
            ),
            datum_keys=[],
            calibration_key=self._save_calibration_json(sensor_frames=sensor_frames, scene_name=scene_name),
            metadata={},
        )

        return frame_data

    def _encode_sensor_frames_by_sensor(self, sensor_frames: List[SensorFrame], scene_name: str):
        sorted_sensor_frames = sorted(sensor_frames, key=lambda x: x.date_time)

        with parallel_backend("threading", n_jobs=self._thread_count):
            sensor_data = Parallel()(
                delayed(self._encode_sensor_frame)(sensor_frame=sf, scene_name=scene_name)
                for sf in sorted_sensor_frames
            )

        padding = 2 * [None]
        for window in windowed(chain(padding, sensor_data, padding), 3, fillvalue=None):
            if window[1] is not None:
                window[1].prev_key = "" if window[0] is None else window[0].key
                window[1].next_key = "" if window[2] is None else window[2].key

        return sensor_data

    def _encode_sensor_frame(self, sensor_frame: SensorFrame, scene_name: str):
        data_key = hashlib.sha256(str(uuid.uuid4()).encode()).hexdigest()

        if sensor_frame.point_cloud is not None:
            annotations = {}
            for a_type in self.annotation_types:
                if a_type in sensor_frame.available_annotation_types:
                    a_key = self._annotation_type_map[a_type]
                    if a_type is AnnotationTypes.BoundingBoxes3D:
                        a_value = self._save_bounding_box_3d(sensor_frame=sensor_frame, scene_name=scene_name)
                    elif a_type is AnnotationTypes.SemanticSegmentation3D:
                        a_value = self._save_semantic_segmentation_3d(sensor_frame=sensor_frame, scene_name=scene_name)
                    elif a_type is AnnotationTypes.InstanceSegmentation3D:
                        a_value = self._save_instance_segmentation_3d(sensor_frame=sensor_frame, scene_name=scene_name)
                    elif a_type is AnnotationTypes.Depth:
                        a_value = self._save_depth(sensor_frame=sensor_frame, scene_name=scene_name)
                    else:
                        a_value = "NOT_IMPLEMENTED"

                    annotations[a_key] = a_value

            point_cloud = SceneDataDatumTypePointCloud(
                filename=self._save_point_cloud(sensor_frame=sensor_frame, scene_name=scene_name),
                annotations=annotations,
                point_format=["X", "Y", "Z", "INTENSITY", "R", "G", "B", "RING", "TIMESTAMP"],
                pose=PoseDTO(
                    translation=TranslationDTO(
                        x=sensor_frame.pose.translation[0],
                        y=sensor_frame.pose.translation[1],
                        z=sensor_frame.pose.translation[2],
                    ),
                    rotation=RotationDTO(
                        qw=sensor_frame.pose.quaternion.w,
                        qx=sensor_frame.pose.quaternion.x,
                        qy=sensor_frame.pose.quaternion.y,
                        qz=sensor_frame.pose.quaternion.z,
                    ),
                ),
                point_fields=[],
                metadata={},
            )
            scene_datum_dto = SceneDataDatumPointCloud(point_cloud=point_cloud)
        elif sensor_frame.image is not None:
            annotations = {}
            for a_type in self.annotation_types:
                if a_type in sensor_frame.available_annotation_types:
                    a_key = self._annotation_type_map[a_type]
                    if a_type is AnnotationTypes.BoundingBoxes2D:
                        a_value = self._save_bounding_box_2d(sensor_frame=sensor_frame, scene_name=scene_name)
                    elif a_type is AnnotationTypes.BoundingBoxes3D:
                        a_value = self._save_bounding_box_3d(sensor_frame=sensor_frame, scene_name=scene_name)
                    elif a_type is AnnotationTypes.SemanticSegmentation2D:
                        a_value = self._save_semantic_segmentation_2d(sensor_frame=sensor_frame, scene_name=scene_name)
                    elif a_type is AnnotationTypes.InstanceSegmentation2D:
                        a_value = self._save_instance_segmentation_2d(sensor_frame=sensor_frame, scene_name=scene_name)
                    elif a_type is AnnotationTypes.OpticalFlow:
                        a_value = self._save_motion_vectors_2d(sensor_frame=sensor_frame, scene_name=scene_name)
                    elif a_type is AnnotationTypes.Depth:
                        a_value = self._save_depth(sensor_frame=sensor_frame, scene_name=scene_name)
                    else:
                        a_value = "NOT_IMPLEMENTED"

                    annotations[a_key] = a_value

            image = SceneDataDatumTypeImage(
                filename=self._save_rgb(sensor_frame=sensor_frame, scene_name=scene_name),
                height=sensor_frame.image.height,
                width=sensor_frame.image.width,
                channels=sensor_frame.image.rgba.shape[2],
                annotations=annotations,
                pose=PoseDTO(
                    translation=TranslationDTO(
                        x=sensor_frame.pose.translation[0],
                        y=sensor_frame.pose.translation[1],
                        z=sensor_frame.pose.translation[2],
                    ),
                    rotation=RotationDTO(
                        qw=sensor_frame.pose.quaternion.w,
                        qx=sensor_frame.pose.quaternion.x,
                        qy=sensor_frame.pose.quaternion.y,
                        qz=sensor_frame.pose.quaternion.z,
                    ),
                ),
                metadata={},
            )

            scene_datum_dto = SceneDataDatumImage(image=image)
        else:
            scene_datum_dto = SceneDataDatum()

        scene_data_dto = SceneDataDTO(
            id=SceneDataIdDTO(
                log="",
                name=sensor_frame.sensor_name,
                timestamp=sensor_frame.date_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                index=str(sensor_frame.frame_id),
            ),
            key=data_key,
            datum=scene_datum_dto,
            next_key="",
            prev_key="",
        )

        return scene_data_dto

    def _save_bounding_box_3d(self, sensor_frame: SensorFrame, scene_name: str) -> str:
        bb3d_dto = AnnotationsBoundingBox3DDTO(annotations=[])
        boxes = sensor_frame.get_annotations(AnnotationTypes.BoundingBoxes3D).boxes
        for b in boxes:
            try:
                occlusion = b.attributes["occlusion"]
                del b.attributes["occlusion"]
            except KeyError:
                occlusion = 0

            try:
                truncation = b.attributes["truncation"]
                del b.attributes["truncation"]
            except KeyError:
                truncation = 0

            """ Post-processed calculation of num_points - makes execution slow!
            num_points=0
            if sensor_frame.point_cloud is None
            else num_points_in_box(sensor_frame.point_cloud.xyz_one, b),
            """

            box_dto = BoundingBox3DDTO(
                class_id=b.class_id,
                instance_id=b.instance_id,
                num_points=b.num_points,
                attributes={_attribute_key_dump(k): _attribute_value_dump(v) for k, v in b.attributes.items()},
                box=BoundingBox3DBoxDTO(
                    width=b.width,
                    length=b.length,
                    height=b.height,
                    occlusion=occlusion,
                    truncation=truncation,
                    pose=PoseDTO(
                        translation=TranslationDTO(
                            x=b.pose.translation[0], y=b.pose.translation[1], z=b.pose.translation[2]
                        ),
                        rotation=RotationDTO(
                            qw=b.pose.quaternion.w,
                            qx=b.pose.quaternion.x,
                            qy=b.pose.quaternion.y,
                            qz=b.pose.quaternion.z,
                        ),
                    ),
                ),
            )

            bb3d_dto.annotations.append(box_dto)

        relative_path = Path("bounding_box_3d") / sensor_frame.sensor_name
        filename = f"{int(sensor_frame.frame_id):018d}.json"
        output_path = self._dataset_path / scene_name / relative_path / filename

        return str(relative_path / _json_write(bb3d_dto.to_dict(), output_path, append_sha1=True))

    def _save_bounding_box_2d(self, sensor_frame: SensorFrame, scene_name: str) -> str:
        bb2d_dto = AnnotationsBoundingBox2DDTO(annotations=[])
        boxes = sensor_frame.get_annotations(AnnotationTypes.BoundingBoxes2D).boxes
        for b in boxes:
            try:
                is_crowd = b.attributes["iscrowd"]
                del b.attributes["iscrowd"]
            except KeyError:
                is_crowd = False
            box_dto = BoundingBox2DDTO(
                class_id=b.class_id,
                instance_id=b.instance_id,
                area=b.area,
                iscrowd=is_crowd,
                attributes={_attribute_key_dump(k): _attribute_value_dump(v) for k, v in b.attributes.items()},
                box=BoundingBox2DBoxDTO(x=b.x, y=b.y, w=b.width, h=b.height),
            )

            bb2d_dto.annotations.append(box_dto)

        relative_path = Path("bounding_box_2d") / sensor_frame.sensor_name
        filename = f"{int(sensor_frame.frame_id):018d}.json"
        output_path = self._dataset_path / scene_name / relative_path / filename

        return str(relative_path / _json_write(bb2d_dto.to_dict(), output_path, append_sha1=True))

    def _save_semantic_segmentation_2d(self, sensor_frame: SensorFrame, scene_name: str) -> str:
        relative_path = Path("semantic_segmentation_2d") / sensor_frame.sensor_name
        filename = f"{int(sensor_frame.frame_id):018d}.png"
        output_path = self._dataset_path / scene_name / relative_path / filename

        return str(
            relative_path
            / _png_write(sensor_frame.get_annotations(AnnotationTypes.SemanticSegmentation2D).rgb_encoded, output_path)
        )

    def _save_semantic_segmentation_3d(self, sensor_frame: SensorFrame, scene_name: str) -> str:
        relative_path = Path("semantic_segmentation_3d") / sensor_frame.sensor_name
        filename = f"{int(sensor_frame.frame_id):018d}.npz"
        output_path = self._dataset_path / scene_name / relative_path / filename

        return str(
            relative_path
            / _npz_write(
                {
                    "segmentation": sensor_frame.get_annotations(
                        AnnotationTypes.SemanticSegmentation3D
                    ).class_ids.astype(np.uint32)
                },
                output_path,
            )
        )

    def _save_instance_segmentation_2d(self, sensor_frame: SensorFrame, scene_name: str) -> str:
        relative_path = Path("instance_segmentation_2d") / sensor_frame.sensor_name
        filename = f"{int(sensor_frame.frame_id):018d}.png"
        output_path = self._dataset_path / scene_name / relative_path / filename

        return str(
            relative_path
            / _png_write(sensor_frame.get_annotations(AnnotationTypes.InstanceSegmentation2D).rgb_encoded, output_path)
        )

    def _save_instance_segmentation_3d(self, sensor_frame: SensorFrame, scene_name: str) -> str:
        relative_path = Path("instance_segmentation_3d") / sensor_frame.sensor_name
        filename = f"{int(sensor_frame.frame_id):018d}.npz"
        output_path = self._dataset_path / scene_name / relative_path / filename

        return str(
            relative_path
            / _npz_write(
                {
                    "instance": sensor_frame.get_annotations(
                        AnnotationTypes.InstanceSegmentation3D
                    ).instance_ids.astype(np.uint32)
                },
                output_path,
            )
        )

    def _save_motion_vectors_2d(self, sensor_frame: SensorFrame, scene_name: str) -> str:
        relative_path = Path("motion_vectors_2d") / sensor_frame.sensor_name
        filename = f"{int(sensor_frame.frame_id):018d}.png"
        output_path = self._dataset_path / scene_name / relative_path / filename

        return str(
            relative_path
            / _png_write(
                _vectors_to_rgba(sensor_frame.get_annotations(AnnotationTypes.OpticalFlow).vectors),
                output_path,
            )
        )

    def _save_depth(self, sensor_frame: SensorFrame, scene_name: str) -> str:
        relative_path = Path("depth") / sensor_frame.sensor_name
        filename = f"{int(sensor_frame.frame_id):018d}.npz"
        output_path = self._dataset_path / scene_name / relative_path / filename

        return str(
            relative_path
            / _npz_write(
                {"data": sensor_frame.get_annotations(AnnotationTypes.Depth).depth[:, :, 0]},
                output_path,
            )
        )

    def _save_rgb(self, sensor_frame: SensorFrame, scene_name: str) -> str:
        relative_path = Path("rgb") / sensor_frame.sensor_name
        filename = f"{int(sensor_frame.frame_id):018d}.png"
        output_path = self._dataset_path / scene_name / relative_path / filename

        return str(relative_path / _png_write(sensor_frame.image.rgba, output_path))

    def _save_point_cloud(self, sensor_frame: SensorFrame, scene_name: str) -> str:
        relative_path = Path("point_cloud") / sensor_frame.sensor_name
        filename = f"{int(sensor_frame.frame_id):018d}.npz"
        output_path = self._dataset_path / scene_name / relative_path / filename

        pc = sensor_frame.point_cloud
        pc_dtypes = [
            ("X", "<f4"),
            ("Y", "<f4"),
            ("Z", "<f4"),
            ("INTENSITY", "<f4"),
            ("R", "<f4"),
            ("G", "<f4"),
            ("B", "<f4"),
            ("RING_ID", "<u4"),
            ("TIMESTAMP", "<u8"),
        ]

        row_count = pc.length
        pc_data = np.empty(row_count, dtype=pc_dtypes)

        pc_data["X"] = pc.xyz[:, 0]
        pc_data["Y"] = pc.xyz[:, 1]
        pc_data["Z"] = pc.xyz[:, 2]
        pc_data["INTENSITY"] = pc.intensity[:, 0]
        pc_data["R"] = pc.rgb[:, 0]
        pc_data["G"] = pc.rgb[:, 1]
        pc_data["B"] = pc.rgb[:, 2]
        pc_data["RING_ID"] = pc.ring[:, 0]
        pc_data["TIMESTAMP"] = pc.ts[:, 0]

        return str(relative_path / _npz_write({"data": pc_data}, output_path))

    def _save_calibration_json(self, sensor_frames: List[SensorFrame], scene_name: str) -> str:
        calib_dto = CalibrationDTO(names=[], extrinsics=[], intrinsics=[])

        for sf in sensor_frames:
            intr = sf.intrinsic
            extr = sf.extrinsic
            calib_dto.names.append(sf.sensor_name)
            calib_dto.extrinsics.append(
                CalibrationExtrinsicDTO(
                    translation=TranslationDTO(x=extr.translation[0], y=extr.translation[1], z=extr.translation[2]),
                    rotation=RotationDTO(
                        qw=extr.quaternion.w, qx=extr.quaternion.x, qy=extr.quaternion.y, qz=extr.quaternion.z
                    ),
                )
            )
            calib_dto.intrinsics.append(
                CalibrationIntrinsicDTO(
                    fx=intr.fx,
                    fy=intr.fy,
                    cx=intr.cx,
                    cy=intr.cy,
                    skew=intr.skew,
                    fov=intr.fov,
                    k1=intr.k1,
                    k2=intr.k2,
                    k3=intr.k3,
                    k4=intr.k4,
                    k5=intr.k5,
                    k6=intr.k6,
                    p1=intr.p1,
                    p2=intr.p2,
                    fisheye=self._fisheye_camera_model_map[intr.camera_model],
                )
            )

        calibration_json_path = self._dataset_path / scene_name / "calibration" / ".json"

        return _json_write(calib_dto.to_dict(), calibration_json_path, append_sha1=True).split(".")[0]

    def _save_dataset_json(self):

        metadata_dto = DatasetMetaDTO(**self.dataset.meta_data.custom_attributes)
        metadata_dto.available_annotation_types = [
            int(self._annotation_type_map[a_type]) for a_type in self.annotation_types
        ]

        ds_dto = DatasetDTO(
            metadata=metadata_dto,  # needs refinement, currently assumes DGP->DGP
            scene_splits={str(i): DatasetSceneSplitDTO(filenames=[s]) for i, s in enumerate(self._scene_paths)},
        )

        dataset_json_path = self._dataset_path / "scene_dataset.json"
        _json_write(ds_dto.to_dict(), dataset_json_path)

    def _save_scene_json(
        self,
        scene: Scene,
        scene_samples: List[SceneSampleDTO],
        scene_data: List[SceneDataDTO],
        ontologies: Dict[str, str],
    ) -> str:
        scene_dto = SceneDTO(
            name=scene.name,
            description=scene.description,
            log="",
            ontologies=ontologies,
            metadata=SceneMetadataDTO.from_dict(scene.metadata),
            samples=scene_samples,
            data=scene_data,
        )

        relative_path = Path(scene.name)
        filename = "scene.json"
        output_path = self._dataset_path / relative_path / filename

        filename = _json_write(scene_dto.to_dict(), output_path, append_sha1=True)

        return str(relative_path / filename)


def main(dataset_input_path, dataset_output_path, frame_slice):
    decoder = DGPDecoder(dataset_path=dataset_input_path)
    dataset = Dataset.from_decoder(decoder=decoder)

    with DGPEncoder(dataset=dataset, output_path=dataset_output_path, frame_slice=frame_slice) as encoder:
        with dataset.get_editable_scene(scene_name=dataset.scene_names[0]) as scene:
            """
            for sn in ["lidar_bl", "lidar_br", "lidar_fl", "Right", "Left", "Rear"]:
                scene.remove_sensor(sn)

            custom_map = ClassMap.from_id_label_dict({1337: "All"})
            custom_id_map = ClassIdMap(class_id_to_class_id={i: 1337 for i in range(256)})

            frame_ids = scene.frame_ids
            for fid in frame_ids:
                sf = scene.get_sensor("Front").get_frame(fid)
                semseg2d = sf.get_annotations(AnnotationTypes.SemanticSegmentation2D)
                semseg2d.update_classes(custom_id_map, custom_map)
            """

            encoder.encode_scene(scene)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DGP to DGP")
    parser.add_argument("-i", "--input", help="<Required> pass input local / s3 path for DGP dataset", required=True)
    parser.add_argument("-o", "--output", help="<Required> pass output local / s3 path for DGP dataset", required=True)
    parser.add_argument("--start", help="Frame Slicing Start", default=None, type=int)
    parser.add_argument("--stop", help="Frame Slicing Stop", default=None, type=int)
    parser.add_argument("--step", help="Frame Slicing Step", default=None, type=int)

    args = parser.parse_args()

    main(args.input, args.output, slice(args.start, args.stop, args.step))
