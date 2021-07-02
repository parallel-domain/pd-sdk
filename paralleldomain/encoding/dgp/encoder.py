import argparse
import hashlib
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Union

from cloudpathlib import CloudPath
from PIL import Image

from paralleldomain import Dataset, Scene
from paralleldomain.decoding.dgp.decoder import DGPDecoder
from paralleldomain.encoding.dgp.dtos import (
    CalibrationDTO,
    CalibrationExtrinsicDTO,
    CalibrationIntrinsicDTO,
    DatasetDTO,
    DatasetMetaDTO,
    DatasetSceneSplitDTO,
    RotationDTO,
    SceneDTO,
    SceneMetadataDTO,
    TranslationDTO,
)
from paralleldomain.encoding.encoder import Encoder
from paralleldomain.model.class_mapping import ClassIdMap, ClassMap
from paralleldomain.model.sensor import Sensor, SensorFrame
from paralleldomain.utilities.any_path import AnyPath

logger = logging.getLogger(__name__)


def json_write(obj: object, fp: Union[Path, CloudPath, str], append_sha256: bool = False):
    fp = AnyPath(fp)
    fp.parent.mkdir(parents=True, exist_ok=True)

    json_str = json.dumps(obj, indent=2)

    if append_sha256:
        json_str_sha256 = hashlib.sha256(json_str.encode()).hexdigest()
        filename = fp.name.split(".")
        if filename[0] == "":
            filename[0] = json_str_sha256
        else:
            filename[0] = f"{filename[0]}_{json_str_sha256}"

        fp = AnyPath(fp.parent / ".".join(filename))

    with fp.open("w") as json_file:
        json_file.write(json_str)


class DGPEncoder(Encoder):
    _fisheye_camera_model_map = defaultdict(
        lambda: 2,
        {
            "brown_conrady": 0,
            "fisheye": 1,
        },
    )

    def __init__(
        self,
        dataset_path: AnyPath,
        custom_map: Optional[ClassMap] = None,
        custom_id_map: Optional[ClassIdMap] = None,
    ):
        self.custom_map = custom_map
        self.custom_id_map = custom_id_map
        self._dataset_path: Union[Path, CloudPath] = AnyPath(dataset_path)

    def encode_dataset(self, dataset: Dataset):
        scene_names = dataset.scene_names
        for s in scene_names:
            self.encode_scene(dataset.get_scene(s))

        self._save_dataset_json(dataset)

    def encode_scene(self, scene: Scene):
        for f in scene.frames:
            sensor_frames = [f.get_sensor(sn) for sn in f.sensor_names]
            self.encode_sensor_frames(sensor_frames=sensor_frames, scene_name=scene.name)
        self._save_scene_json(scene=scene)

    def encode_sensor_frames(self, sensor_frames: List[SensorFrame], scene_name: str):
        for sf in sensor_frames:
            self._save_rgb(sensor_frame=sf, scene_name=scene_name)

        self._save_calibration_json(sensor_frames=sensor_frames, scene_name=scene_name)

    def _save_rgb(self, sensor_frame: SensorFrame, scene_name: str):
        if sensor_frame.image is not None:
            rgb_image_path = (
                self._dataset_path
                / scene_name
                / "rgb"
                / sensor_frame.sensor_name
                / f"{int(sensor_frame.frame_id):018d}.png"
            )
            rgb_image_path.parent.mkdir(parents=True, exist_ok=True)
            with rgb_image_path.open("wb") as image_file:
                Image.fromarray(sensor_frame.image.rgba).save(image_file, "png")

    def _save_calibration_json(self, sensor_frames: List[SensorFrame], scene_name: str):
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

        json_write(calib_dto.to_dict(), calibration_json_path, append_sha256=True)

    def _save_dataset_json(self, dataset: Dataset):
        ds_dto = DatasetDTO(
            metadata=DatasetMetaDTO(
                **dataset.meta_data.custom_attributes
            ),  # needs refinement, currently assumes DGP->DGP
            scene_splits={
                str(i): DatasetSceneSplitDTO(filenames=[f"{s}/scene.json"]) for i, s in enumerate(dataset.scene_names)
            },
        )

        dataset_json_path = self._dataset_path / "scene_dataset.json"
        json_write(ds_dto.to_dict(), dataset_json_path)

    def _save_scene_json(self, scene: Scene):
        scene_dto = SceneDTO(
            name=scene.name,
            description=scene.description,
            log="",
            ontologies={},
            metadata=SceneMetadataDTO.from_dict(scene.metadata),
            samples=[],
            data=[],
        )  # Todo Scene -> Scene DTO

        scene_json_path = self._dataset_path / scene.name / "scene.json"

        json_write(scene_dto.to_dict(), scene_json_path, append_sha256=True)


def main(dataset_input_path, dataset_output_path):
    decoder = DGPDecoder(dataset_path=dataset_input_path)
    dataset = Dataset.from_decoder(decoder=decoder)

    encoder = DGPEncoder(dataset_path=dataset_output_path)
    encoder.encode_dataset(dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DGP to DGP")
    parser.add_argument("-i", "--input", help="<Required> pass input local / s3 path for DGP dataset", required=True)

    parser.add_argument("-o", "--output", help="<Required> pass output local / s3 path for DGP dataset", required=True)

    parser.add_argument("-m", "--max", const=None, type=int, help="Set the number of max frames to be encoded")

    args = parser.parse_args()

    MAX_FRAMES = args.max

    main(args.input, args.output)
