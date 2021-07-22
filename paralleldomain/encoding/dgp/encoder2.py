import argparse

import numpy as np

from paralleldomain.encoding.encoder import DatasetEncoder, SceneEncoder
from paralleldomain.encoding.utils import fsio
from paralleldomain.model.sensor import SensorFrame


class DGPSceneEncoder(SceneEncoder):
    def _encode_rgb(self, sensor_frame: SensorFrame):
        output_path = self._output_path / "rgb" / sensor_frame.sensor_name / f"{int(sensor_frame.frame_id):018d}.png"
        self._run_async(func=fsio.write_png, obj=sensor_frame.image.rgba, path=output_path)
        return output_path

    def _encode_point_cloud(self, sensor_frame: SensorFrame):
        output_path = (
            self._output_path / "point_cloud" / sensor_frame.sensor_name / f"{int(sensor_frame.frame_id):018d}.npz"
        )

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

        self._run_async(func=fsio.write_npz, data=pc_data)

        return output_path

    def _encode_camera_frame(self, camera_frame: SensorFrame):
        self._encode_rgb(sensor_frame=camera_frame)

    def _encode_lidar_frame(self, lidar_frame: SensorFrame):
        self._encode_point_cloud(sensor_frame=lidar_frame)

    def _prepare_output_directories(self) -> None:
        super()._prepare_output_directories()
        for camera_name in self._scene.camera_names:
            (self._output_path / "rgb" / camera_name).mkdir(exist_ok=True, parents=True)

        for lidar_name in self._scene.lidar_names:
            (self._output_path / "point_cloud" / lidar_name).mkdir(exist_ok=True, parents=True)


class DGPDatasetEncoder(DatasetEncoder):
    scene_encoder = DGPSceneEncoder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs a data encoders")
    parser.add_argument("-i", "--input", type=str, help="A local or cloud path to a DGP dataset", required=True)
    parser.add_argument("-o", "--output", type=str, help="A local or cloud path for the encoded dataset", required=True)
    parser.add_argument(
        "--scene_names",
        nargs="*",
        type=str,
        help="""Define one or multiple specific scenes to be processed.
                When provided, overwrites any scene_start and scene_stop arguments""",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--scene_start",
        type=int,
        help="An integer defining the start index for scene processing",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--scene_stop",
        type=int,
        help="An integer defining the stop index for scene processing",
        required=False,
        default=None,
    )

    parser.add_argument(
        "--n_parallel",
        type=int,
        help="Define how many scenes should be processed in parallel",
        required=False,
        default=1,
    )

    args = parser.parse_args()

    DGPDatasetEncoder(
        input_path=args.input,
        output_path=args.output,
        scene_names=args.scene_names,
        scene_start=args.scene_start,
        scene_stop=args.scene_stop,
        n_parallel=args.n_parallel,
    ).run()
