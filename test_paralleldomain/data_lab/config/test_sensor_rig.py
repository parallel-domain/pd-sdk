import logging

import pytest
from pd.core import PdError

from paralleldomain.data_lab import SensorRig
from paralleldomain.data_lab.config.sensor_rig import DistortionParams
from paralleldomain.utilities.transformation import Transformation


class ListHandler(logging.Handler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []

    def emit(self, record):
        self.records.append(record)


class TestSensorResolutionBounds:
    def test_exceeding_resolution_one_dimension(self):
        with pytest.raises(PdError) as exc:
            SensorRig().add_camera(
                name="Front",
                width=4201,
                height=1080,
                field_of_view_degrees=90,
                pose=Transformation(),
            )

        assert all([x in str(exc.value) for x in ("Front", "resolution")])

    def test_exceeding_resolution_two_dimensions(self):
        with pytest.raises(PdError) as exc:
            SensorRig().add_camera(
                name="Front",
                width=4201,
                height=4201,
                field_of_view_degrees=90,
                pose=Transformation(),
            )

        assert all([x in str(exc.value) for x in ("Front", "resolution")])

    def test_exceeding_resolution_two_cameras(self):
        with pytest.raises(PdError) as exc:
            SensorRig().add_camera(
                name="Front",
                width=1920,
                height=1080,
                field_of_view_degrees=90,
                pose=Transformation(),
            ).add_camera(
                name="Left",
                width=4201,
                height=4201,
                field_of_view_degrees=90,
                pose=Transformation(),
            )

        assert all([x in str(exc.value) for x in ("Left", "resolution")])

    def test_left_exceeding_resolution(self):
        with pytest.raises(PdError) as exc:
            SensorRig().add_camera(
                name="Front",
                width=1,
                height=1,
                field_of_view_degrees=90,
                pose=Transformation(),
            )

        assert all([x in str(exc.value) for x in ("Front", "resolution")])

    def test_negative_resolution(self):
        with pytest.raises(PdError) as exc:
            SensorRig().add_camera(
                name="Front",
                width=-1,
                height=-1,
                field_of_view_degrees=90,
                pose=Transformation(),
            )

        assert all([x in str(exc.value) for x in ("Front", "resolution")])

    def test_valid_resolution(self):
        SensorRig().add_camera(
            name="Front",
            width=1920,
            height=1080,
            field_of_view_degrees=90,
            pose=Transformation(),
        )

    def test_valid_max_resolution(self):
        SensorRig().add_camera(
            name="Front",
            width=4200,
            height=4200,
            field_of_view_degrees=90,
            pose=Transformation(),
        )

    def test_valid_min_resolution(self):
        SensorRig().add_camera(
            name="Front",
            width=2,
            height=2,
            field_of_view_degrees=90,
            pose=Transformation(),
        )


class TestSensorRigResolutionBounds:
    def test_exceeding_combined_resolution_pinhole(self):
        with pytest.raises(PdError) as exc:
            SensorRig().add_camera(
                name="Front",
                width=4200,
                height=4200,
                field_of_view_degrees=90,
                pose=Transformation(),
            ).add_camera(
                name="Left",
                width=4200,
                height=4200,
                field_of_view_degrees=90,
                pose=Transformation(),
            ).add_camera(
                name="Right",
                width=4200,
                height=4200,
                field_of_view_degrees=90,
                pose=Transformation(),
            ).add_camera(
                name="Rear",
                width=4200,
                height=4200,
                field_of_view_degrees=90,
                pose=Transformation(),
            )

        assert all([x in str(exc.value) for x in ("combined", "resolution")])

    def test_exceeding_combined_resolution_fisheye(self):
        with pytest.raises(PdError) as exc:
            SensorRig().add_camera(
                name="Front",
                width=4200,
                height=4200,
                field_of_view_degrees=90,
                pose=Transformation(),
                distortion_params=DistortionParams(fisheye_model=3),
            ).add_camera(
                name="Left",
                width=4200,
                height=4200,
                field_of_view_degrees=90,
                pose=Transformation(),
                distortion_params=DistortionParams(fisheye_model=3),
            )

        assert all([x in str(exc.value) for x in ("combined", "resolution")])

    def test_exceeding_combined_resolution_pinhole_supersampling(self):
        with pytest.raises(PdError) as exc:
            SensorRig().add_camera(
                name="Front",
                width=4200,
                height=4200,
                field_of_view_degrees=90,
                pose=Transformation(),
                supersample=3.7,
            )

        assert all([x in str(exc.value) for x in ("combined", "resolution")])

    def test_exceeding_combined_resolution_fisheye_supersampling(self):
        with pytest.raises(PdError) as exc:
            SensorRig().add_camera(
                name="Front",
                width=4200,
                height=4200,
                field_of_view_degrees=90,
                pose=Transformation(),
                distortion_params=DistortionParams(fisheye_model=3),
                supersample=1.3,
            )

        assert all([x in str(exc.value) for x in ("combined", "resolution")])

    def test_valid_combined_resolution_pinhole_supersampling(self):
        SensorRig().add_camera(
            name="Front",
            width=4200,
            height=4200,
            field_of_view_degrees=90,
            pose=Transformation(),
            supersample=3.61,
        )

    def test_valid_combined_resolution_fisheye_supersampling(self):
        SensorRig().add_camera(
            name="Front",
            width=4200,
            height=4200,
            field_of_view_degrees=90,
            pose=Transformation(),
            distortion_params=DistortionParams(fisheye_model=3),
            supersample=1.2,
        )


class TestSensorOrthographic:
    def test_near_clip_plane_clip(self):
        logger = logging.getLogger()
        handler = ListHandler()
        logger.addHandler(handler)

        SensorRig().add_camera(
            name="Front",
            width=10,
            height=10,
            pose=Transformation(),
            distortion_params=DistortionParams(
                fisheye_model=6,
                fx=10,
                fy=10,
                p1=-301,
                p2=500,
            ),
        )

        assert any(
            record.levelno == logging.WARNING and all([x in record.message for x in ("Clipping", "near", "Front")])
            for record in handler.records
        )

    def test_far_clip_plane_clip(self):
        logger = logging.getLogger()
        handler = ListHandler()
        logger.addHandler(handler)

        SensorRig().add_camera(
            name="Front",
            width=10,
            height=10,
            pose=Transformation(),
            distortion_params=DistortionParams(
                fisheye_model=6,
                fx=10,
                fy=10,
                p1=-300,
                p2=501,
            ),
        )

        assert any(
            record.levelno == logging.WARNING and all([x in record.message for x in ("Clipping", "far", "Front")])
            for record in handler.records
        )

    def test_near_larger_than_far_plane(self):
        with pytest.raises(PdError) as exc:
            SensorRig().add_camera(
                name="Front",
                width=10,
                height=10,
                pose=Transformation(),
                distortion_params=DistortionParams(
                    fisheye_model=6,
                    fx=10,
                    fy=10,
                    p1=301,
                    p2=300,
                ),
            )

        assert all([x in str(exc.value) for x in ("Front", "further")])

    def test_near_larger_than_far_plane_after_clipping(self):
        logger = logging.getLogger()
        handler = ListHandler()
        logger.addHandler(handler)

        with pytest.raises(PdError) as exc:
            SensorRig().add_camera(
                name="Front",
                width=10,
                height=10,
                pose=Transformation(),
                distortion_params=DistortionParams(
                    fisheye_model=6,
                    fx=10,
                    fy=10,
                    p1=-700,  # going to be -300 after clipping and violating the near <= far rule
                    p2=-600,
                ),
            )

        assert any(
            record.levelno == logging.WARNING and all([x in record.message for x in ("Clipping", "near", "Front")])
            for record in handler.records
        )

        assert all([x in str(exc.value) for x in ("Front", "further")])
