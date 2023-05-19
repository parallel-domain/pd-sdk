from typing import List, Union

import pd.state

from paralleldomain.model.annotation import AnnotationType, AnnotationTypes


def get_sensor_rig_annotation_types(
    sensor_rig: List[Union[pd.state.CameraSensor, pd.state.LiDARSensor]]
) -> List[AnnotationType]:
    annotation_types = set()
    for sensor in sensor_rig:
        annotation_types.update(get_annotation_types(sensor=sensor))
    return list(annotation_types)


def get_annotation_types(sensor: Union[pd.state.CameraSensor, pd.state.LiDARSensor]) -> List[AnnotationType]:
    anno_types = list()
    if sensor.capture_segmentation:
        anno_types.append(AnnotationTypes.SemanticSegmentation2D)
    if sensor.capture_depth:
        anno_types.append(AnnotationTypes.Depth)
    if sensor.capture_instances:
        anno_types.append(AnnotationTypes.InstanceSegmentation2D)
    if sensor.capture_normals:
        anno_types.append(AnnotationTypes.SurfaceNormals2D)
    if isinstance(sensor, pd.state.CameraSensor) and sensor.capture_properties:
        anno_types.append(AnnotationTypes.MaterialProperties2D)
    if isinstance(sensor, pd.state.CameraSensor) and sensor.capture_basecolor:
        anno_types.append(AnnotationTypes.Albedo2D)
    if sensor.capture_motionvectors:
        anno_types.append(AnnotationTypes.OpticalFlow)
    if isinstance(sensor, pd.state.CameraSensor) and sensor.capture_backwardmotionvectors:
        anno_types.append(AnnotationTypes.BackwardOpticalFlow)
    return anno_types
