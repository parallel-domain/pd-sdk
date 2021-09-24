from paralleldomain.model.class_mapping import ClassDetail
from paralleldomain.model.type_aliases import SceneName
from paralleldomain.utilities.any_path import AnyPath


def get_scene_path(dataset_path: AnyPath, scene_name: SceneName, camera_name: str) -> AnyPath:
    split = scene_name.split("-")[0]
    city_name = scene_name.split("-")[1]
    return dataset_path / camera_name / split / city_name


def get_scene_labels_path(dataset_path: AnyPath, scene_name: SceneName) -> AnyPath:
    split = scene_name.split("-")[0]
    city_name = scene_name.split("-")[1]
    return dataset_path / "gtFine" / split / city_name


CITYSCAPE_CLASSES = [
    ClassDetail(
        name="unlabeled",
        id=0,
        instanced=False,
        meta=dict(ignoreInEval=True, catId=0, trainId=255, category="void", color=(0, 0, 0)),
    ),
    ClassDetail(
        name="ego vehicle",
        id=1,
        instanced=False,
        meta=dict(ignoreInEval=True, catId=0, trainId=255, category="void", color=(0, 0, 0)),
    ),
    ClassDetail(
        name="rectification border",
        id=2,
        instanced=False,
        meta=dict(ignoreInEval=True, catId=0, trainId=255, category="void", color=(0, 0, 0)),
    ),
    ClassDetail(
        name="out of roi",
        id=3,
        instanced=False,
        meta=dict(ignoreInEval=True, catId=0, trainId=255, category="void", color=(0, 0, 0)),
    ),
    ClassDetail(
        name="static",
        id=4,
        instanced=False,
        meta=dict(ignoreInEval=True, catId=0, trainId=255, category="void", color=(0, 0, 0)),
    ),
    ClassDetail(
        name="dynamic",
        id=5,
        instanced=False,
        meta=dict(ignoreInEval=True, catId=0, trainId=255, category="void", color=(111, 74, 0)),
    ),
    ClassDetail(
        name="ground",
        id=6,
        instanced=False,
        meta=dict(ignoreInEval=True, catId=0, trainId=255, category="void", color=(81, 0, 81)),
    ),
    ClassDetail(
        name="road",
        id=7,
        instanced=False,
        meta=dict(ignoreInEval=False, catId=1, trainId=0, category="flat", color=(128, 64, 128)),
    ),
    ClassDetail(
        name="sidewalk",
        id=8,
        instanced=False,
        meta=dict(ignoreInEval=False, catId=1, trainId=1, category="flat", color=(244, 35, 232)),
    ),
    ClassDetail(
        name="parking",
        id=9,
        instanced=False,
        meta=dict(ignoreInEval=True, catId=1, trainId=255, category="flat", color=(250, 170, 160)),
    ),
    ClassDetail(
        name="rail track",
        id=10,
        instanced=False,
        meta=dict(ignoreInEval=True, catId=1, trainId=255, category="flat", color=(230, 150, 140)),
    ),
    ClassDetail(
        name="building",
        id=11,
        instanced=False,
        meta=dict(ignoreInEval=False, catId=2, trainId=2, category="construction", color=(70, 70, 70)),
    ),
    ClassDetail(
        name="wall",
        id=12,
        instanced=False,
        meta=dict(ignoreInEval=False, catId=2, trainId=3, category="construction", color=(102, 102, 156)),
    ),
    ClassDetail(
        name="fence",
        id=13,
        instanced=False,
        meta=dict(ignoreInEval=False, catId=2, trainId=4, category="construction", color=(190, 153, 153)),
    ),
    ClassDetail(
        name="guard rail",
        id=14,
        instanced=False,
        meta=dict(ignoreInEval=True, catId=2, trainId=255, category="construction", color=(180, 165, 180)),
    ),
    ClassDetail(
        name="bridge",
        id=15,
        instanced=False,
        meta=dict(ignoreInEval=True, catId=2, trainId=255, category="construction", color=(150, 100, 100)),
    ),
    ClassDetail(
        name="tunnel",
        id=16,
        instanced=False,
        meta=dict(ignoreInEval=True, catId=2, trainId=255, category="construction", color=(150, 120, 90)),
    ),
    ClassDetail(
        name="pole",
        id=17,
        instanced=False,
        meta=dict(ignoreInEval=False, catId=3, trainId=5, category="object", color=(153, 153, 153)),
    ),
    ClassDetail(
        name="polegroup",
        id=18,
        instanced=False,
        meta=dict(ignoreInEval=True, catId=3, trainId=255, category="object", color=(153, 153, 153)),
    ),
    ClassDetail(
        name="traffic light",
        id=19,
        instanced=False,
        meta=dict(ignoreInEval=False, catId=3, trainId=6, category="object", color=(250, 170, 30)),
    ),
    ClassDetail(
        name="traffic sign",
        id=20,
        instanced=False,
        meta=dict(ignoreInEval=False, catId=3, trainId=7, category="object", color=(220, 220, 0)),
    ),
    ClassDetail(
        name="vegetation",
        id=21,
        instanced=False,
        meta=dict(ignoreInEval=False, catId=4, trainId=8, category="nature", color=(107, 142, 35)),
    ),
    ClassDetail(
        name="terrain",
        id=22,
        instanced=False,
        meta=dict(ignoreInEval=False, catId=4, trainId=9, category="nature", color=(152, 251, 152)),
    ),
    ClassDetail(
        name="sky",
        id=23,
        instanced=False,
        meta=dict(ignoreInEval=False, catId=5, trainId=10, category="sky", color=(70, 130, 180)),
    ),
    ClassDetail(
        name="person",
        id=24,
        instanced=True,
        meta=dict(ignoreInEval=False, catId=6, trainId=11, category="sky", color=(220, 20, 60)),
    ),
    ClassDetail(
        name="rider",
        id=25,
        instanced=True,
        meta=dict(ignoreInEval=False, catId=6, trainId=12, category="sky", color=(255, 0, 0)),
    ),
    ClassDetail(
        name="car",
        id=26,
        instanced=True,
        meta=dict(ignoreInEval=False, catId=7, trainId=13, category="vehicle", color=(0, 0, 142)),
    ),
    ClassDetail(
        name="truck",
        id=27,
        instanced=True,
        meta=dict(ignoreInEval=False, catId=7, trainId=14, category="vehicle", color=(0, 0, 70)),
    ),
    ClassDetail(
        name="bus",
        id=28,
        instanced=True,
        meta=dict(ignoreInEval=False, catId=7, trainId=15, category="vehicle", color=(0, 60, 100)),
    ),
    ClassDetail(
        name="caravan",
        id=29,
        instanced=True,
        meta=dict(ignoreInEval=True, catId=7, trainId=255, category="vehicle", color=(0, 0, 90)),
    ),
    ClassDetail(
        name="trailer",
        id=30,
        instanced=True,
        meta=dict(ignoreInEval=True, catId=7, trainId=255, category="vehicle", color=(0, 0, 110)),
    ),
    ClassDetail(
        name="train",
        id=31,
        instanced=True,
        meta=dict(ignoreInEval=False, catId=7, trainId=16, category="vehicle", color=(0, 80, 100)),
    ),
    ClassDetail(
        name="motorcycle",
        id=32,
        instanced=True,
        meta=dict(ignoreInEval=False, catId=7, trainId=17, category="vehicle", color=(0, 0, 230)),
    ),
    ClassDetail(
        name="bicycle",
        id=33,
        instanced=True,
        meta=dict(ignoreInEval=False, catId=7, trainId=18, category="vehicle", color=(119, 11, 32)),
    ),
    ClassDetail(
        name="license plate",
        id=255,
        instanced=False,
        meta=dict(ignoreInEval=True, catId=7, trainId=-1, category="vehicle", color=(0, 0, 142)),
    ),
]
