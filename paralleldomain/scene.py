from .dto import SceneDTO
from typing import Dict


class Scene:
    def __init__(self, scene_dto: SceneDTO):
        self._dto = scene_dto

    @property
    def name(self):
        return self._dto.name

    @property
    def description(self):
        return self._dto.description

    @staticmethod
    def from_dict(scene_data: Dict):
        scene = Scene(SceneDTO.from_dict(scene_data))
        return scene
