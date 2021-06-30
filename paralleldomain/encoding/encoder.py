import abc

from paralleldomain import Scene
from paralleldomain.model.dataset import Dataset


class Encoder(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def encode_dataset(self, dataset: Dataset):
        pass

    @abc.abstractmethod
    def encode_scene(self, scene: Scene):
        pass
