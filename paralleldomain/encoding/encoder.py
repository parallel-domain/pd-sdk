import abc

from paralleldomain import Scene


class Encoder(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def encode_dataset(self):
        pass

    @abc.abstractmethod
    def finalize(self):
        pass

    @abc.abstractmethod
    def encode_scene(self, scene: Scene):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finalize()
