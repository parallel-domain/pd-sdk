import os
import logging
import importlib
from typing import List
from abc import abstractmethod

from paralleldomain.model.scene import Scene
from paralleldomain.model.sensor import SensorFrame
from paralleldomain.utilities.observable import Observable
from paralleldomain.utilities.any_path import AnyPath
from paralleldomain.model.statistics.constants import STATISTICS_REGISTRY

logger = logging.getLogger(__name__)


def validate_optional_imports():
    required_modules = ["watchdog", "filelock", "pandas"]
    for module_name in required_modules:
        if importlib.util.find_spec(module_name) is None:
            return False
    return True


class Statistic(Observable):
    def __init__(self):
        if not validate_optional_imports():
            raise ImportError("To use Statistic module, please install extra dependency 'statistics'.")

        super().__init__()

    @abstractmethod
    def _reset(self):
        pass

    def reset(self):
        self._reset()
        self.notify_subscribers()

    def update(self, scene: Scene, sensor_frame: SensorFrame):
        self._update(scene=scene, sensor_frame=sensor_frame)
        self.notify_subscribers()

    @abstractmethod
    def _update(self, scene: Scene, sensor_frame: SensorFrame):
        pass

    def load(self, path: str):
        from filelock import FileLock

        file_path = os.path.join(path, f"{self.__class__.__name__}.pkl")
        lock_path = f"{file_path}.lock"
        lock = FileLock(lock_path, timeout=60)
        with lock:
            self._load(file_path=file_path)
        self.notify_subscribers()

    @abstractmethod
    def _load(self, file_path: str):
        pass

    def save(self, path: str):
        from filelock import FileLock

        file_path = os.path.join(path, f"{self.__class__.__name__}.pkl")
        lock_path = f"{file_path}.lock"
        lock = FileLock(lock_path, timeout=60)
        with lock:
            self._save(file_path=file_path)
        self._save(file_path=file_path)

    @abstractmethod
    def _save(self, file_path: str):
        pass


class CompositeStatistic(Statistic):
    def __init__(self, sub_models: List[Statistic]):
        super().__init__()
        self._sub_models = sub_models

    def append_statistic(self, model: Statistic):
        self._sub_models.append(model)

    def _reset(self):
        for sub_model in self._sub_models:
            sub_model.reset()

    def _update(self, scene: Scene, sensor_frame: SensorFrame):
        for sub_model in self._sub_models:
            sub_model.update(scene=scene, sensor_frame=sensor_frame)

    def notify_subscribers(self):
        for subscriber in self._subscribers:
            subscriber.notify()
        for sub_model in self._sub_models:
            for subscriber in sub_model._subscribers:
                subscriber.notify()

    def load(self, path: str):
        for sub_model in self._sub_models:
            sub_model.load(path=path)

    def _load(self, file_path: str):
        for sub_model in self._sub_models:
            sub_model.load(path=file_path)

    def save(self, path: str):
        for sub_model in self._sub_models:
            sub_model.save(path=path)

    def _save(self, path: str):
        self.save(path)

    def watch(self, path: str):
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler

        class FileSystemWatcher(FileSystemEventHandler):
            def __init__(self, statistic: CompositeStatistic):
                super().__init__()
                self.statistic = statistic

            def on_modified(self, event):
                if event.is_directory:
                    return

                filepath = AnyPath(event.src_path)
                filename = filepath.name
                extension = filepath.suffix
                for model in self.statistic._sub_models:
                    if model.__class__.__name__ == filename[: -len(extension)]:
                        model.load(str(filepath.parent))

        event_handler = FileSystemWatcher(statistic=self)
        observer = Observer()
        observer.schedule(event_handler, path, recursive=False)
        observer.start()

    @staticmethod
    def from_path(path: str, watch_changes: bool = False) -> "CompositeStatistic":
        path = AnyPath(path)
        filepaths = path.glob("*.pkl")

        sub_models = []
        for filepath in filepaths:
            # filename = filepath.name
            extension = filepath.suffix

            if extension == ".pkl":
                for name, entry in STATISTICS_REGISTRY.items():
                    if name == filepath.stem:
                        sub_model = entry.module_class()
                        sub_model.load(str(path))
                        sub_models.append(sub_model)

        model = CompositeStatistic(sub_models=sub_models)
        if watch_changes:
            model.watch(path=str(path))
        return model
