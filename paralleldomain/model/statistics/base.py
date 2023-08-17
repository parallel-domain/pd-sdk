import logging
import importlib
from typing import Dict, List, Union
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

    def parse_sensor_frame_properties(self, scene: Scene, sensor_frame: SensorFrame) -> Dict:
        return dict(scene_name=scene.name, sensor_name=sensor_frame.sensor_name, frame_id=sensor_frame.frame_id)

    def update(self, scene: Scene, sensor_frame: SensorFrame):
        self._update(scene=scene, sensor_frame=sensor_frame)
        self.notify_subscribers()

    @abstractmethod
    def _update(self, scene: Scene, sensor_frame: SensorFrame):
        """
        Args:
            scene (Scene): scene
            sensor_frame (SensorFrame): sensor_frame

        """
        pass

    def load(self, path: Union[str, AnyPath]):
        from filelock import FileLock

        path = AnyPath(path)
        file_path = path / f"{self.__class__.__name__}.pkl"
        lock_path = path / f"{self.__class__.__name__}.pkl.lock"
        if lock_path.is_cloud_path:
            self._load(file_path=file_path)
        else:
            lock = FileLock(str(lock_path), timeout=60)
            with lock:
                self._load(file_path=file_path)
        self.notify_subscribers()

    @abstractmethod
    def _load(self, file_path: Union[str, AnyPath]):
        pass

    def save(self, path: Union[str, AnyPath]):
        path = AnyPath(path)
        from filelock import FileLock

        file_path = path / f"{self.__class__.__name__}.pkl"
        lock_path = path / f"{self.__class__.__name__}.pkl.lock"
        lock_path.parent.mkdir(exist_ok=True, parents=True)
        lock = FileLock(str(lock_path), timeout=60)

        with lock:
            self._save(file_path=str(file_path))

    @abstractmethod
    def _save(self, file_path: Union[str, AnyPath]):
        pass


class CompositeStatistic(Statistic):
    def __init__(self, sub_models: List[Statistic]):
        super().__init__()
        self._sub_models = sub_models

    @property
    def child_statistics(self) -> List[Statistic]:
        return self._sub_models

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

    def load(self, path: Union[str, AnyPath]):
        path = AnyPath(path)
        for sub_model in self._sub_models:
            sub_model.load(path=path)

    def _load(self, file_path: Union[str, AnyPath]):
        file_path = AnyPath(file_path)
        for sub_model in self._sub_models:
            sub_model.load(path=file_path)

    def save(self, path: Union[str, AnyPath]):
        path = AnyPath(path)
        for sub_model in self._sub_models:
            sub_model.save(path=path)

    def _save(self, path: Union[str, AnyPath]):
        path = AnyPath(path)
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
    def from_path(path: Union[str, AnyPath], watch_changes: bool = False) -> "CompositeStatistic":
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


StatisticAliases = Union[Statistic, List[Statistic], str, List[str]]


def resolve_statistics(statistics: StatisticAliases) -> Statistic:
    if isinstance(statistics, Statistic):
        return statistics
    elif isinstance(statistics, str):
        statistics = statistics.strip()
        if "," in statistics:
            return resolve_statistics(statistics=statistics.split(","))
        elif statistics == "all":
            return CompositeStatistic(
                sub_models=[resolve_statistics(statistics=name) for name, _ in STATISTICS_REGISTRY.items()]
            )
        elif statistics in STATISTICS_REGISTRY.keys():
            return STATISTICS_REGISTRY[statistics].module_class()
        else:
            for name, entry in STATISTICS_REGISTRY.items():
                if statistics == entry.tags.get("name", None):
                    return entry.module_class()
    elif isinstance(statistics, list):
        stats = list()
        for stat in statistics:
            stats.append(resolve_statistics(statistics=stat))
        return CompositeStatistic(sub_models=stats)
