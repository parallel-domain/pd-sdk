from typing import List, Tuple, Type, TypeVar, Generic
from enum import Enum
from abc import ABC, abstractmethod
import logging


import webbrowser

from paralleldomain.model.statistics.base import Statistic, CompositeStatistic
from paralleldomain.utilities.observable import Observer
from paralleldomain.utilities.module_registry import ModuleRegistry


STATISTIC_VIS_REGISTRY = ModuleRegistry("reference_model", "is_default", "backend")

S = TypeVar("S", bound=Statistic)


class BACKEND(Enum):
    DASH = "dash"
    JUPYTER_DASH = "jupyter_dash"
    RERUN = "rerun"


class ViewComponent(Observer, Generic[S]):
    def __init__(self, model: S):
        self._model: S = model
        self._model.add_subscriber(self)
        self._vis_need_update: bool = True

    def visualize(self):
        if self._vis_need_update:
            self._fig = self._visualize()

        return self._fig

    def notify(self):
        self._vis_need_update = True

    @abstractmethod
    def _visualize(self):
        pass

    @property
    @abstractmethod
    def title(self):
        pass


class StatisticViewer(ABC):
    def __init__(self, view_components: List[ViewComponent]):
        self._view_components = view_components

    @classmethod
    @abstractmethod
    def backend(cls) -> BACKEND:
        pass

    @abstractmethod
    def to_html(self, filename: str = "dashboard.html") -> None:
        pass

    def show_html(self) -> None:
        self.to_html(filename="dashboard.html")
        webbrowser.open("dashboard.html")

    @abstractmethod
    def show(self) -> None:
        pass

    @staticmethod
    def get_default_view_component(statistic: Statistic, backend: BACKEND) -> Type[ViewComponent]:
        for _, entry in STATISTIC_VIS_REGISTRY.items():
            if (
                isinstance(statistic, entry.tags.get("reference_model", None))
                and entry.tags.get("is_default", False)
                and entry.tags.get("backend", None) == backend
            ):
                return entry.module_class

    @staticmethod
    def get_supported_statistics(backend: BACKEND) -> List[Type[Statistic]]:
        supported_statistics = []
        for _, entry in STATISTIC_VIS_REGISTRY.items():
            if entry.tags.get("backend", None) == backend:
                supported_statistics.append(entry.tags["reference_model"])
        return supported_statistics

    @classmethod
    def create_view_components(cls, models: List[Statistic], **kwargs):
        view_components = []

        for sub_model in models:
            view_class = cls.get_default_view_component(statistic=sub_model, backend=cls.backend())

            if view_class is None:
                logging.warning(
                    f"Could not find suitable view component for {sub_model.__class__.__name__} "
                    f"with backend {cls.backend()}."
                )
                continue

            applicable_kwargs = {
                key: value for key, value in kwargs.items() if key in view_class.__init__.__code__.co_varnames
            }

            view_components.append(view_class(model=sub_model, **applicable_kwargs))

        return view_components

    @classmethod
    def create_with_default_components(cls, **kwargs) -> Tuple["StatisticViewer", CompositeStatistic]:
        sub_models = []
        for statistic_class in cls.get_supported_statistics(backend=cls.backend()):
            applicable_kwargs = {
                key: value for key, value in kwargs.items() if key in statistic_class.__init__.__code__.co_varnames
            }
            sub_models.append(statistic_class(**applicable_kwargs))

        view_components = cls.create_view_components(models=sub_models, **kwargs)

        viewer = cls(view_components=view_components)
        return viewer, CompositeStatistic(sub_models=sub_models)

    @classmethod
    def create_from_filepath(
        cls, path: str, watch_changes: bool = False
    ) -> Tuple["StatisticViewer", CompositeStatistic]:
        view_components = []
        model = CompositeStatistic.from_path(path=path, watch_changes=watch_changes)

        view_components = cls.create_view_components(model._sub_models)

        return cls(view_components=view_components), model

    @staticmethod
    def resolve_default_viewers(statistic: Statistic, backend: BACKEND) -> List["StatisticViewer"]:
        visualizers = list()
        if isinstance(statistic, CompositeStatistic):
            for stat in statistic.child_statistics:
                viewer = StatisticViewer.get_default_view_component(statistic=stat, backend=backend)(model=stat)
                visualizers.append(viewer)
        elif isinstance(statistic, Statistic):
            viewer = StatisticViewer.get_default_view_component(statistic=statistic, backend=backend)(model=statistic)
            visualizers.append(viewer)
        else:
            raise ValueError(f"Unsupported type: {type(statistic)}!")
        return visualizers
