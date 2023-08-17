from typing import Dict, List, Type, TypeVar, Any

from dataclasses import dataclass

T = TypeVar("T")


@dataclass
class ModuleRegistryEntry:
    module_class: Type[T]
    tags: Dict[str, Any]


class ModuleRegistry:
    def __init__(self, *required_params: str):
        self._modules: Dict[str, ModuleRegistryEntry] = {}
        self._required_params: List[str] = list(required_params)

    def keys(self):
        return self._modules.keys()

    def items(self):
        return self._modules.items()

    def __getitem__(self, item):
        return self._modules[item]

    def register_module(self, **kwargs: Any):
        def decorator(module: T):
            module_name = module.__name__
            if module_name in self._modules:
                raise ValueError(f"Cannot register module {module_name}, already exists.")
            self._modules[module.__name__] = ModuleRegistryEntry(module_class=module, tags={})
            entry = self._modules[module_name]
            entry.tags.update(kwargs)
            for param in self._required_params:
                if param not in entry.tags:
                    raise ValueError(f"Mandatory parameter '{param}' is missing for module '{module_name}'.")

            return module

        return decorator
