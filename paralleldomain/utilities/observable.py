from typing import List
from abc import ABC, abstractmethod


class Observer(ABC):
    @abstractmethod
    def notify(self):
        pass


class Observable(ABC):
    def __init__(self):
        self._subscribers: List[Observer] = []

    def notify_subscribers(self):
        for subscriber in self._subscribers:
            subscriber.notify()

    def add_subscriber(self, subscriber: Observer):
        self._subscribers.append(subscriber)

    def remove_subscriber(self, subscriber: Observer):
        self._subscribers.remove(subscriber)
