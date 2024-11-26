from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Union, List, Set


SubjectState = TypeVar("SubjectState")


class ISubject(ABC, Generic[SubjectState]):
    """
    The Subject interface declares a set of methods for managing subscribers.
    """

    _state: SubjectState

    @abstractmethod
    def attach(self, observer: Union["IObserver", List["IObserver"]]) -> None:
        """
        Attach an observer to the subject.
        """
        pass

    @abstractmethod
    def detach(self, observer: "IObserver") -> None:
        """
        Detach an observer from the subject.
        """
        pass

    @abstractmethod
    def notify(self) -> None:
        """
        Notify all observers about an event.
        """
        pass

    def get_state(self) -> SubjectState:
        return self._state


class Subject(ISubject[SubjectState]):
    _observers: set["IObserver"]

    def __init__(self):
        super().__init__()
        self._observers = set()

    def attach(self, observer: Union[List["IObserver"], "IObserver"]) -> None:
        if isinstance(observer, list):
            self._observers.update(observer)
        else:
            self._observers.add(observer)

    def detach(self, observer: "IObserver") -> None:
        self._observers.remove(observer)

    def notify(self) -> None:
        for observer in self._observers:
            observer.update(self)


class IObserver(ABC, Generic[SubjectState]):
    """
    The Observer interface declares the update method, used by subjects.
    """

    @abstractmethod
    def update(self, subject: ISubject[SubjectState]) -> None:
        """
        Receive update from subject.
        """
        pass


class Concrete(IObserver[int]):
    def update(self, subject: ISubject[int]) -> None:
        subject._state
