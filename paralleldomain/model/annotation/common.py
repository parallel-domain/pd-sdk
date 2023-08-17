from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Optional, Type, TypeVar, List


class Annotation:
    ...


T = TypeVar("T", bound=Annotation)
AnnotationType = Type[Annotation]


@dataclass(frozen=True)
class AnnotationIdentifier(Generic[T]):
    """
    Used to identify annotations. In case there are multiple annotations of the same type you'll be able
    to differentiate them through their name.
    For backwards compatibility reasons, a name with some annotation_type T and name None is equal to the type T, also
    having the same hash.
    """

    annotation_type: Type[T]
    name: Optional[str] = None

    def __hash__(self):
        if self.name is None:
            return hash(self.annotation_type)
        else:
            return hash(self.annotation_type) + hash(self.name)

    def __eq__(self, other):
        if isinstance(other, AnnotationIdentifier):
            return str(other) == str(self)
        elif self.name is None and isinstance(other, type):
            return self.annotation_type.__name__ == other.__name__
        return False

    def __repr__(self):
        return f"{self.annotation_type.__name__}-{self.name}"

    def __str__(self):
        return f"{self.annotation_type.__name__}-{self.name}"

    @property
    def __name__(self):
        return str(self)

    @staticmethod
    def resolve_annotation_identifier(
        available_annotation_identifiers: List[AnnotationIdentifier],
        annotation_type: Optional[Type[T]] = None,
        annotation_identifier: Optional["AnnotationIdentifier"] = None,
        name: Optional[str] = None,
    ) -> "AnnotationIdentifier":
        """
        Resolves the given arguments into a valid annotation identifier, throws a ValueError if that's not possible.
        This method is used to be backwards compatible in most cases: Specifying the annotation_type is enough to
        resolve an identifier, unless there are multiple identifiers of the same annotation type in
        available_annotation_identifiers.
        Either annotation_type or annotation_identifier has to be not None.
        Args:
            available_annotation_identifiers: The list of identifiers that the result is selected from
            annotation_type: (Optional) the annotation type the identifier should have
            annotation_identifier: (Optional) an annotation identifier, specifying the type + optionally the name.
            name: (Optional) the name of the annotation identifier

        Returns:
            The resolved annotation identifier.
        """
        if annotation_identifier is None and annotation_type is not None:
            annotation_identifier = AnnotationIdentifier(name=name, annotation_type=annotation_type)
        elif annotation_identifier is None:
            raise ValueError("Either annotation_type or annotation_identifier need to be passed and not be None!")

        matching_identifiers = [
            i for i in available_annotation_identifiers if i.annotation_type == annotation_identifier.annotation_type
        ]
        if annotation_identifier.name is not None:
            matching_identifiers = [i for i in matching_identifiers if i.name == annotation_identifier.name]

        if len(matching_identifiers) == 0:
            raise ValueError(f"No annotation identifier {annotation_identifier} available!")
        if len(matching_identifiers) > 1:
            raise ValueError(
                f"Multiple annotation identifiers matching {annotation_identifier} available! Please specify a name to "
                f"select one."
            )

        return matching_identifiers[0]
