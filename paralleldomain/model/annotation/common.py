# from typing import TypeVar, Generic, Optional
#
# from paralleldomain.utilities.any_path import AnyPath
#
# try:
#     from typing import Protocol
# except ImportError:
#     from typing_extensions import Protocol  # type: ignore
#
# T = TypeVar("T")
#
#
# class AnnotationDecoderProtocol(Protocol[T]):
#     def get_annotation_data(self) -> T:
#         pass
#
#     def get_annotation_data_path(self) -> Optional[AnyPath]:
#         pass
#
#
# class Annotation(Generic[T]):
#     def __init__(self, annotation_decoder: AnnotationDecoderProtocol):
#         self._annotation_decoder = annotation_decoder
#         self._annotation_data_lazy_load = None
#         self._annotation_data_path_lazy_load = None
#
#     @property
#     def _annotation_data(self) -> T:
#         if self._annotation_data_lazy_load is None:
#             self._annotation_data_lazy_load = self._annotation_decoder.get_annotation_data()
#         return self._annotation_data_lazy_load
#
#     @property
#     def file_path(self) -> Optional[AnyPath]:
#         if self._annotation_data_path_lazy_load is None:
#             self._annotation_data_path_lazy_load = self._annotation_decoder.get_annotation_data_path()
#         return self._annotation_data_path_lazy_load


class Annotation:
    ...
