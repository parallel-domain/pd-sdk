from typing import List, Type

from paralleldomain.encoding.dgp.v1.encoding_format import DGPV1EncodingFormat
from paralleldomain.encoding.pipeline_encoder import EncodingFormat

known_formats: List[Type[EncodingFormat]] = [DGPV1EncodingFormat]


def register_encoding_format(format_type: Type[EncodingFormat]):
    if format_type not in known_formats:
        known_formats.append(format_type)


def get_encoding_format(
    format_name: str = "dgpv1",
    **format_kwargs,
) -> EncodingFormat:
    decoder_type = next((dtype for dtype in known_formats if format_name == dtype.get_format()), None)
    if decoder_type is not None:
        return decoder_type(**format_kwargs)
    else:
        known_format_names = [dt.get_format() for dt in known_formats]
        raise ValueError(
            f"Unknown Dataset format {known_formats}. Currently supported dataset formats are {known_format_names}"
        )
