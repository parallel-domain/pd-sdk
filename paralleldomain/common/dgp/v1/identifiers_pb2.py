# flake8: noqa
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: identifiers.proto

import sys

_b = sys.version_info[0] < 3 and (lambda x: x) or (lambda x: x.encode("latin1"))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import timestamp_pb2 as google_dot_protobuf_dot_timestamp__pb2

DESCRIPTOR = _descriptor.FileDescriptor(
    name="identifiers.proto",
    package="dgp.proto",
    syntax="proto3",
    serialized_options=None,
    serialized_pb=_b(
        '\n\x11identifiers.proto\x12\tdgp.proto\x1a\x1fgoogle/protobuf/timestamp.proto"b\n\x07\x44\x61tumId\x12\x0b\n\x03log\x18\x01 \x01(\t\x12\x0c\n\x04name\x18\x02 \x01(\t\x12-\n\ttimestamp\x18\x03 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\r\n\x05index\x18\x04 \x01(\x04\x62\x06proto3'
    ),
    dependencies=[
        google_dot_protobuf_dot_timestamp__pb2.DESCRIPTOR,
    ],
)


_DATUMID = _descriptor.Descriptor(
    name="DatumId",
    full_name="dgp.proto.DatumId",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name="log",
            full_name="dgp.proto.DatumId.log",
            index=0,
            number=1,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=_b("").decode("utf-8"),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
        _descriptor.FieldDescriptor(
            name="name",
            full_name="dgp.proto.DatumId.name",
            index=1,
            number=2,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=_b("").decode("utf-8"),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
        _descriptor.FieldDescriptor(
            name="timestamp",
            full_name="dgp.proto.DatumId.timestamp",
            index=2,
            number=3,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
        _descriptor.FieldDescriptor(
            name="index",
            full_name="dgp.proto.DatumId.index",
            index=3,
            number=4,
            type=4,
            cpp_type=4,
            label=1,
            has_default_value=False,
            default_value=0,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=65,
    serialized_end=163,
)

_DATUMID.fields_by_name["timestamp"].message_type = google_dot_protobuf_dot_timestamp__pb2._TIMESTAMP
DESCRIPTOR.message_types_by_name["DatumId"] = _DATUMID
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

DatumId = _reflection.GeneratedProtocolMessageType(
    "DatumId",
    (_message.Message,),
    dict(
        DESCRIPTOR=_DATUMID,
        __module__="identifiers_pb2"
        # @@protoc_insertion_point(class_scope:dgp.proto.DatumId)
    ),
)
_sym_db.RegisterMessage(DatumId)


# @@protoc_insertion_point(module_scope)