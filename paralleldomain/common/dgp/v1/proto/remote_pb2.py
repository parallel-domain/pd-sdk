# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: remote.proto

import sys

_b = sys.version_info[0] < 3 and (lambda x: x) or (lambda x: x.encode("latin1"))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


DESCRIPTOR = _descriptor.FileDescriptor(
    name="remote.proto",
    package="dgp.proto",
    syntax="proto3",
    serialized_options=None,
    serialized_pb=_b(
        '\n\x0cremote.proto\x12\tdgp.proto"\x1b\n\nRemotePath\x12\r\n\x05value\x18\x01 \x01(\t"Q\n\x0eRemoteArtifact\x12"\n\x03url\x18\x01 \x01(\x0b\x32\x15.dgp.proto.RemotePath\x12\x0c\n\x04sha1\x18\x02 \x01(\t\x12\r\n\x05isdir\x18\x03 \x01(\x08\x62\x06proto3'
    ),
)


_REMOTEPATH = _descriptor.Descriptor(
    name="RemotePath",
    full_name="dgp.proto.RemotePath",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name="value",
            full_name="dgp.proto.RemotePath.value",
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
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=27,
    serialized_end=54,
)


_REMOTEARTIFACT = _descriptor.Descriptor(
    name="RemoteArtifact",
    full_name="dgp.proto.RemoteArtifact",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name="url",
            full_name="dgp.proto.RemoteArtifact.url",
            index=0,
            number=1,
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
            name="sha1",
            full_name="dgp.proto.RemoteArtifact.sha1",
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
            name="isdir",
            full_name="dgp.proto.RemoteArtifact.isdir",
            index=2,
            number=3,
            type=8,
            cpp_type=7,
            label=1,
            has_default_value=False,
            default_value=False,
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
    serialized_start=56,
    serialized_end=137,
)

_REMOTEARTIFACT.fields_by_name["url"].message_type = _REMOTEPATH
DESCRIPTOR.message_types_by_name["RemotePath"] = _REMOTEPATH
DESCRIPTOR.message_types_by_name["RemoteArtifact"] = _REMOTEARTIFACT
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

RemotePath = _reflection.GeneratedProtocolMessageType(
    "RemotePath",
    (_message.Message,),
    dict(
        DESCRIPTOR=_REMOTEPATH,
        __module__="remote_pb2"
        # @@protoc_insertion_point(class_scope:dgp.proto.RemotePath)
    ),
)
_sym_db.RegisterMessage(RemotePath)

RemoteArtifact = _reflection.GeneratedProtocolMessageType(
    "RemoteArtifact",
    (_message.Message,),
    dict(
        DESCRIPTOR=_REMOTEARTIFACT,
        __module__="remote_pb2"
        # @@protoc_insertion_point(class_scope:dgp.proto.RemoteArtifact)
    ),
)
_sym_db.RegisterMessage(RemoteArtifact)


# @@protoc_insertion_point(module_scope)
