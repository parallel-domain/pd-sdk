# flake8: noqa 
# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: dgp/proto/features.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from paralleldomain.common.dgp.v1.src.dgp.proto import geometry_pb2 as dgp_dot_proto_dot_geometry__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
    name="dgp/proto/features.proto",
    package="dgp.proto",
    syntax="proto3",
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
    serialized_pb=b"\n\x18\x64gp/proto/features.proto\x12\tdgp.proto\x1a\x18\x64gp/proto/geometry.proto*l\n\x0b\x46\x65\x61tureType\x12\x0c\n\x08\x41GENT_2D\x10\x00\x12\x0c\n\x08\x41GENT_3D\x10\x01\x12\x11\n\rEGO_INTENTION\x10\x02\x12\x0c\n\x08\x43ORRIDOR\x10\x03\x12\x10\n\x0cINTERSECTION\x10\x04\x12\x0e\n\nPARKED_CAR\x10\x05*)\n\x10\x46\x65\x61tureValueType\x12\x0b\n\x07NUMERIC\x10\x00\x12\x08\n\x04\x46ILE\x10\x01\x62\x06proto3",
    dependencies=[
        dgp_dot_proto_dot_geometry__pb2.DESCRIPTOR,
    ],
)

_FEATURETYPE = _descriptor.EnumDescriptor(
    name="FeatureType",
    full_name="dgp.proto.FeatureType",
    filename=None,
    file=DESCRIPTOR,
    create_key=_descriptor._internal_create_key,
    values=[
        _descriptor.EnumValueDescriptor(
            name="AGENT_2D",
            index=0,
            number=0,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.EnumValueDescriptor(
            name="AGENT_3D",
            index=1,
            number=1,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.EnumValueDescriptor(
            name="EGO_INTENTION",
            index=2,
            number=2,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.EnumValueDescriptor(
            name="CORRIDOR",
            index=3,
            number=3,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.EnumValueDescriptor(
            name="INTERSECTION",
            index=4,
            number=4,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.EnumValueDescriptor(
            name="PARKED_CAR",
            index=5,
            number=5,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    containing_type=None,
    serialized_options=None,
    serialized_start=65,
    serialized_end=173,
)
_sym_db.RegisterEnumDescriptor(_FEATURETYPE)

FeatureType = enum_type_wrapper.EnumTypeWrapper(_FEATURETYPE)
_FEATUREVALUETYPE = _descriptor.EnumDescriptor(
    name="FeatureValueType",
    full_name="dgp.proto.FeatureValueType",
    filename=None,
    file=DESCRIPTOR,
    create_key=_descriptor._internal_create_key,
    values=[
        _descriptor.EnumValueDescriptor(
            name="NUMERIC",
            index=0,
            number=0,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.EnumValueDescriptor(
            name="FILE",
            index=1,
            number=1,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    containing_type=None,
    serialized_options=None,
    serialized_start=175,
    serialized_end=216,
)
_sym_db.RegisterEnumDescriptor(_FEATUREVALUETYPE)

FeatureValueType = enum_type_wrapper.EnumTypeWrapper(_FEATUREVALUETYPE)
AGENT_2D = 0
AGENT_3D = 1
EGO_INTENTION = 2
CORRIDOR = 3
INTERSECTION = 4
PARKED_CAR = 5
NUMERIC = 0
FILE = 1


DESCRIPTOR.enum_types_by_name["FeatureType"] = _FEATURETYPE
DESCRIPTOR.enum_types_by_name["FeatureValueType"] = _FEATUREVALUETYPE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)


# @@protoc_insertion_point(module_scope)
