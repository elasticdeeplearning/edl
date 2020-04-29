# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: master.proto

from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()

import common_pb2 as common__pb2

DESCRIPTOR = _descriptor.FileDescriptor(
    name='master.proto',
    package='master',
    syntax='proto3',
    serialized_options=None,
    serialized_pb=b'\n\x0cmaster.proto\x12\x06master\x1a\x0c\x63ommon.proto\"E\n\x0cRecordStatus\x12\x11\n\trecord_no\x18\x01 \x01(\x03\x12\"\n\x06status\x18\x02 \x01(\x0e\x32\x12.master.ProcStatus\"\x99\x01\n\x0b\x46ileDataSet\x12\x13\n\x0b\x64\x61ta_server\x18\x01 \x01(\t\x12\x13\n\x0bidx_in_list\x18\x02 \x01(\x03\x12\x11\n\tfile_path\x18\x03 \x01(\t\x12\'\n\x0b\x66ile_status\x18\x04 \x01(\x0e\x32\x12.master.ProcStatus\x12$\n\x06record\x18\x05 \x03(\x0b\x32\x14.master.RecordStatus\"M\n\nSubDataSet\x12\x1b\n\x03ret\x18\x01 \x01(\x0b\x32\x0e.common.RPCRet\x12\"\n\x05\x66iles\x18\x02 \x03(\x0b\x32\x13.master.FileDataSet\"=\n\x17ReportSubDataSetRequest\x12\"\n\x05\x66iles\x18\x01 \x03(\x0b\x32\x13.master.FileDataSet\"\x1c\n\x07\x44\x61taSet\x12\x11\n\tfile_list\x18\x01 \x01(\t\"\x11\n\x0fNewEpochRequest\"\x13\n\x11SubDataSetRequest*<\n\nProcStatus\x12\x0b\n\x07INITIAL\x10\x00\x12\x12\n\x0ePART_PROCESSED\x10\x01\x12\r\n\tPROCESSED\x10\x02\x32\xc8\x02\n\x06Master\x12?\n\nGetCluster\x12\x16.common.ClusterRequest\x1a\x17.common.ClusterResponse\"\x00\x12\x37\n\x12\x41ssignDataResource\x12\x0f.master.DataSet\x1a\x0e.common.RPCRet\"\x00\x12\x35\n\x08NewEpoch\x12\x17.master.NewEpochRequest\x1a\x0e.common.RPCRet\"\x00\x12@\n\rGetSubDataSet\x12\x19.master.SubDataSetRequest\x1a\x12.master.SubDataSet\"\x00\x12K\n\x16ReportSubDataSetStatus\x12\x1f.master.ReportSubDataSetRequest\x1a\x0e.common.RPCRet\"\x00\x62\x06proto3',
    dependencies=[common__pb2.DESCRIPTOR, ])

_PROCSTATUS = _descriptor.EnumDescriptor(
    name='ProcStatus',
    full_name='master.ProcStatus',
    filename=None,
    file=DESCRIPTOR,
    values=[
        _descriptor.EnumValueDescriptor(
            name='INITIAL',
            index=0,
            number=0,
            serialized_options=None,
            type=None),
        _descriptor.EnumValueDescriptor(
            name='PART_PROCESSED',
            index=1,
            number=1,
            serialized_options=None,
            type=None),
        _descriptor.EnumValueDescriptor(
            name='PROCESSED',
            index=2,
            number=2,
            serialized_options=None,
            type=None),
    ],
    containing_type=None,
    serialized_options=None,
    serialized_start=477,
    serialized_end=537, )
_sym_db.RegisterEnumDescriptor(_PROCSTATUS)

ProcStatus = enum_type_wrapper.EnumTypeWrapper(_PROCSTATUS)
INITIAL = 0
PART_PROCESSED = 1
PROCESSED = 2

_RECORDSTATUS = _descriptor.Descriptor(
    name='RecordStatus',
    full_name='master.RecordStatus',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='record_no',
            full_name='master.RecordStatus.record_no',
            index=0,
            number=1,
            type=3,
            cpp_type=2,
            label=1,
            has_default_value=False,
            default_value=0,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='status',
            full_name='master.RecordStatus.status',
            index=1,
            number=2,
            type=14,
            cpp_type=8,
            label=1,
            has_default_value=False,
            default_value=0,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[],
    serialized_start=38,
    serialized_end=107, )

_FILEDATASET = _descriptor.Descriptor(
    name='FileDataSet',
    full_name='master.FileDataSet',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='data_server',
            full_name='master.FileDataSet.data_server',
            index=0,
            number=1,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=b"".decode('utf-8'),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='idx_in_list',
            full_name='master.FileDataSet.idx_in_list',
            index=1,
            number=2,
            type=3,
            cpp_type=2,
            label=1,
            has_default_value=False,
            default_value=0,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='file_path',
            full_name='master.FileDataSet.file_path',
            index=2,
            number=3,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=b"".decode('utf-8'),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='file_status',
            full_name='master.FileDataSet.file_status',
            index=3,
            number=4,
            type=14,
            cpp_type=8,
            label=1,
            has_default_value=False,
            default_value=0,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='record',
            full_name='master.FileDataSet.record',
            index=4,
            number=5,
            type=11,
            cpp_type=10,
            label=3,
            has_default_value=False,
            default_value=[],
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[],
    serialized_start=110,
    serialized_end=263, )

_SUBDATASET = _descriptor.Descriptor(
    name='SubDataSet',
    full_name='master.SubDataSet',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='ret',
            full_name='master.SubDataSet.ret',
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
            file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='files',
            full_name='master.SubDataSet.files',
            index=1,
            number=2,
            type=11,
            cpp_type=10,
            label=3,
            has_default_value=False,
            default_value=[],
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[],
    serialized_start=265,
    serialized_end=342, )

_REPORTSUBDATASETREQUEST = _descriptor.Descriptor(
    name='ReportSubDataSetRequest',
    full_name='master.ReportSubDataSetRequest',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='files',
            full_name='master.ReportSubDataSetRequest.files',
            index=0,
            number=1,
            type=11,
            cpp_type=10,
            label=3,
            has_default_value=False,
            default_value=[],
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[],
    serialized_start=344,
    serialized_end=405, )

_DATASET = _descriptor.Descriptor(
    name='DataSet',
    full_name='master.DataSet',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='file_list',
            full_name='master.DataSet.file_list',
            index=0,
            number=1,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=b"".decode('utf-8'),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[],
    serialized_start=407,
    serialized_end=435, )

_NEWEPOCHREQUEST = _descriptor.Descriptor(
    name='NewEpochRequest',
    full_name='master.NewEpochRequest',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[],
    serialized_start=437,
    serialized_end=454, )

_SUBDATASETREQUEST = _descriptor.Descriptor(
    name='SubDataSetRequest',
    full_name='master.SubDataSetRequest',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[],
    serialized_start=456,
    serialized_end=475, )

_RECORDSTATUS.fields_by_name['status'].enum_type = _PROCSTATUS
_FILEDATASET.fields_by_name['file_status'].enum_type = _PROCSTATUS
_FILEDATASET.fields_by_name['record'].message_type = _RECORDSTATUS
_SUBDATASET.fields_by_name['ret'].message_type = common__pb2._RPCRET
_SUBDATASET.fields_by_name['files'].message_type = _FILEDATASET
_REPORTSUBDATASETREQUEST.fields_by_name['files'].message_type = _FILEDATASET
DESCRIPTOR.message_types_by_name['RecordStatus'] = _RECORDSTATUS
DESCRIPTOR.message_types_by_name['FileDataSet'] = _FILEDATASET
DESCRIPTOR.message_types_by_name['SubDataSet'] = _SUBDATASET
DESCRIPTOR.message_types_by_name[
    'ReportSubDataSetRequest'] = _REPORTSUBDATASETREQUEST
DESCRIPTOR.message_types_by_name['DataSet'] = _DATASET
DESCRIPTOR.message_types_by_name['NewEpochRequest'] = _NEWEPOCHREQUEST
DESCRIPTOR.message_types_by_name['SubDataSetRequest'] = _SUBDATASETREQUEST
DESCRIPTOR.enum_types_by_name['ProcStatus'] = _PROCSTATUS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

RecordStatus = _reflection.GeneratedProtocolMessageType(
    'RecordStatus',
    (_message.Message, ),
    {
        'DESCRIPTOR': _RECORDSTATUS,
        '__module__': 'master_pb2'
        # @@protoc_insertion_point(class_scope:master.RecordStatus)
    })
_sym_db.RegisterMessage(RecordStatus)

FileDataSet = _reflection.GeneratedProtocolMessageType(
    'FileDataSet',
    (_message.Message, ),
    {
        'DESCRIPTOR': _FILEDATASET,
        '__module__': 'master_pb2'
        # @@protoc_insertion_point(class_scope:master.FileDataSet)
    })
_sym_db.RegisterMessage(FileDataSet)

SubDataSet = _reflection.GeneratedProtocolMessageType(
    'SubDataSet',
    (_message.Message, ),
    {
        'DESCRIPTOR': _SUBDATASET,
        '__module__': 'master_pb2'
        # @@protoc_insertion_point(class_scope:master.SubDataSet)
    })
_sym_db.RegisterMessage(SubDataSet)

ReportSubDataSetRequest = _reflection.GeneratedProtocolMessageType(
    'ReportSubDataSetRequest',
    (_message.Message, ),
    {
        'DESCRIPTOR': _REPORTSUBDATASETREQUEST,
        '__module__': 'master_pb2'
        # @@protoc_insertion_point(class_scope:master.ReportSubDataSetRequest)
    })
_sym_db.RegisterMessage(ReportSubDataSetRequest)

DataSet = _reflection.GeneratedProtocolMessageType(
    'DataSet',
    (_message.Message, ),
    {
        'DESCRIPTOR': _DATASET,
        '__module__': 'master_pb2'
        # @@protoc_insertion_point(class_scope:master.DataSet)
    })
_sym_db.RegisterMessage(DataSet)

NewEpochRequest = _reflection.GeneratedProtocolMessageType(
    'NewEpochRequest',
    (_message.Message, ),
    {
        'DESCRIPTOR': _NEWEPOCHREQUEST,
        '__module__': 'master_pb2'
        # @@protoc_insertion_point(class_scope:master.NewEpochRequest)
    })
_sym_db.RegisterMessage(NewEpochRequest)

SubDataSetRequest = _reflection.GeneratedProtocolMessageType(
    'SubDataSetRequest',
    (_message.Message, ),
    {
        'DESCRIPTOR': _SUBDATASETREQUEST,
        '__module__': 'master_pb2'
        # @@protoc_insertion_point(class_scope:master.SubDataSetRequest)
    })
_sym_db.RegisterMessage(SubDataSetRequest)

_MASTER = _descriptor.ServiceDescriptor(
    name='Master',
    full_name='master.Master',
    file=DESCRIPTOR,
    index=0,
    serialized_options=None,
    serialized_start=540,
    serialized_end=868,
    methods=[
        _descriptor.MethodDescriptor(
            name='GetCluster',
            full_name='master.Master.GetCluster',
            index=0,
            containing_service=None,
            input_type=common__pb2._CLUSTERREQUEST,
            output_type=common__pb2._CLUSTERRESPONSE,
            serialized_options=None, ),
        _descriptor.MethodDescriptor(
            name='AssignDataResource',
            full_name='master.Master.AssignDataResource',
            index=1,
            containing_service=None,
            input_type=_DATASET,
            output_type=common__pb2._RPCRET,
            serialized_options=None, ),
        _descriptor.MethodDescriptor(
            name='NewEpoch',
            full_name='master.Master.NewEpoch',
            index=2,
            containing_service=None,
            input_type=_NEWEPOCHREQUEST,
            output_type=common__pb2._RPCRET,
            serialized_options=None, ),
        _descriptor.MethodDescriptor(
            name='GetSubDataSet',
            full_name='master.Master.GetSubDataSet',
            index=3,
            containing_service=None,
            input_type=_SUBDATASETREQUEST,
            output_type=_SUBDATASET,
            serialized_options=None, ),
        _descriptor.MethodDescriptor(
            name='ReportSubDataSetStatus',
            full_name='master.Master.ReportSubDataSetStatus',
            index=4,
            containing_service=None,
            input_type=_REPORTSUBDATASETREQUEST,
            output_type=common__pb2._RPCRET,
            serialized_options=None, ),
    ])
_sym_db.RegisterServiceDescriptor(_MASTER)

DESCRIPTOR.services_by_name['Master'] = _MASTER

# @@protoc_insertion_point(module_scope)
