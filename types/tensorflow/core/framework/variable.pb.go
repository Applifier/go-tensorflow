// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: tensorflow/core/framework/variable.proto

package framework

import (
	fmt "fmt"
	proto "github.com/gogo/protobuf/proto"
	io "io"
	math "math"
	math_bits "math/bits"
)

// Reference imports to suppress errors if they are not otherwise used.
var _ = proto.Marshal
var _ = fmt.Errorf
var _ = math.Inf

// This is a compile-time assertion to ensure that this generated file
// is compatible with the proto package it is being compiled against.
// A compilation error at this line likely means your copy of the
// proto package needs to be updated.
const _ = proto.GoGoProtoPackageIsVersion3 // please upgrade the proto package

// Indicates when a distributed variable will be synced.
type VariableSynchronization int32

const (
	// `AUTO`: Indicates that the synchronization will be determined by the
	// current `DistributionStrategy` (eg. With `MirroredStrategy` this would be
	// `ON_WRITE`).
	VariableSynchronization_VARIABLE_SYNCHRONIZATION_AUTO VariableSynchronization = 0
	// `NONE`: Indicates that there will only be one copy of the variable, so
	// there is no need to sync.
	VariableSynchronization_VARIABLE_SYNCHRONIZATION_NONE VariableSynchronization = 1
	// `ON_WRITE`: Indicates that the variable will be updated across devices
	// every time it is written.
	VariableSynchronization_VARIABLE_SYNCHRONIZATION_ON_WRITE VariableSynchronization = 2
	// `ON_READ`: Indicates that the variable will be aggregated across devices
	// when it is read (eg. when checkpointing or when evaluating an op that uses
	// the variable).
	VariableSynchronization_VARIABLE_SYNCHRONIZATION_ON_READ VariableSynchronization = 3
)

var VariableSynchronization_name = map[int32]string{
	0: "VARIABLE_SYNCHRONIZATION_AUTO",
	1: "VARIABLE_SYNCHRONIZATION_NONE",
	2: "VARIABLE_SYNCHRONIZATION_ON_WRITE",
	3: "VARIABLE_SYNCHRONIZATION_ON_READ",
}

var VariableSynchronization_value = map[string]int32{
	"VARIABLE_SYNCHRONIZATION_AUTO":     0,
	"VARIABLE_SYNCHRONIZATION_NONE":     1,
	"VARIABLE_SYNCHRONIZATION_ON_WRITE": 2,
	"VARIABLE_SYNCHRONIZATION_ON_READ":  3,
}

func (x VariableSynchronization) String() string {
	return proto.EnumName(VariableSynchronization_name, int32(x))
}

func (VariableSynchronization) EnumDescriptor() ([]byte, []int) {
	return fileDescriptor_908f2d03adae2778, []int{0}
}

// Indicates how a distributed variable will be aggregated.
type VariableAggregation int32

const (
	// `NONE`: This is the default, giving an error if you use a
	// variable-update operation with multiple replicas.
	VariableAggregation_VARIABLE_AGGREGATION_NONE VariableAggregation = 0
	// `SUM`: Add the updates across replicas.
	VariableAggregation_VARIABLE_AGGREGATION_SUM VariableAggregation = 1
	// `MEAN`: Take the arithmetic mean ("average") of the updates across
	// replicas.
	VariableAggregation_VARIABLE_AGGREGATION_MEAN VariableAggregation = 2
	// `ONLY_FIRST_REPLICA`: This is for when every replica is performing the same
	// update, but we only want to perform the update once. Used, e.g., for the
	// global step counter.
	VariableAggregation_VARIABLE_AGGREGATION_ONLY_FIRST_REPLICA VariableAggregation = 3
)

var VariableAggregation_name = map[int32]string{
	0: "VARIABLE_AGGREGATION_NONE",
	1: "VARIABLE_AGGREGATION_SUM",
	2: "VARIABLE_AGGREGATION_MEAN",
	3: "VARIABLE_AGGREGATION_ONLY_FIRST_REPLICA",
}

var VariableAggregation_value = map[string]int32{
	"VARIABLE_AGGREGATION_NONE":               0,
	"VARIABLE_AGGREGATION_SUM":                1,
	"VARIABLE_AGGREGATION_MEAN":               2,
	"VARIABLE_AGGREGATION_ONLY_FIRST_REPLICA": 3,
}

func (x VariableAggregation) String() string {
	return proto.EnumName(VariableAggregation_name, int32(x))
}

func (VariableAggregation) EnumDescriptor() ([]byte, []int) {
	return fileDescriptor_908f2d03adae2778, []int{1}
}

// Protocol buffer representing a Variable.
type VariableDef struct {
	// Name of the variable tensor.
	VariableName string `protobuf:"bytes,1,opt,name=variable_name,json=variableName,proto3" json:"variable_name,omitempty"`
	// Name of the tensor holding the variable's initial value.
	InitialValueName string `protobuf:"bytes,6,opt,name=initial_value_name,json=initialValueName,proto3" json:"initial_value_name,omitempty"`
	// Name of the initializer op.
	InitializerName string `protobuf:"bytes,2,opt,name=initializer_name,json=initializerName,proto3" json:"initializer_name,omitempty"`
	// Name of the snapshot tensor.
	SnapshotName string `protobuf:"bytes,3,opt,name=snapshot_name,json=snapshotName,proto3" json:"snapshot_name,omitempty"`
	// Support for saving variables as slices of a larger variable.
	SaveSliceInfoDef *SaveSliceInfoDef `protobuf:"bytes,4,opt,name=save_slice_info_def,json=saveSliceInfoDef,proto3" json:"save_slice_info_def,omitempty"`
	// Whether to represent this as a ResourceVariable.
	IsResource bool `protobuf:"varint,5,opt,name=is_resource,json=isResource,proto3" json:"is_resource,omitempty"`
	// Whether this variable should be trained.
	Trainable bool `protobuf:"varint,7,opt,name=trainable,proto3" json:"trainable,omitempty"`
	// Indicates when a distributed variable will be synced.
	Synchronization VariableSynchronization `protobuf:"varint,8,opt,name=synchronization,proto3,enum=tensorflow.VariableSynchronization" json:"synchronization,omitempty"`
	// Indicates how a distributed variable will be aggregated.
	Aggregation VariableAggregation `protobuf:"varint,9,opt,name=aggregation,proto3,enum=tensorflow.VariableAggregation" json:"aggregation,omitempty"`
}

func (m *VariableDef) Reset()         { *m = VariableDef{} }
func (m *VariableDef) String() string { return proto.CompactTextString(m) }
func (*VariableDef) ProtoMessage()    {}
func (*VariableDef) Descriptor() ([]byte, []int) {
	return fileDescriptor_908f2d03adae2778, []int{0}
}
func (m *VariableDef) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *VariableDef) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_VariableDef.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *VariableDef) XXX_Merge(src proto.Message) {
	xxx_messageInfo_VariableDef.Merge(m, src)
}
func (m *VariableDef) XXX_Size() int {
	return m.Size()
}
func (m *VariableDef) XXX_DiscardUnknown() {
	xxx_messageInfo_VariableDef.DiscardUnknown(m)
}

var xxx_messageInfo_VariableDef proto.InternalMessageInfo

func (m *VariableDef) GetVariableName() string {
	if m != nil {
		return m.VariableName
	}
	return ""
}

func (m *VariableDef) GetInitialValueName() string {
	if m != nil {
		return m.InitialValueName
	}
	return ""
}

func (m *VariableDef) GetInitializerName() string {
	if m != nil {
		return m.InitializerName
	}
	return ""
}

func (m *VariableDef) GetSnapshotName() string {
	if m != nil {
		return m.SnapshotName
	}
	return ""
}

func (m *VariableDef) GetSaveSliceInfoDef() *SaveSliceInfoDef {
	if m != nil {
		return m.SaveSliceInfoDef
	}
	return nil
}

func (m *VariableDef) GetIsResource() bool {
	if m != nil {
		return m.IsResource
	}
	return false
}

func (m *VariableDef) GetTrainable() bool {
	if m != nil {
		return m.Trainable
	}
	return false
}

func (m *VariableDef) GetSynchronization() VariableSynchronization {
	if m != nil {
		return m.Synchronization
	}
	return VariableSynchronization_VARIABLE_SYNCHRONIZATION_AUTO
}

func (m *VariableDef) GetAggregation() VariableAggregation {
	if m != nil {
		return m.Aggregation
	}
	return VariableAggregation_VARIABLE_AGGREGATION_NONE
}

type SaveSliceInfoDef struct {
	// Name of the full variable of which this is a slice.
	FullName string `protobuf:"bytes,1,opt,name=full_name,json=fullName,proto3" json:"full_name,omitempty"`
	// Shape of the full variable.
	FullShape []int64 `protobuf:"varint,2,rep,packed,name=full_shape,json=fullShape,proto3" json:"full_shape,omitempty"`
	// Offset of this variable into the full variable.
	VarOffset []int64 `protobuf:"varint,3,rep,packed,name=var_offset,json=varOffset,proto3" json:"var_offset,omitempty"`
	// Shape of this variable.
	VarShape []int64 `protobuf:"varint,4,rep,packed,name=var_shape,json=varShape,proto3" json:"var_shape,omitempty"`
}

func (m *SaveSliceInfoDef) Reset()         { *m = SaveSliceInfoDef{} }
func (m *SaveSliceInfoDef) String() string { return proto.CompactTextString(m) }
func (*SaveSliceInfoDef) ProtoMessage()    {}
func (*SaveSliceInfoDef) Descriptor() ([]byte, []int) {
	return fileDescriptor_908f2d03adae2778, []int{1}
}
func (m *SaveSliceInfoDef) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *SaveSliceInfoDef) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_SaveSliceInfoDef.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *SaveSliceInfoDef) XXX_Merge(src proto.Message) {
	xxx_messageInfo_SaveSliceInfoDef.Merge(m, src)
}
func (m *SaveSliceInfoDef) XXX_Size() int {
	return m.Size()
}
func (m *SaveSliceInfoDef) XXX_DiscardUnknown() {
	xxx_messageInfo_SaveSliceInfoDef.DiscardUnknown(m)
}

var xxx_messageInfo_SaveSliceInfoDef proto.InternalMessageInfo

func (m *SaveSliceInfoDef) GetFullName() string {
	if m != nil {
		return m.FullName
	}
	return ""
}

func (m *SaveSliceInfoDef) GetFullShape() []int64 {
	if m != nil {
		return m.FullShape
	}
	return nil
}

func (m *SaveSliceInfoDef) GetVarOffset() []int64 {
	if m != nil {
		return m.VarOffset
	}
	return nil
}

func (m *SaveSliceInfoDef) GetVarShape() []int64 {
	if m != nil {
		return m.VarShape
	}
	return nil
}

func init() {
	proto.RegisterEnum("tensorflow.VariableSynchronization", VariableSynchronization_name, VariableSynchronization_value)
	proto.RegisterEnum("tensorflow.VariableAggregation", VariableAggregation_name, VariableAggregation_value)
	proto.RegisterType((*VariableDef)(nil), "tensorflow.VariableDef")
	proto.RegisterType((*SaveSliceInfoDef)(nil), "tensorflow.SaveSliceInfoDef")
}

func init() {
	proto.RegisterFile("tensorflow/core/framework/variable.proto", fileDescriptor_908f2d03adae2778)
}

var fileDescriptor_908f2d03adae2778 = []byte{
	// 610 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x7c, 0x94, 0xc1, 0x52, 0xd3, 0x40,
	0x18, 0xc7, 0xbb, 0x04, 0xb1, 0xdd, 0x0a, 0x64, 0x96, 0x83, 0x71, 0x84, 0x52, 0x40, 0xc7, 0x8a,
	0xda, 0xce, 0xe0, 0x13, 0xa4, 0x10, 0x31, 0x23, 0x24, 0xcc, 0xa6, 0xe0, 0xc0, 0x25, 0xb3, 0xd4,
	0x4d, 0xbb, 0x63, 0x9a, 0xed, 0xec, 0xa6, 0x61, 0xe0, 0x11, 0xf4, 0xe2, 0x13, 0xf8, 0x04, 0x3e,
	0x88, 0x27, 0x87, 0xa3, 0x47, 0x07, 0x5e, 0xc2, 0xa3, 0x93, 0x6d, 0x43, 0x63, 0xa7, 0xe5, 0xfa,
	0xff, 0xff, 0xfe, 0x5f, 0xf6, 0xfb, 0x36, 0xfb, 0xc1, 0x5a, 0x4c, 0x23, 0xc9, 0x45, 0x10, 0xf2,
	0x8b, 0x46, 0x9b, 0x0b, 0xda, 0x08, 0x04, 0xe9, 0xd1, 0x0b, 0x2e, 0x3e, 0x37, 0x12, 0x22, 0x18,
	0x39, 0x0f, 0x69, 0xbd, 0x2f, 0x78, 0xcc, 0x11, 0x1c, 0x93, 0x9b, 0xbf, 0x34, 0x58, 0x3e, 0x19,
	0xd9, 0x7b, 0x34, 0x40, 0x5b, 0x70, 0x31, 0xa3, 0xfd, 0x88, 0xf4, 0xa8, 0x01, 0xaa, 0xa0, 0x56,
	0xc2, 0x8f, 0x32, 0xd1, 0x21, 0x3d, 0x8a, 0x5e, 0x43, 0xc4, 0x22, 0x16, 0x33, 0x12, 0xfa, 0x09,
	0x09, 0x07, 0x23, 0x72, 0x41, 0x91, 0xfa, 0xc8, 0x39, 0x49, 0x0d, 0x45, 0xbf, 0x84, 0x99, 0xc6,
	0xae, 0xa8, 0x18, 0xb2, 0x73, 0x8a, 0x5d, 0xce, 0xe9, 0x0a, 0xdd, 0x82, 0x8b, 0x32, 0x22, 0x7d,
	0xd9, 0xe5, 0xf1, 0x90, 0xd3, 0x86, 0x5f, 0xcf, 0x44, 0x05, 0x7d, 0x80, 0x2b, 0x92, 0x24, 0xd4,
	0x97, 0x21, 0x6b, 0x53, 0x9f, 0x45, 0x01, 0xf7, 0x3f, 0xd1, 0xc0, 0x98, 0xaf, 0x82, 0x5a, 0x79,
	0x67, 0xb5, 0x3e, 0x6e, 0xae, 0xee, 0x91, 0x84, 0x7a, 0x29, 0x65, 0x47, 0x01, 0xdf, 0xa3, 0x01,
	0xd6, 0xe5, 0x84, 0x82, 0xd6, 0x61, 0x99, 0x49, 0x5f, 0x50, 0xc9, 0x07, 0xa2, 0x4d, 0x8d, 0x07,
	0x55, 0x50, 0x2b, 0x62, 0xc8, 0x24, 0x1e, 0x29, 0x68, 0x15, 0x96, 0x62, 0x41, 0x58, 0x94, 0x36,
	0x6f, 0x3c, 0x54, 0xf6, 0x58, 0x40, 0x87, 0x70, 0x59, 0x5e, 0x46, 0xed, 0xae, 0xe0, 0x11, 0xbb,
	0x22, 0x31, 0xe3, 0x91, 0x51, 0xac, 0x82, 0xda, 0xd2, 0xce, 0x56, 0xfe, 0x1c, 0xd9, 0x80, 0xbd,
	0xff, 0x51, 0x3c, 0x99, 0x45, 0x26, 0x2c, 0x93, 0x4e, 0x47, 0xd0, 0xce, 0xb0, 0x54, 0x49, 0x95,
	0x5a, 0x9f, 0x56, 0xca, 0x1c, 0x63, 0x38, 0x9f, 0xd9, 0xfc, 0x02, 0xa0, 0x3e, 0xd9, 0x37, 0x7a,
	0x0a, 0x4b, 0xc1, 0x20, 0x0c, 0xf3, 0x37, 0x5a, 0x4c, 0x05, 0x35, 0xcf, 0x35, 0x08, 0x95, 0x29,
	0xbb, 0xa4, 0x9f, 0xde, 0x8c, 0x56, 0xd3, 0xb0, 0xc2, 0xbd, 0x54, 0x48, 0xed, 0x84, 0x08, 0x9f,
	0x07, 0x81, 0xa4, 0xb1, 0xa1, 0x0d, 0xed, 0x84, 0x08, 0x57, 0x09, 0x69, 0xe9, 0xd4, 0x1e, 0x86,
	0xe7, 0x95, 0x5b, 0x4c, 0x88, 0x50, 0xd9, 0xed, 0x1f, 0x00, 0x3e, 0x9e, 0xd1, 0x3c, 0xda, 0x80,
	0x6b, 0x27, 0x26, 0xb6, 0xcd, 0xe6, 0x81, 0xe5, 0x7b, 0xa7, 0xce, 0xee, 0x7b, 0xec, 0x3a, 0xf6,
	0x99, 0xd9, 0xb2, 0x5d, 0xc7, 0x37, 0x8f, 0x5b, 0xae, 0x5e, 0xb8, 0x17, 0x71, 0x5c, 0xc7, 0xd2,
	0x01, 0x7a, 0x0e, 0x37, 0x66, 0x22, 0xae, 0xe3, 0x7f, 0xc4, 0x76, 0xcb, 0xd2, 0xe7, 0xd0, 0x33,
	0x58, 0xbd, 0x0f, 0xc3, 0x96, 0xb9, 0xa7, 0x6b, 0xdb, 0xdf, 0x01, 0x5c, 0x99, 0x32, 0x60, 0xb4,
	0x06, 0x9f, 0xdc, 0xa5, 0xcd, 0xfd, 0x7d, 0x6c, 0xed, 0xe7, 0xce, 0x50, 0x40, 0xab, 0xd0, 0x98,
	0x6a, 0x7b, 0xc7, 0x87, 0x3a, 0x98, 0x19, 0x3e, 0xb4, 0x4c, 0x47, 0x9f, 0x43, 0xaf, 0xe0, 0x8b,
	0xa9, 0xb6, 0xeb, 0x1c, 0x9c, 0xfa, 0xef, 0x6c, 0xec, 0xb5, 0x7c, 0x6c, 0x1d, 0x1d, 0xd8, 0xbb,
	0xa6, 0xae, 0x35, 0xbf, 0x82, 0x9f, 0x37, 0x15, 0x70, 0x7d, 0x53, 0x01, 0x7f, 0x6e, 0x2a, 0xe0,
	0xdb, 0x6d, 0xa5, 0x70, 0x7d, 0x5b, 0x29, 0xfc, 0xbe, 0xad, 0x14, 0xa0, 0xc1, 0x45, 0x27, 0xff,
	0xa3, 0xdc, 0xbd, 0xfe, 0xe6, 0x52, 0xd6, 0xd2, 0x51, 0xfa, 0xfa, 0xe5, 0x11, 0x38, 0x6b, 0x76,
	0x58, 0xdc, 0x1d, 0x9c, 0xd7, 0xdb, 0xbc, 0xd7, 0x30, 0xfb, 0xfd, 0x90, 0x05, 0x8c, 0x8a, 0x46,
	0x87, 0xbf, 0xc9, 0xad, 0x90, 0xf8, 0xb2, 0x4f, 0x65, 0x63, 0xe6, 0x4e, 0xf9, 0x0b, 0xc0, 0xf9,
	0x82, 0x5a, 0x27, 0x6f, 0xff, 0x05, 0x00, 0x00, 0xff, 0xff, 0xe5, 0x0c, 0xde, 0xa0, 0x7a, 0x04,
	0x00, 0x00,
}

func (m *VariableDef) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *VariableDef) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *VariableDef) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if m.Aggregation != 0 {
		i = encodeVarintVariable(dAtA, i, uint64(m.Aggregation))
		i--
		dAtA[i] = 0x48
	}
	if m.Synchronization != 0 {
		i = encodeVarintVariable(dAtA, i, uint64(m.Synchronization))
		i--
		dAtA[i] = 0x40
	}
	if m.Trainable {
		i--
		if m.Trainable {
			dAtA[i] = 1
		} else {
			dAtA[i] = 0
		}
		i--
		dAtA[i] = 0x38
	}
	if len(m.InitialValueName) > 0 {
		i -= len(m.InitialValueName)
		copy(dAtA[i:], m.InitialValueName)
		i = encodeVarintVariable(dAtA, i, uint64(len(m.InitialValueName)))
		i--
		dAtA[i] = 0x32
	}
	if m.IsResource {
		i--
		if m.IsResource {
			dAtA[i] = 1
		} else {
			dAtA[i] = 0
		}
		i--
		dAtA[i] = 0x28
	}
	if m.SaveSliceInfoDef != nil {
		{
			size, err := m.SaveSliceInfoDef.MarshalToSizedBuffer(dAtA[:i])
			if err != nil {
				return 0, err
			}
			i -= size
			i = encodeVarintVariable(dAtA, i, uint64(size))
		}
		i--
		dAtA[i] = 0x22
	}
	if len(m.SnapshotName) > 0 {
		i -= len(m.SnapshotName)
		copy(dAtA[i:], m.SnapshotName)
		i = encodeVarintVariable(dAtA, i, uint64(len(m.SnapshotName)))
		i--
		dAtA[i] = 0x1a
	}
	if len(m.InitializerName) > 0 {
		i -= len(m.InitializerName)
		copy(dAtA[i:], m.InitializerName)
		i = encodeVarintVariable(dAtA, i, uint64(len(m.InitializerName)))
		i--
		dAtA[i] = 0x12
	}
	if len(m.VariableName) > 0 {
		i -= len(m.VariableName)
		copy(dAtA[i:], m.VariableName)
		i = encodeVarintVariable(dAtA, i, uint64(len(m.VariableName)))
		i--
		dAtA[i] = 0xa
	}
	return len(dAtA) - i, nil
}

func (m *SaveSliceInfoDef) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *SaveSliceInfoDef) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *SaveSliceInfoDef) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if len(m.VarShape) > 0 {
		dAtA3 := make([]byte, len(m.VarShape)*10)
		var j2 int
		for _, num1 := range m.VarShape {
			num := uint64(num1)
			for num >= 1<<7 {
				dAtA3[j2] = uint8(uint64(num)&0x7f | 0x80)
				num >>= 7
				j2++
			}
			dAtA3[j2] = uint8(num)
			j2++
		}
		i -= j2
		copy(dAtA[i:], dAtA3[:j2])
		i = encodeVarintVariable(dAtA, i, uint64(j2))
		i--
		dAtA[i] = 0x22
	}
	if len(m.VarOffset) > 0 {
		dAtA5 := make([]byte, len(m.VarOffset)*10)
		var j4 int
		for _, num1 := range m.VarOffset {
			num := uint64(num1)
			for num >= 1<<7 {
				dAtA5[j4] = uint8(uint64(num)&0x7f | 0x80)
				num >>= 7
				j4++
			}
			dAtA5[j4] = uint8(num)
			j4++
		}
		i -= j4
		copy(dAtA[i:], dAtA5[:j4])
		i = encodeVarintVariable(dAtA, i, uint64(j4))
		i--
		dAtA[i] = 0x1a
	}
	if len(m.FullShape) > 0 {
		dAtA7 := make([]byte, len(m.FullShape)*10)
		var j6 int
		for _, num1 := range m.FullShape {
			num := uint64(num1)
			for num >= 1<<7 {
				dAtA7[j6] = uint8(uint64(num)&0x7f | 0x80)
				num >>= 7
				j6++
			}
			dAtA7[j6] = uint8(num)
			j6++
		}
		i -= j6
		copy(dAtA[i:], dAtA7[:j6])
		i = encodeVarintVariable(dAtA, i, uint64(j6))
		i--
		dAtA[i] = 0x12
	}
	if len(m.FullName) > 0 {
		i -= len(m.FullName)
		copy(dAtA[i:], m.FullName)
		i = encodeVarintVariable(dAtA, i, uint64(len(m.FullName)))
		i--
		dAtA[i] = 0xa
	}
	return len(dAtA) - i, nil
}

func encodeVarintVariable(dAtA []byte, offset int, v uint64) int {
	offset -= sovVariable(v)
	base := offset
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return base
}
func (m *VariableDef) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	l = len(m.VariableName)
	if l > 0 {
		n += 1 + l + sovVariable(uint64(l))
	}
	l = len(m.InitializerName)
	if l > 0 {
		n += 1 + l + sovVariable(uint64(l))
	}
	l = len(m.SnapshotName)
	if l > 0 {
		n += 1 + l + sovVariable(uint64(l))
	}
	if m.SaveSliceInfoDef != nil {
		l = m.SaveSliceInfoDef.Size()
		n += 1 + l + sovVariable(uint64(l))
	}
	if m.IsResource {
		n += 2
	}
	l = len(m.InitialValueName)
	if l > 0 {
		n += 1 + l + sovVariable(uint64(l))
	}
	if m.Trainable {
		n += 2
	}
	if m.Synchronization != 0 {
		n += 1 + sovVariable(uint64(m.Synchronization))
	}
	if m.Aggregation != 0 {
		n += 1 + sovVariable(uint64(m.Aggregation))
	}
	return n
}

func (m *SaveSliceInfoDef) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	l = len(m.FullName)
	if l > 0 {
		n += 1 + l + sovVariable(uint64(l))
	}
	if len(m.FullShape) > 0 {
		l = 0
		for _, e := range m.FullShape {
			l += sovVariable(uint64(e))
		}
		n += 1 + sovVariable(uint64(l)) + l
	}
	if len(m.VarOffset) > 0 {
		l = 0
		for _, e := range m.VarOffset {
			l += sovVariable(uint64(e))
		}
		n += 1 + sovVariable(uint64(l)) + l
	}
	if len(m.VarShape) > 0 {
		l = 0
		for _, e := range m.VarShape {
			l += sovVariable(uint64(e))
		}
		n += 1 + sovVariable(uint64(l)) + l
	}
	return n
}

func sovVariable(x uint64) (n int) {
	return (math_bits.Len64(x|1) + 6) / 7
}
func sozVariable(x uint64) (n int) {
	return sovVariable(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (m *VariableDef) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowVariable
			}
			if iNdEx >= l {
				return io.ErrUnexpectedEOF
			}
			b := dAtA[iNdEx]
			iNdEx++
			wire |= uint64(b&0x7F) << shift
			if b < 0x80 {
				break
			}
		}
		fieldNum := int32(wire >> 3)
		wireType := int(wire & 0x7)
		if wireType == 4 {
			return fmt.Errorf("proto: VariableDef: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: VariableDef: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field VariableName", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowVariable
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				stringLen |= uint64(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			intStringLen := int(stringLen)
			if intStringLen < 0 {
				return ErrInvalidLengthVariable
			}
			postIndex := iNdEx + intStringLen
			if postIndex < 0 {
				return ErrInvalidLengthVariable
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.VariableName = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		case 2:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field InitializerName", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowVariable
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				stringLen |= uint64(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			intStringLen := int(stringLen)
			if intStringLen < 0 {
				return ErrInvalidLengthVariable
			}
			postIndex := iNdEx + intStringLen
			if postIndex < 0 {
				return ErrInvalidLengthVariable
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.InitializerName = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		case 3:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field SnapshotName", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowVariable
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				stringLen |= uint64(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			intStringLen := int(stringLen)
			if intStringLen < 0 {
				return ErrInvalidLengthVariable
			}
			postIndex := iNdEx + intStringLen
			if postIndex < 0 {
				return ErrInvalidLengthVariable
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.SnapshotName = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		case 4:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field SaveSliceInfoDef", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowVariable
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				msglen |= int(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			if msglen < 0 {
				return ErrInvalidLengthVariable
			}
			postIndex := iNdEx + msglen
			if postIndex < 0 {
				return ErrInvalidLengthVariable
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			if m.SaveSliceInfoDef == nil {
				m.SaveSliceInfoDef = &SaveSliceInfoDef{}
			}
			if err := m.SaveSliceInfoDef.Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		case 5:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field IsResource", wireType)
			}
			var v int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowVariable
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				v |= int(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			m.IsResource = bool(v != 0)
		case 6:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field InitialValueName", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowVariable
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				stringLen |= uint64(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			intStringLen := int(stringLen)
			if intStringLen < 0 {
				return ErrInvalidLengthVariable
			}
			postIndex := iNdEx + intStringLen
			if postIndex < 0 {
				return ErrInvalidLengthVariable
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.InitialValueName = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		case 7:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field Trainable", wireType)
			}
			var v int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowVariable
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				v |= int(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			m.Trainable = bool(v != 0)
		case 8:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field Synchronization", wireType)
			}
			m.Synchronization = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowVariable
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.Synchronization |= VariableSynchronization(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		case 9:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field Aggregation", wireType)
			}
			m.Aggregation = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowVariable
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.Aggregation |= VariableAggregation(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		default:
			iNdEx = preIndex
			skippy, err := skipVariable(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if skippy < 0 {
				return ErrInvalidLengthVariable
			}
			if (iNdEx + skippy) < 0 {
				return ErrInvalidLengthVariable
			}
			if (iNdEx + skippy) > l {
				return io.ErrUnexpectedEOF
			}
			iNdEx += skippy
		}
	}

	if iNdEx > l {
		return io.ErrUnexpectedEOF
	}
	return nil
}
func (m *SaveSliceInfoDef) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowVariable
			}
			if iNdEx >= l {
				return io.ErrUnexpectedEOF
			}
			b := dAtA[iNdEx]
			iNdEx++
			wire |= uint64(b&0x7F) << shift
			if b < 0x80 {
				break
			}
		}
		fieldNum := int32(wire >> 3)
		wireType := int(wire & 0x7)
		if wireType == 4 {
			return fmt.Errorf("proto: SaveSliceInfoDef: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: SaveSliceInfoDef: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field FullName", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowVariable
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				stringLen |= uint64(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			intStringLen := int(stringLen)
			if intStringLen < 0 {
				return ErrInvalidLengthVariable
			}
			postIndex := iNdEx + intStringLen
			if postIndex < 0 {
				return ErrInvalidLengthVariable
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.FullName = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		case 2:
			if wireType == 0 {
				var v int64
				for shift := uint(0); ; shift += 7 {
					if shift >= 64 {
						return ErrIntOverflowVariable
					}
					if iNdEx >= l {
						return io.ErrUnexpectedEOF
					}
					b := dAtA[iNdEx]
					iNdEx++
					v |= int64(b&0x7F) << shift
					if b < 0x80 {
						break
					}
				}
				m.FullShape = append(m.FullShape, v)
			} else if wireType == 2 {
				var packedLen int
				for shift := uint(0); ; shift += 7 {
					if shift >= 64 {
						return ErrIntOverflowVariable
					}
					if iNdEx >= l {
						return io.ErrUnexpectedEOF
					}
					b := dAtA[iNdEx]
					iNdEx++
					packedLen |= int(b&0x7F) << shift
					if b < 0x80 {
						break
					}
				}
				if packedLen < 0 {
					return ErrInvalidLengthVariable
				}
				postIndex := iNdEx + packedLen
				if postIndex < 0 {
					return ErrInvalidLengthVariable
				}
				if postIndex > l {
					return io.ErrUnexpectedEOF
				}
				var elementCount int
				var count int
				for _, integer := range dAtA[iNdEx:postIndex] {
					if integer < 128 {
						count++
					}
				}
				elementCount = count
				if elementCount != 0 && len(m.FullShape) == 0 {
					m.FullShape = make([]int64, 0, elementCount)
				}
				for iNdEx < postIndex {
					var v int64
					for shift := uint(0); ; shift += 7 {
						if shift >= 64 {
							return ErrIntOverflowVariable
						}
						if iNdEx >= l {
							return io.ErrUnexpectedEOF
						}
						b := dAtA[iNdEx]
						iNdEx++
						v |= int64(b&0x7F) << shift
						if b < 0x80 {
							break
						}
					}
					m.FullShape = append(m.FullShape, v)
				}
			} else {
				return fmt.Errorf("proto: wrong wireType = %d for field FullShape", wireType)
			}
		case 3:
			if wireType == 0 {
				var v int64
				for shift := uint(0); ; shift += 7 {
					if shift >= 64 {
						return ErrIntOverflowVariable
					}
					if iNdEx >= l {
						return io.ErrUnexpectedEOF
					}
					b := dAtA[iNdEx]
					iNdEx++
					v |= int64(b&0x7F) << shift
					if b < 0x80 {
						break
					}
				}
				m.VarOffset = append(m.VarOffset, v)
			} else if wireType == 2 {
				var packedLen int
				for shift := uint(0); ; shift += 7 {
					if shift >= 64 {
						return ErrIntOverflowVariable
					}
					if iNdEx >= l {
						return io.ErrUnexpectedEOF
					}
					b := dAtA[iNdEx]
					iNdEx++
					packedLen |= int(b&0x7F) << shift
					if b < 0x80 {
						break
					}
				}
				if packedLen < 0 {
					return ErrInvalidLengthVariable
				}
				postIndex := iNdEx + packedLen
				if postIndex < 0 {
					return ErrInvalidLengthVariable
				}
				if postIndex > l {
					return io.ErrUnexpectedEOF
				}
				var elementCount int
				var count int
				for _, integer := range dAtA[iNdEx:postIndex] {
					if integer < 128 {
						count++
					}
				}
				elementCount = count
				if elementCount != 0 && len(m.VarOffset) == 0 {
					m.VarOffset = make([]int64, 0, elementCount)
				}
				for iNdEx < postIndex {
					var v int64
					for shift := uint(0); ; shift += 7 {
						if shift >= 64 {
							return ErrIntOverflowVariable
						}
						if iNdEx >= l {
							return io.ErrUnexpectedEOF
						}
						b := dAtA[iNdEx]
						iNdEx++
						v |= int64(b&0x7F) << shift
						if b < 0x80 {
							break
						}
					}
					m.VarOffset = append(m.VarOffset, v)
				}
			} else {
				return fmt.Errorf("proto: wrong wireType = %d for field VarOffset", wireType)
			}
		case 4:
			if wireType == 0 {
				var v int64
				for shift := uint(0); ; shift += 7 {
					if shift >= 64 {
						return ErrIntOverflowVariable
					}
					if iNdEx >= l {
						return io.ErrUnexpectedEOF
					}
					b := dAtA[iNdEx]
					iNdEx++
					v |= int64(b&0x7F) << shift
					if b < 0x80 {
						break
					}
				}
				m.VarShape = append(m.VarShape, v)
			} else if wireType == 2 {
				var packedLen int
				for shift := uint(0); ; shift += 7 {
					if shift >= 64 {
						return ErrIntOverflowVariable
					}
					if iNdEx >= l {
						return io.ErrUnexpectedEOF
					}
					b := dAtA[iNdEx]
					iNdEx++
					packedLen |= int(b&0x7F) << shift
					if b < 0x80 {
						break
					}
				}
				if packedLen < 0 {
					return ErrInvalidLengthVariable
				}
				postIndex := iNdEx + packedLen
				if postIndex < 0 {
					return ErrInvalidLengthVariable
				}
				if postIndex > l {
					return io.ErrUnexpectedEOF
				}
				var elementCount int
				var count int
				for _, integer := range dAtA[iNdEx:postIndex] {
					if integer < 128 {
						count++
					}
				}
				elementCount = count
				if elementCount != 0 && len(m.VarShape) == 0 {
					m.VarShape = make([]int64, 0, elementCount)
				}
				for iNdEx < postIndex {
					var v int64
					for shift := uint(0); ; shift += 7 {
						if shift >= 64 {
							return ErrIntOverflowVariable
						}
						if iNdEx >= l {
							return io.ErrUnexpectedEOF
						}
						b := dAtA[iNdEx]
						iNdEx++
						v |= int64(b&0x7F) << shift
						if b < 0x80 {
							break
						}
					}
					m.VarShape = append(m.VarShape, v)
				}
			} else {
				return fmt.Errorf("proto: wrong wireType = %d for field VarShape", wireType)
			}
		default:
			iNdEx = preIndex
			skippy, err := skipVariable(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if skippy < 0 {
				return ErrInvalidLengthVariable
			}
			if (iNdEx + skippy) < 0 {
				return ErrInvalidLengthVariable
			}
			if (iNdEx + skippy) > l {
				return io.ErrUnexpectedEOF
			}
			iNdEx += skippy
		}
	}

	if iNdEx > l {
		return io.ErrUnexpectedEOF
	}
	return nil
}
func skipVariable(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	depth := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowVariable
			}
			if iNdEx >= l {
				return 0, io.ErrUnexpectedEOF
			}
			b := dAtA[iNdEx]
			iNdEx++
			wire |= (uint64(b) & 0x7F) << shift
			if b < 0x80 {
				break
			}
		}
		wireType := int(wire & 0x7)
		switch wireType {
		case 0:
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return 0, ErrIntOverflowVariable
				}
				if iNdEx >= l {
					return 0, io.ErrUnexpectedEOF
				}
				iNdEx++
				if dAtA[iNdEx-1] < 0x80 {
					break
				}
			}
		case 1:
			iNdEx += 8
		case 2:
			var length int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return 0, ErrIntOverflowVariable
				}
				if iNdEx >= l {
					return 0, io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				length |= (int(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			if length < 0 {
				return 0, ErrInvalidLengthVariable
			}
			iNdEx += length
		case 3:
			depth++
		case 4:
			if depth == 0 {
				return 0, ErrUnexpectedEndOfGroupVariable
			}
			depth--
		case 5:
			iNdEx += 4
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
		if iNdEx < 0 {
			return 0, ErrInvalidLengthVariable
		}
		if depth == 0 {
			return iNdEx, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}

var (
	ErrInvalidLengthVariable        = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowVariable          = fmt.Errorf("proto: integer overflow")
	ErrUnexpectedEndOfGroupVariable = fmt.Errorf("proto: unexpected end of group")
)
