// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: tensorflow/core/protobuf/trackable_object_graph.proto

package protobuf

import (
	fmt "fmt"
	proto "github.com/gogo/protobuf/proto"
	io "io"
	math "math"
)

// Reference imports to suppress errors if they are not otherwise used.
var _ = proto.Marshal
var _ = fmt.Errorf
var _ = math.Inf

// This is a compile-time assertion to ensure that this generated file
// is compatible with the proto package it is being compiled against.
// A compilation error at this line likely means your copy of the
// proto package needs to be updated.
const _ = proto.GoGoProtoPackageIsVersion2 // please upgrade the proto package

type TrackableObjectGraph struct {
	Nodes []*TrackableObjectGraph_TrackableObject `protobuf:"bytes,1,rep,name=nodes,proto3" json:"nodes,omitempty"`
}

func (m *TrackableObjectGraph) Reset()         { *m = TrackableObjectGraph{} }
func (m *TrackableObjectGraph) String() string { return proto.CompactTextString(m) }
func (*TrackableObjectGraph) ProtoMessage()    {}
func (*TrackableObjectGraph) Descriptor() ([]byte, []int) {
	return fileDescriptor_120a5309f807e789, []int{0}
}
func (m *TrackableObjectGraph) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *TrackableObjectGraph) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_TrackableObjectGraph.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalTo(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *TrackableObjectGraph) XXX_Merge(src proto.Message) {
	xxx_messageInfo_TrackableObjectGraph.Merge(m, src)
}
func (m *TrackableObjectGraph) XXX_Size() int {
	return m.Size()
}
func (m *TrackableObjectGraph) XXX_DiscardUnknown() {
	xxx_messageInfo_TrackableObjectGraph.DiscardUnknown(m)
}

var xxx_messageInfo_TrackableObjectGraph proto.InternalMessageInfo

func (m *TrackableObjectGraph) GetNodes() []*TrackableObjectGraph_TrackableObject {
	if m != nil {
		return m.Nodes
	}
	return nil
}

type TrackableObjectGraph_TrackableObject struct {
	// Objects which this object depends on.
	Children []*TrackableObjectGraph_TrackableObject_ObjectReference `protobuf:"bytes,1,rep,name=children,proto3" json:"children,omitempty"`
	// Serialized data specific to this object.
	Attributes []*TrackableObjectGraph_TrackableObject_SerializedTensor `protobuf:"bytes,2,rep,name=attributes,proto3" json:"attributes,omitempty"`
	// Slot variables owned by this object.
	SlotVariables []*TrackableObjectGraph_TrackableObject_SlotVariableReference `protobuf:"bytes,3,rep,name=slot_variables,json=slotVariables,proto3" json:"slot_variables,omitempty"`
}

func (m *TrackableObjectGraph_TrackableObject) Reset()         { *m = TrackableObjectGraph_TrackableObject{} }
func (m *TrackableObjectGraph_TrackableObject) String() string { return proto.CompactTextString(m) }
func (*TrackableObjectGraph_TrackableObject) ProtoMessage()    {}
func (*TrackableObjectGraph_TrackableObject) Descriptor() ([]byte, []int) {
	return fileDescriptor_120a5309f807e789, []int{0, 0}
}
func (m *TrackableObjectGraph_TrackableObject) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *TrackableObjectGraph_TrackableObject) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_TrackableObjectGraph_TrackableObject.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalTo(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *TrackableObjectGraph_TrackableObject) XXX_Merge(src proto.Message) {
	xxx_messageInfo_TrackableObjectGraph_TrackableObject.Merge(m, src)
}
func (m *TrackableObjectGraph_TrackableObject) XXX_Size() int {
	return m.Size()
}
func (m *TrackableObjectGraph_TrackableObject) XXX_DiscardUnknown() {
	xxx_messageInfo_TrackableObjectGraph_TrackableObject.DiscardUnknown(m)
}

var xxx_messageInfo_TrackableObjectGraph_TrackableObject proto.InternalMessageInfo

func (m *TrackableObjectGraph_TrackableObject) GetChildren() []*TrackableObjectGraph_TrackableObject_ObjectReference {
	if m != nil {
		return m.Children
	}
	return nil
}

func (m *TrackableObjectGraph_TrackableObject) GetAttributes() []*TrackableObjectGraph_TrackableObject_SerializedTensor {
	if m != nil {
		return m.Attributes
	}
	return nil
}

func (m *TrackableObjectGraph_TrackableObject) GetSlotVariables() []*TrackableObjectGraph_TrackableObject_SlotVariableReference {
	if m != nil {
		return m.SlotVariables
	}
	return nil
}

type TrackableObjectGraph_TrackableObject_ObjectReference struct {
	// An index into `TrackableObjectGraph.nodes`, indicating the object
	// being referenced.
	NodeId int32 `protobuf:"varint,1,opt,name=node_id,json=nodeId,proto3" json:"node_id,omitempty"`
	// A user-provided name for the edge.
	LocalName string `protobuf:"bytes,2,opt,name=local_name,json=localName,proto3" json:"local_name,omitempty"`
}

func (m *TrackableObjectGraph_TrackableObject_ObjectReference) Reset() {
	*m = TrackableObjectGraph_TrackableObject_ObjectReference{}
}
func (m *TrackableObjectGraph_TrackableObject_ObjectReference) String() string {
	return proto.CompactTextString(m)
}
func (*TrackableObjectGraph_TrackableObject_ObjectReference) ProtoMessage() {}
func (*TrackableObjectGraph_TrackableObject_ObjectReference) Descriptor() ([]byte, []int) {
	return fileDescriptor_120a5309f807e789, []int{0, 0, 0}
}
func (m *TrackableObjectGraph_TrackableObject_ObjectReference) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *TrackableObjectGraph_TrackableObject_ObjectReference) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_TrackableObjectGraph_TrackableObject_ObjectReference.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalTo(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *TrackableObjectGraph_TrackableObject_ObjectReference) XXX_Merge(src proto.Message) {
	xxx_messageInfo_TrackableObjectGraph_TrackableObject_ObjectReference.Merge(m, src)
}
func (m *TrackableObjectGraph_TrackableObject_ObjectReference) XXX_Size() int {
	return m.Size()
}
func (m *TrackableObjectGraph_TrackableObject_ObjectReference) XXX_DiscardUnknown() {
	xxx_messageInfo_TrackableObjectGraph_TrackableObject_ObjectReference.DiscardUnknown(m)
}

var xxx_messageInfo_TrackableObjectGraph_TrackableObject_ObjectReference proto.InternalMessageInfo

func (m *TrackableObjectGraph_TrackableObject_ObjectReference) GetNodeId() int32 {
	if m != nil {
		return m.NodeId
	}
	return 0
}

func (m *TrackableObjectGraph_TrackableObject_ObjectReference) GetLocalName() string {
	if m != nil {
		return m.LocalName
	}
	return ""
}

type TrackableObjectGraph_TrackableObject_SerializedTensor struct {
	// A name for the Tensor. Simple variables have only one
	// `SerializedTensor` named "VARIABLE_VALUE" by convention. This value may
	// be restored on object creation as an optimization.
	Name string `protobuf:"bytes,1,opt,name=name,proto3" json:"name,omitempty"`
	// The full name of the variable/tensor, if applicable. Used to allow
	// name-based loading of checkpoints which were saved using an
	// object-based API. Should match the checkpoint key which would have been
	// assigned by tf.train.Saver.
	FullName string `protobuf:"bytes,2,opt,name=full_name,json=fullName,proto3" json:"full_name,omitempty"`
	// The generated name of the Tensor in the checkpoint.
	CheckpointKey string `protobuf:"bytes,3,opt,name=checkpoint_key,json=checkpointKey,proto3" json:"checkpoint_key,omitempty"`
	// Whether checkpoints should be considered as matching even without this
	// value restored. Used for non-critical values which don't affect the
	// TensorFlow graph, such as layer configurations.
	OptionalRestore bool `protobuf:"varint,4,opt,name=optional_restore,json=optionalRestore,proto3" json:"optional_restore,omitempty"`
}

func (m *TrackableObjectGraph_TrackableObject_SerializedTensor) Reset() {
	*m = TrackableObjectGraph_TrackableObject_SerializedTensor{}
}
func (m *TrackableObjectGraph_TrackableObject_SerializedTensor) String() string {
	return proto.CompactTextString(m)
}
func (*TrackableObjectGraph_TrackableObject_SerializedTensor) ProtoMessage() {}
func (*TrackableObjectGraph_TrackableObject_SerializedTensor) Descriptor() ([]byte, []int) {
	return fileDescriptor_120a5309f807e789, []int{0, 0, 1}
}
func (m *TrackableObjectGraph_TrackableObject_SerializedTensor) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *TrackableObjectGraph_TrackableObject_SerializedTensor) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_TrackableObjectGraph_TrackableObject_SerializedTensor.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalTo(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *TrackableObjectGraph_TrackableObject_SerializedTensor) XXX_Merge(src proto.Message) {
	xxx_messageInfo_TrackableObjectGraph_TrackableObject_SerializedTensor.Merge(m, src)
}
func (m *TrackableObjectGraph_TrackableObject_SerializedTensor) XXX_Size() int {
	return m.Size()
}
func (m *TrackableObjectGraph_TrackableObject_SerializedTensor) XXX_DiscardUnknown() {
	xxx_messageInfo_TrackableObjectGraph_TrackableObject_SerializedTensor.DiscardUnknown(m)
}

var xxx_messageInfo_TrackableObjectGraph_TrackableObject_SerializedTensor proto.InternalMessageInfo

func (m *TrackableObjectGraph_TrackableObject_SerializedTensor) GetName() string {
	if m != nil {
		return m.Name
	}
	return ""
}

func (m *TrackableObjectGraph_TrackableObject_SerializedTensor) GetFullName() string {
	if m != nil {
		return m.FullName
	}
	return ""
}

func (m *TrackableObjectGraph_TrackableObject_SerializedTensor) GetCheckpointKey() string {
	if m != nil {
		return m.CheckpointKey
	}
	return ""
}

func (m *TrackableObjectGraph_TrackableObject_SerializedTensor) GetOptionalRestore() bool {
	if m != nil {
		return m.OptionalRestore
	}
	return false
}

type TrackableObjectGraph_TrackableObject_SlotVariableReference struct {
	// An index into `TrackableObjectGraph.nodes`, indicating the
	// variable object this slot was created for.
	OriginalVariableNodeId int32 `protobuf:"varint,1,opt,name=original_variable_node_id,json=originalVariableNodeId,proto3" json:"original_variable_node_id,omitempty"`
	// The name of the slot (e.g. "m"/"v").
	SlotName string `protobuf:"bytes,2,opt,name=slot_name,json=slotName,proto3" json:"slot_name,omitempty"`
	// An index into `TrackableObjectGraph.nodes`, indicating the
	// `Object` with the value of the slot variable.
	SlotVariableNodeId int32 `protobuf:"varint,3,opt,name=slot_variable_node_id,json=slotVariableNodeId,proto3" json:"slot_variable_node_id,omitempty"`
}

func (m *TrackableObjectGraph_TrackableObject_SlotVariableReference) Reset() {
	*m = TrackableObjectGraph_TrackableObject_SlotVariableReference{}
}
func (m *TrackableObjectGraph_TrackableObject_SlotVariableReference) String() string {
	return proto.CompactTextString(m)
}
func (*TrackableObjectGraph_TrackableObject_SlotVariableReference) ProtoMessage() {}
func (*TrackableObjectGraph_TrackableObject_SlotVariableReference) Descriptor() ([]byte, []int) {
	return fileDescriptor_120a5309f807e789, []int{0, 0, 2}
}
func (m *TrackableObjectGraph_TrackableObject_SlotVariableReference) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *TrackableObjectGraph_TrackableObject_SlotVariableReference) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_TrackableObjectGraph_TrackableObject_SlotVariableReference.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalTo(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *TrackableObjectGraph_TrackableObject_SlotVariableReference) XXX_Merge(src proto.Message) {
	xxx_messageInfo_TrackableObjectGraph_TrackableObject_SlotVariableReference.Merge(m, src)
}
func (m *TrackableObjectGraph_TrackableObject_SlotVariableReference) XXX_Size() int {
	return m.Size()
}
func (m *TrackableObjectGraph_TrackableObject_SlotVariableReference) XXX_DiscardUnknown() {
	xxx_messageInfo_TrackableObjectGraph_TrackableObject_SlotVariableReference.DiscardUnknown(m)
}

var xxx_messageInfo_TrackableObjectGraph_TrackableObject_SlotVariableReference proto.InternalMessageInfo

func (m *TrackableObjectGraph_TrackableObject_SlotVariableReference) GetOriginalVariableNodeId() int32 {
	if m != nil {
		return m.OriginalVariableNodeId
	}
	return 0
}

func (m *TrackableObjectGraph_TrackableObject_SlotVariableReference) GetSlotName() string {
	if m != nil {
		return m.SlotName
	}
	return ""
}

func (m *TrackableObjectGraph_TrackableObject_SlotVariableReference) GetSlotVariableNodeId() int32 {
	if m != nil {
		return m.SlotVariableNodeId
	}
	return 0
}

func init() {
	proto.RegisterType((*TrackableObjectGraph)(nil), "tensorflow.TrackableObjectGraph")
	proto.RegisterType((*TrackableObjectGraph_TrackableObject)(nil), "tensorflow.TrackableObjectGraph.TrackableObject")
	proto.RegisterType((*TrackableObjectGraph_TrackableObject_ObjectReference)(nil), "tensorflow.TrackableObjectGraph.TrackableObject.ObjectReference")
	proto.RegisterType((*TrackableObjectGraph_TrackableObject_SerializedTensor)(nil), "tensorflow.TrackableObjectGraph.TrackableObject.SerializedTensor")
	proto.RegisterType((*TrackableObjectGraph_TrackableObject_SlotVariableReference)(nil), "tensorflow.TrackableObjectGraph.TrackableObject.SlotVariableReference")
}

func init() {
	proto.RegisterFile("tensorflow/core/protobuf/trackable_object_graph.proto", fileDescriptor_120a5309f807e789)
}

var fileDescriptor_120a5309f807e789 = []byte{
	// 491 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x94, 0x93, 0xcd, 0x6e, 0xd3, 0x40,
	0x14, 0x85, 0x3b, 0x4d, 0x13, 0x92, 0x8b, 0xda, 0x54, 0x23, 0x0a, 0x26, 0x15, 0x56, 0x84, 0x84,
	0x14, 0x16, 0xd8, 0xfc, 0x88, 0x05, 0x3b, 0xc2, 0xa2, 0xa8, 0x42, 0x2a, 0x92, 0xa9, 0x58, 0x20,
	0x84, 0x35, 0xb6, 0xaf, 0x93, 0x21, 0x13, 0x8f, 0x35, 0x33, 0x01, 0x85, 0xa7, 0x60, 0xc3, 0x0b,
	0xb0, 0xe4, 0x49, 0x60, 0xd7, 0x25, 0x4b, 0x94, 0xbc, 0x04, 0x4b, 0xe4, 0x49, 0xdc, 0x38, 0x51,
	0x59, 0x64, 0x67, 0x9f, 0x39, 0xe7, 0x1b, 0x9d, 0x7b, 0x35, 0xf0, 0xd4, 0x60, 0xa6, 0xa5, 0x4a,
	0x85, 0xfc, 0xec, 0xc7, 0x52, 0xa1, 0x9f, 0x2b, 0x69, 0x64, 0x34, 0x49, 0x7d, 0xa3, 0x58, 0x3c,
	0x62, 0x91, 0xc0, 0x50, 0x46, 0x1f, 0x31, 0x36, 0xe1, 0x40, 0xb1, 0x7c, 0xe8, 0xd9, 0x73, 0x0a,
	0xab, 0xd8, 0xdd, 0x1f, 0x0d, 0xb8, 0x71, 0x5e, 0x9a, 0x5f, 0x5b, 0xef, 0xcb, 0xc2, 0x4a, 0x4f,
	0xa0, 0x9e, 0xc9, 0x04, 0xb5, 0x43, 0xba, 0xb5, 0xde, 0xf5, 0xc7, 0x0f, 0xbd, 0x55, 0xc8, 0xbb,
	0x2a, 0xb0, 0x29, 0x06, 0x8b, 0x78, 0xe7, 0x57, 0x1d, 0xda, 0x1b, 0x47, 0xf4, 0x3d, 0x34, 0xe3,
	0x21, 0x17, 0x89, 0xc2, 0x6c, 0x89, 0x7f, 0xbe, 0x2d, 0xde, 0x5b, 0xde, 0x82, 0x29, 0x2a, 0xcc,
	0x62, 0x0c, 0x2e, 0x89, 0x94, 0x01, 0x30, 0x63, 0x14, 0x8f, 0x26, 0x06, 0xb5, 0xb3, 0x6b, 0xf9,
	0xfd, 0xad, 0xf9, 0x6f, 0x50, 0x71, 0x26, 0xf8, 0x17, 0x4c, 0xce, 0x6d, 0x32, 0xa8, 0x40, 0xe9,
	0x18, 0x0e, 0xb4, 0x90, 0x26, 0xfc, 0xc4, 0x14, 0x2f, 0x32, 0xda, 0xa9, 0xd9, 0x6b, 0x4e, 0xb6,
	0xbf, 0x46, 0x48, 0xf3, 0x76, 0x49, 0x59, 0x95, 0xd9, 0xd7, 0x15, 0x59, 0x77, 0x4e, 0xa1, 0xbd,
	0x51, 0x97, 0xde, 0x82, 0x6b, 0xc5, 0x7c, 0x43, 0x9e, 0x38, 0xa4, 0x4b, 0x7a, 0xf5, 0xa0, 0x51,
	0xfc, 0x9e, 0x26, 0xf4, 0x0e, 0x80, 0x90, 0x31, 0x13, 0x61, 0xc6, 0xc6, 0xe8, 0xec, 0x76, 0x49,
	0xaf, 0x15, 0xb4, 0xac, 0x72, 0xc6, 0xc6, 0xd8, 0xf9, 0x46, 0xe0, 0x70, 0xb3, 0x1a, 0xa5, 0xb0,
	0x67, 0xdd, 0xc4, 0xba, 0xed, 0x37, 0x3d, 0x86, 0x56, 0x3a, 0x11, 0x6b, 0x98, 0x66, 0x21, 0x14,
	0x14, 0x7a, 0x0f, 0x0e, 0xe2, 0x21, 0xc6, 0xa3, 0x5c, 0xf2, 0xcc, 0x84, 0x23, 0x9c, 0x3a, 0x35,
	0xeb, 0xd8, 0x5f, 0xa9, 0xaf, 0x70, 0x4a, 0xef, 0xc3, 0xa1, 0xcc, 0x0d, 0x97, 0x19, 0x13, 0xa1,
	0x42, 0x6d, 0xa4, 0x42, 0x67, 0xaf, 0x4b, 0x7a, 0xcd, 0xa0, 0x5d, 0xea, 0xc1, 0x42, 0xee, 0x7c,
	0x27, 0x70, 0x74, 0xe5, 0x2c, 0xe8, 0x33, 0xb8, 0x2d, 0x15, 0x1f, 0xf0, 0x02, 0x52, 0xce, 0x3b,
	0x5c, 0xef, 0x7e, 0xb3, 0x34, 0x94, 0xe9, 0xb3, 0xc5, 0x2c, 0x8e, 0xa1, 0x65, 0xd7, 0x54, 0xed,
	0x50, 0x08, 0xb6, 0xc3, 0x23, 0x38, 0x5a, 0xdb, 0xe1, 0x25, 0xb3, 0x66, 0x99, 0xb4, 0xba, 0x82,
	0x05, 0xef, 0xc5, 0x87, 0x9f, 0x33, 0x97, 0x5c, 0xcc, 0x5c, 0xf2, 0x67, 0xe6, 0x92, 0xaf, 0x73,
	0x77, 0xe7, 0x62, 0xee, 0xee, 0xfc, 0x9e, 0xbb, 0x3b, 0xef, 0xfa, 0x03, 0x6e, 0x86, 0x93, 0xc8,
	0x8b, 0xe5, 0xd8, 0xef, 0xe7, 0xb9, 0xe0, 0x29, 0x47, 0xe5, 0x0f, 0xe4, 0x83, 0xca, 0x0b, 0x35,
	0xd3, 0x1c, 0xb5, 0xff, 0xbf, 0x27, 0xfb, 0x97, 0x90, 0xa8, 0x61, 0x7f, 0x9e, 0xfc, 0x0b, 0x00,
	0x00, 0xff, 0xff, 0xdd, 0xf6, 0x36, 0x06, 0xd8, 0x03, 0x00, 0x00,
}

func (m *TrackableObjectGraph) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalTo(dAtA)
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *TrackableObjectGraph) MarshalTo(dAtA []byte) (int, error) {
	var i int
	_ = i
	var l int
	_ = l
	if len(m.Nodes) > 0 {
		for _, msg := range m.Nodes {
			dAtA[i] = 0xa
			i++
			i = encodeVarintTrackableObjectGraph(dAtA, i, uint64(msg.Size()))
			n, err := msg.MarshalTo(dAtA[i:])
			if err != nil {
				return 0, err
			}
			i += n
		}
	}
	return i, nil
}

func (m *TrackableObjectGraph_TrackableObject) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalTo(dAtA)
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *TrackableObjectGraph_TrackableObject) MarshalTo(dAtA []byte) (int, error) {
	var i int
	_ = i
	var l int
	_ = l
	if len(m.Children) > 0 {
		for _, msg := range m.Children {
			dAtA[i] = 0xa
			i++
			i = encodeVarintTrackableObjectGraph(dAtA, i, uint64(msg.Size()))
			n, err := msg.MarshalTo(dAtA[i:])
			if err != nil {
				return 0, err
			}
			i += n
		}
	}
	if len(m.Attributes) > 0 {
		for _, msg := range m.Attributes {
			dAtA[i] = 0x12
			i++
			i = encodeVarintTrackableObjectGraph(dAtA, i, uint64(msg.Size()))
			n, err := msg.MarshalTo(dAtA[i:])
			if err != nil {
				return 0, err
			}
			i += n
		}
	}
	if len(m.SlotVariables) > 0 {
		for _, msg := range m.SlotVariables {
			dAtA[i] = 0x1a
			i++
			i = encodeVarintTrackableObjectGraph(dAtA, i, uint64(msg.Size()))
			n, err := msg.MarshalTo(dAtA[i:])
			if err != nil {
				return 0, err
			}
			i += n
		}
	}
	return i, nil
}

func (m *TrackableObjectGraph_TrackableObject_ObjectReference) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalTo(dAtA)
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *TrackableObjectGraph_TrackableObject_ObjectReference) MarshalTo(dAtA []byte) (int, error) {
	var i int
	_ = i
	var l int
	_ = l
	if m.NodeId != 0 {
		dAtA[i] = 0x8
		i++
		i = encodeVarintTrackableObjectGraph(dAtA, i, uint64(m.NodeId))
	}
	if len(m.LocalName) > 0 {
		dAtA[i] = 0x12
		i++
		i = encodeVarintTrackableObjectGraph(dAtA, i, uint64(len(m.LocalName)))
		i += copy(dAtA[i:], m.LocalName)
	}
	return i, nil
}

func (m *TrackableObjectGraph_TrackableObject_SerializedTensor) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalTo(dAtA)
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *TrackableObjectGraph_TrackableObject_SerializedTensor) MarshalTo(dAtA []byte) (int, error) {
	var i int
	_ = i
	var l int
	_ = l
	if len(m.Name) > 0 {
		dAtA[i] = 0xa
		i++
		i = encodeVarintTrackableObjectGraph(dAtA, i, uint64(len(m.Name)))
		i += copy(dAtA[i:], m.Name)
	}
	if len(m.FullName) > 0 {
		dAtA[i] = 0x12
		i++
		i = encodeVarintTrackableObjectGraph(dAtA, i, uint64(len(m.FullName)))
		i += copy(dAtA[i:], m.FullName)
	}
	if len(m.CheckpointKey) > 0 {
		dAtA[i] = 0x1a
		i++
		i = encodeVarintTrackableObjectGraph(dAtA, i, uint64(len(m.CheckpointKey)))
		i += copy(dAtA[i:], m.CheckpointKey)
	}
	if m.OptionalRestore {
		dAtA[i] = 0x20
		i++
		if m.OptionalRestore {
			dAtA[i] = 1
		} else {
			dAtA[i] = 0
		}
		i++
	}
	return i, nil
}

func (m *TrackableObjectGraph_TrackableObject_SlotVariableReference) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalTo(dAtA)
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *TrackableObjectGraph_TrackableObject_SlotVariableReference) MarshalTo(dAtA []byte) (int, error) {
	var i int
	_ = i
	var l int
	_ = l
	if m.OriginalVariableNodeId != 0 {
		dAtA[i] = 0x8
		i++
		i = encodeVarintTrackableObjectGraph(dAtA, i, uint64(m.OriginalVariableNodeId))
	}
	if len(m.SlotName) > 0 {
		dAtA[i] = 0x12
		i++
		i = encodeVarintTrackableObjectGraph(dAtA, i, uint64(len(m.SlotName)))
		i += copy(dAtA[i:], m.SlotName)
	}
	if m.SlotVariableNodeId != 0 {
		dAtA[i] = 0x18
		i++
		i = encodeVarintTrackableObjectGraph(dAtA, i, uint64(m.SlotVariableNodeId))
	}
	return i, nil
}

func encodeVarintTrackableObjectGraph(dAtA []byte, offset int, v uint64) int {
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return offset + 1
}
func (m *TrackableObjectGraph) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if len(m.Nodes) > 0 {
		for _, e := range m.Nodes {
			l = e.Size()
			n += 1 + l + sovTrackableObjectGraph(uint64(l))
		}
	}
	return n
}

func (m *TrackableObjectGraph_TrackableObject) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if len(m.Children) > 0 {
		for _, e := range m.Children {
			l = e.Size()
			n += 1 + l + sovTrackableObjectGraph(uint64(l))
		}
	}
	if len(m.Attributes) > 0 {
		for _, e := range m.Attributes {
			l = e.Size()
			n += 1 + l + sovTrackableObjectGraph(uint64(l))
		}
	}
	if len(m.SlotVariables) > 0 {
		for _, e := range m.SlotVariables {
			l = e.Size()
			n += 1 + l + sovTrackableObjectGraph(uint64(l))
		}
	}
	return n
}

func (m *TrackableObjectGraph_TrackableObject_ObjectReference) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if m.NodeId != 0 {
		n += 1 + sovTrackableObjectGraph(uint64(m.NodeId))
	}
	l = len(m.LocalName)
	if l > 0 {
		n += 1 + l + sovTrackableObjectGraph(uint64(l))
	}
	return n
}

func (m *TrackableObjectGraph_TrackableObject_SerializedTensor) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	l = len(m.Name)
	if l > 0 {
		n += 1 + l + sovTrackableObjectGraph(uint64(l))
	}
	l = len(m.FullName)
	if l > 0 {
		n += 1 + l + sovTrackableObjectGraph(uint64(l))
	}
	l = len(m.CheckpointKey)
	if l > 0 {
		n += 1 + l + sovTrackableObjectGraph(uint64(l))
	}
	if m.OptionalRestore {
		n += 2
	}
	return n
}

func (m *TrackableObjectGraph_TrackableObject_SlotVariableReference) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if m.OriginalVariableNodeId != 0 {
		n += 1 + sovTrackableObjectGraph(uint64(m.OriginalVariableNodeId))
	}
	l = len(m.SlotName)
	if l > 0 {
		n += 1 + l + sovTrackableObjectGraph(uint64(l))
	}
	if m.SlotVariableNodeId != 0 {
		n += 1 + sovTrackableObjectGraph(uint64(m.SlotVariableNodeId))
	}
	return n
}

func sovTrackableObjectGraph(x uint64) (n int) {
	for {
		n++
		x >>= 7
		if x == 0 {
			break
		}
	}
	return n
}
func sozTrackableObjectGraph(x uint64) (n int) {
	return sovTrackableObjectGraph(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (m *TrackableObjectGraph) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowTrackableObjectGraph
			}
			if iNdEx >= l {
				return io.ErrUnexpectedEOF
			}
			b := dAtA[iNdEx]
			iNdEx++
			wire |= (uint64(b) & 0x7F) << shift
			if b < 0x80 {
				break
			}
		}
		fieldNum := int32(wire >> 3)
		wireType := int(wire & 0x7)
		if wireType == 4 {
			return fmt.Errorf("proto: TrackableObjectGraph: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: TrackableObjectGraph: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Nodes", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowTrackableObjectGraph
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				msglen |= (int(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			if msglen < 0 {
				return ErrInvalidLengthTrackableObjectGraph
			}
			postIndex := iNdEx + msglen
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.Nodes = append(m.Nodes, &TrackableObjectGraph_TrackableObject{})
			if err := m.Nodes[len(m.Nodes)-1].Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipTrackableObjectGraph(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if skippy < 0 {
				return ErrInvalidLengthTrackableObjectGraph
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
func (m *TrackableObjectGraph_TrackableObject) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowTrackableObjectGraph
			}
			if iNdEx >= l {
				return io.ErrUnexpectedEOF
			}
			b := dAtA[iNdEx]
			iNdEx++
			wire |= (uint64(b) & 0x7F) << shift
			if b < 0x80 {
				break
			}
		}
		fieldNum := int32(wire >> 3)
		wireType := int(wire & 0x7)
		if wireType == 4 {
			return fmt.Errorf("proto: TrackableObject: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: TrackableObject: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Children", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowTrackableObjectGraph
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				msglen |= (int(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			if msglen < 0 {
				return ErrInvalidLengthTrackableObjectGraph
			}
			postIndex := iNdEx + msglen
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.Children = append(m.Children, &TrackableObjectGraph_TrackableObject_ObjectReference{})
			if err := m.Children[len(m.Children)-1].Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		case 2:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Attributes", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowTrackableObjectGraph
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				msglen |= (int(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			if msglen < 0 {
				return ErrInvalidLengthTrackableObjectGraph
			}
			postIndex := iNdEx + msglen
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.Attributes = append(m.Attributes, &TrackableObjectGraph_TrackableObject_SerializedTensor{})
			if err := m.Attributes[len(m.Attributes)-1].Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		case 3:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field SlotVariables", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowTrackableObjectGraph
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				msglen |= (int(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			if msglen < 0 {
				return ErrInvalidLengthTrackableObjectGraph
			}
			postIndex := iNdEx + msglen
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.SlotVariables = append(m.SlotVariables, &TrackableObjectGraph_TrackableObject_SlotVariableReference{})
			if err := m.SlotVariables[len(m.SlotVariables)-1].Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipTrackableObjectGraph(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if skippy < 0 {
				return ErrInvalidLengthTrackableObjectGraph
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
func (m *TrackableObjectGraph_TrackableObject_ObjectReference) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowTrackableObjectGraph
			}
			if iNdEx >= l {
				return io.ErrUnexpectedEOF
			}
			b := dAtA[iNdEx]
			iNdEx++
			wire |= (uint64(b) & 0x7F) << shift
			if b < 0x80 {
				break
			}
		}
		fieldNum := int32(wire >> 3)
		wireType := int(wire & 0x7)
		if wireType == 4 {
			return fmt.Errorf("proto: ObjectReference: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: ObjectReference: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field NodeId", wireType)
			}
			m.NodeId = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowTrackableObjectGraph
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.NodeId |= (int32(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		case 2:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field LocalName", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowTrackableObjectGraph
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				stringLen |= (uint64(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			intStringLen := int(stringLen)
			if intStringLen < 0 {
				return ErrInvalidLengthTrackableObjectGraph
			}
			postIndex := iNdEx + intStringLen
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.LocalName = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipTrackableObjectGraph(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if skippy < 0 {
				return ErrInvalidLengthTrackableObjectGraph
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
func (m *TrackableObjectGraph_TrackableObject_SerializedTensor) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowTrackableObjectGraph
			}
			if iNdEx >= l {
				return io.ErrUnexpectedEOF
			}
			b := dAtA[iNdEx]
			iNdEx++
			wire |= (uint64(b) & 0x7F) << shift
			if b < 0x80 {
				break
			}
		}
		fieldNum := int32(wire >> 3)
		wireType := int(wire & 0x7)
		if wireType == 4 {
			return fmt.Errorf("proto: SerializedTensor: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: SerializedTensor: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Name", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowTrackableObjectGraph
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				stringLen |= (uint64(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			intStringLen := int(stringLen)
			if intStringLen < 0 {
				return ErrInvalidLengthTrackableObjectGraph
			}
			postIndex := iNdEx + intStringLen
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.Name = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		case 2:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field FullName", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowTrackableObjectGraph
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				stringLen |= (uint64(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			intStringLen := int(stringLen)
			if intStringLen < 0 {
				return ErrInvalidLengthTrackableObjectGraph
			}
			postIndex := iNdEx + intStringLen
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.FullName = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		case 3:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field CheckpointKey", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowTrackableObjectGraph
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				stringLen |= (uint64(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			intStringLen := int(stringLen)
			if intStringLen < 0 {
				return ErrInvalidLengthTrackableObjectGraph
			}
			postIndex := iNdEx + intStringLen
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.CheckpointKey = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		case 4:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field OptionalRestore", wireType)
			}
			var v int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowTrackableObjectGraph
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				v |= (int(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			m.OptionalRestore = bool(v != 0)
		default:
			iNdEx = preIndex
			skippy, err := skipTrackableObjectGraph(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if skippy < 0 {
				return ErrInvalidLengthTrackableObjectGraph
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
func (m *TrackableObjectGraph_TrackableObject_SlotVariableReference) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowTrackableObjectGraph
			}
			if iNdEx >= l {
				return io.ErrUnexpectedEOF
			}
			b := dAtA[iNdEx]
			iNdEx++
			wire |= (uint64(b) & 0x7F) << shift
			if b < 0x80 {
				break
			}
		}
		fieldNum := int32(wire >> 3)
		wireType := int(wire & 0x7)
		if wireType == 4 {
			return fmt.Errorf("proto: SlotVariableReference: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: SlotVariableReference: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field OriginalVariableNodeId", wireType)
			}
			m.OriginalVariableNodeId = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowTrackableObjectGraph
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.OriginalVariableNodeId |= (int32(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		case 2:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field SlotName", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowTrackableObjectGraph
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				stringLen |= (uint64(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			intStringLen := int(stringLen)
			if intStringLen < 0 {
				return ErrInvalidLengthTrackableObjectGraph
			}
			postIndex := iNdEx + intStringLen
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.SlotName = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		case 3:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field SlotVariableNodeId", wireType)
			}
			m.SlotVariableNodeId = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowTrackableObjectGraph
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.SlotVariableNodeId |= (int32(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		default:
			iNdEx = preIndex
			skippy, err := skipTrackableObjectGraph(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if skippy < 0 {
				return ErrInvalidLengthTrackableObjectGraph
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
func skipTrackableObjectGraph(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowTrackableObjectGraph
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
					return 0, ErrIntOverflowTrackableObjectGraph
				}
				if iNdEx >= l {
					return 0, io.ErrUnexpectedEOF
				}
				iNdEx++
				if dAtA[iNdEx-1] < 0x80 {
					break
				}
			}
			return iNdEx, nil
		case 1:
			iNdEx += 8
			return iNdEx, nil
		case 2:
			var length int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return 0, ErrIntOverflowTrackableObjectGraph
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
			iNdEx += length
			if length < 0 {
				return 0, ErrInvalidLengthTrackableObjectGraph
			}
			return iNdEx, nil
		case 3:
			for {
				var innerWire uint64
				var start int = iNdEx
				for shift := uint(0); ; shift += 7 {
					if shift >= 64 {
						return 0, ErrIntOverflowTrackableObjectGraph
					}
					if iNdEx >= l {
						return 0, io.ErrUnexpectedEOF
					}
					b := dAtA[iNdEx]
					iNdEx++
					innerWire |= (uint64(b) & 0x7F) << shift
					if b < 0x80 {
						break
					}
				}
				innerWireType := int(innerWire & 0x7)
				if innerWireType == 4 {
					break
				}
				next, err := skipTrackableObjectGraph(dAtA[start:])
				if err != nil {
					return 0, err
				}
				iNdEx = start + next
			}
			return iNdEx, nil
		case 4:
			return iNdEx, nil
		case 5:
			iNdEx += 4
			return iNdEx, nil
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
	}
	panic("unreachable")
}

var (
	ErrInvalidLengthTrackableObjectGraph = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowTrackableObjectGraph   = fmt.Errorf("proto: integer overflow")
)