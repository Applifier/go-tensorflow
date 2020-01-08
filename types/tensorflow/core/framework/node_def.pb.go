// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: tensorflow/core/framework/node_def.proto

package framework

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

type NodeDef struct {
	// The name given to this operator. Used for naming inputs,
	// logging, visualization, etc.  Unique within a single GraphDef.
	// Must match the regexp "[A-Za-z0-9.][A-Za-z0-9_>./]*".
	Name string `protobuf:"bytes,1,opt,name=name,proto3" json:"name,omitempty"`
	// The operation name.  There may be custom parameters in attrs.
	// Op names starting with an underscore are reserved for internal use.
	Op string `protobuf:"bytes,2,opt,name=op,proto3" json:"op,omitempty"`
	// Each input is "node:src_output" with "node" being a string name and
	// "src_output" indicating which output tensor to use from "node". If
	// "src_output" is 0 the ":0" suffix can be omitted.  Regular inputs
	// may optionally be followed by control inputs that have the format
	// "^node".
	Input []string `protobuf:"bytes,3,rep,name=input,proto3" json:"input,omitempty"`
	// A (possibly partial) specification for the device on which this
	// node should be placed.
	// The expected syntax for this string is as follows:
	//
	// DEVICE_SPEC ::= PARTIAL_SPEC
	//
	// PARTIAL_SPEC ::= ("/" CONSTRAINT) *
	// CONSTRAINT ::= ("job:" JOB_NAME)
	//              | ("replica:" [1-9][0-9]*)
	//              | ("task:" [1-9][0-9]*)
	//              | ("device:" [A-Za-z]* ":" ([1-9][0-9]* | "*") )
	//
	// Valid values for this string include:
	// * "/job:worker/replica:0/task:1/device:GPU:3"  (full specification)
	// * "/job:worker/device:GPU:3"                   (partial specification)
	// * ""                                    (no specification)
	//
	// If the constraints do not resolve to a single device (or if this
	// field is empty or not present), the runtime will attempt to
	// choose a device automatically.
	Device string `protobuf:"bytes,4,opt,name=device,proto3" json:"device,omitempty"`
	// Operation-specific graph-construction-time configuration.
	// Note that this should include all attrs defined in the
	// corresponding OpDef, including those with a value matching
	// the default -- this allows the default to change and makes
	// NodeDefs easier to interpret on their own.  However, if
	// an attr with a default is not specified in this list, the
	// default will be used.
	// The "names" (keys) must match the regexp "[a-z][a-z0-9_]+" (and
	// one of the names from the corresponding OpDef's attr field).
	// The values must have a type matching the corresponding OpDef
	// attr's type field.
	// TODO(josh11b): Add some examples here showing best practices.
	Attr map[string]*AttrValue `protobuf:"bytes,5,rep,name=attr,proto3" json:"attr,omitempty" protobuf_key:"bytes,1,opt,name=key,proto3" protobuf_val:"bytes,2,opt,name=value,proto3"`
	// This stores debug information associated with the node.
	ExperimentalDebugInfo *NodeDef_ExperimentalDebugInfo `protobuf:"bytes,6,opt,name=experimental_debug_info,json=experimentalDebugInfo,proto3" json:"experimental_debug_info,omitempty"`
}

func (m *NodeDef) Reset()         { *m = NodeDef{} }
func (m *NodeDef) String() string { return proto.CompactTextString(m) }
func (*NodeDef) ProtoMessage()    {}
func (*NodeDef) Descriptor() ([]byte, []int) {
	return fileDescriptor_b34b3b836a96140b, []int{0}
}
func (m *NodeDef) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *NodeDef) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_NodeDef.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalTo(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *NodeDef) XXX_Merge(src proto.Message) {
	xxx_messageInfo_NodeDef.Merge(m, src)
}
func (m *NodeDef) XXX_Size() int {
	return m.Size()
}
func (m *NodeDef) XXX_DiscardUnknown() {
	xxx_messageInfo_NodeDef.DiscardUnknown(m)
}

var xxx_messageInfo_NodeDef proto.InternalMessageInfo

func (m *NodeDef) GetName() string {
	if m != nil {
		return m.Name
	}
	return ""
}

func (m *NodeDef) GetOp() string {
	if m != nil {
		return m.Op
	}
	return ""
}

func (m *NodeDef) GetInput() []string {
	if m != nil {
		return m.Input
	}
	return nil
}

func (m *NodeDef) GetDevice() string {
	if m != nil {
		return m.Device
	}
	return ""
}

func (m *NodeDef) GetAttr() map[string]*AttrValue {
	if m != nil {
		return m.Attr
	}
	return nil
}

func (m *NodeDef) GetExperimentalDebugInfo() *NodeDef_ExperimentalDebugInfo {
	if m != nil {
		return m.ExperimentalDebugInfo
	}
	return nil
}

type NodeDef_ExperimentalDebugInfo struct {
	// Opaque string inserted into error messages created by the runtime.
	//
	// This is intended to store the list of names of the nodes from the
	// original graph that this node was derived. For example if this node, say
	// C, was result of a fusion of 2 nodes A and B, then 'original_node' would
	// be {A, B}. This information can be used to map errors originating at the
	// current node to some top level source code.
	OriginalNodeNames []string `protobuf:"bytes,1,rep,name=original_node_names,json=originalNodeNames,proto3" json:"original_node_names,omitempty"`
	// This is intended to store the list of names of the functions from the
	// original graph that this node was derived. For example if this node, say
	// C, was result of a fusion of node A in function FA and node B in function
	// FB, then `original_funcs` would be {FA, FB}. If the node is in the top
	// level graph, the `original_func` is empty. This information, with the
	// `original_node_names` can be used to map errors originating at the
	// current ndoe to some top level source code.
	OriginalFuncNames []string `protobuf:"bytes,2,rep,name=original_func_names,json=originalFuncNames,proto3" json:"original_func_names,omitempty"`
}

func (m *NodeDef_ExperimentalDebugInfo) Reset()         { *m = NodeDef_ExperimentalDebugInfo{} }
func (m *NodeDef_ExperimentalDebugInfo) String() string { return proto.CompactTextString(m) }
func (*NodeDef_ExperimentalDebugInfo) ProtoMessage()    {}
func (*NodeDef_ExperimentalDebugInfo) Descriptor() ([]byte, []int) {
	return fileDescriptor_b34b3b836a96140b, []int{0, 1}
}
func (m *NodeDef_ExperimentalDebugInfo) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *NodeDef_ExperimentalDebugInfo) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_NodeDef_ExperimentalDebugInfo.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalTo(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *NodeDef_ExperimentalDebugInfo) XXX_Merge(src proto.Message) {
	xxx_messageInfo_NodeDef_ExperimentalDebugInfo.Merge(m, src)
}
func (m *NodeDef_ExperimentalDebugInfo) XXX_Size() int {
	return m.Size()
}
func (m *NodeDef_ExperimentalDebugInfo) XXX_DiscardUnknown() {
	xxx_messageInfo_NodeDef_ExperimentalDebugInfo.DiscardUnknown(m)
}

var xxx_messageInfo_NodeDef_ExperimentalDebugInfo proto.InternalMessageInfo

func (m *NodeDef_ExperimentalDebugInfo) GetOriginalNodeNames() []string {
	if m != nil {
		return m.OriginalNodeNames
	}
	return nil
}

func (m *NodeDef_ExperimentalDebugInfo) GetOriginalFuncNames() []string {
	if m != nil {
		return m.OriginalFuncNames
	}
	return nil
}

func init() {
	proto.RegisterType((*NodeDef)(nil), "tensorflow.NodeDef")
	proto.RegisterMapType((map[string]*AttrValue)(nil), "tensorflow.NodeDef.AttrEntry")
	proto.RegisterType((*NodeDef_ExperimentalDebugInfo)(nil), "tensorflow.NodeDef.ExperimentalDebugInfo")
}

func init() {
	proto.RegisterFile("tensorflow/core/framework/node_def.proto", fileDescriptor_b34b3b836a96140b)
}

var fileDescriptor_b34b3b836a96140b = []byte{
	// 421 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x74, 0x92, 0x31, 0x8f, 0xd3, 0x30,
	0x14, 0xc7, 0xeb, 0xa4, 0x2d, 0xaa, 0x2b, 0x21, 0x30, 0x14, 0xa2, 0x4a, 0x44, 0x15, 0x53, 0x00,
	0x91, 0x88, 0xb2, 0x20, 0xb6, 0xab, 0xee, 0x90, 0x58, 0xaa, 0x53, 0x06, 0x06, 0x96, 0x28, 0x4d,
	0x9e, 0x83, 0x75, 0xa9, 0x9f, 0xe5, 0x3a, 0x57, 0xfa, 0x05, 0x98, 0xf9, 0x3c, 0x7c, 0x02, 0xc6,
	0x1b, 0x19, 0x51, 0xfb, 0x25, 0x18, 0x91, 0xdd, 0xde, 0x5d, 0x4e, 0xea, 0x6d, 0xb6, 0xdf, 0xef,
	0xef, 0xbf, 0xdf, 0xf3, 0x9f, 0x46, 0x06, 0xe4, 0x0a, 0x35, 0xaf, 0x71, 0x9d, 0x14, 0xa8, 0x21,
	0xe1, 0x3a, 0x5f, 0xc2, 0x1a, 0xf5, 0x45, 0x22, 0xb1, 0x84, 0xac, 0x04, 0x1e, 0x2b, 0x8d, 0x06,
	0x19, 0xbd, 0x25, 0xc7, 0xaf, 0xef, 0x57, 0xe5, 0xc6, 0xe8, 0xec, 0x32, 0xaf, 0x1b, 0xd8, 0xeb,
	0x5e, 0xfe, 0xf2, 0xe9, 0x83, 0x39, 0x96, 0x70, 0x0a, 0x9c, 0x31, 0xda, 0x95, 0xf9, 0x12, 0x02,
	0x32, 0x21, 0xd1, 0x20, 0x75, 0x6b, 0xf6, 0x90, 0x7a, 0xa8, 0x02, 0xcf, 0x9d, 0x78, 0xa8, 0xd8,
	0x53, 0xda, 0x13, 0x52, 0x35, 0x26, 0xf0, 0x27, 0x7e, 0x34, 0x48, 0xf7, 0x1b, 0xf6, 0x8c, 0xf6,
	0x4b, 0xb8, 0x14, 0x05, 0x04, 0x5d, 0x47, 0x1e, 0x76, 0xec, 0x1d, 0xed, 0x5a, 0xc7, 0xa0, 0x37,
	0xf1, 0xa3, 0xe1, 0xf4, 0x45, 0x7c, 0xfb, 0xb0, 0xf8, 0x60, 0x1a, 0x9f, 0x18, 0xa3, 0xcf, 0xa4,
	0xd1, 0x9b, 0xd4, 0xa1, 0x2c, 0xa7, 0xcf, 0xe1, 0xbb, 0x02, 0x2d, 0x96, 0x20, 0x4d, 0x5e, 0x67,
	0x25, 0x2c, 0x9a, 0x2a, 0x13, 0x92, 0x63, 0xd0, 0x9f, 0x90, 0x68, 0x38, 0x7d, 0x75, 0xec, 0x96,
	0xb3, 0x96, 0xe4, 0xd4, 0x2a, 0x3e, 0x4b, 0x8e, 0xe9, 0x08, 0x8e, 0x1d, 0x8f, 0xe7, 0x74, 0x70,
	0xe3, 0xca, 0x1e, 0x51, 0xff, 0x02, 0x36, 0x87, 0x9e, 0xed, 0x92, 0xbd, 0xa1, 0x3d, 0x37, 0x21,
	0xd7, 0xf5, 0x70, 0x3a, 0x6a, 0xfb, 0x59, 0xdd, 0x17, 0x5b, 0x4c, 0xf7, 0xcc, 0x47, 0xef, 0x03,
	0x19, 0xaf, 0xe9, 0xe8, 0xa8, 0x3f, 0x8b, 0xe9, 0x13, 0xd4, 0xa2, 0x12, 0x32, 0xaf, 0x33, 0xf7,
	0x5f, 0x76, 0xa4, 0xab, 0x80, 0xb8, 0xd1, 0x3d, 0xbe, 0x2e, 0xd9, 0x1e, 0xe6, 0xb6, 0x70, 0x87,
	0xe7, 0x8d, 0x2c, 0x0e, 0xbc, 0x77, 0x97, 0xff, 0xd4, 0xc8, 0xc2, 0xf1, 0xb3, 0x1f, 0xe4, 0xf7,
	0x36, 0x24, 0x57, 0xdb, 0x90, 0xfc, 0xdd, 0x86, 0xe4, 0xe7, 0x2e, 0xec, 0x5c, 0xed, 0xc2, 0xce,
	0x9f, 0x5d, 0xd8, 0xa1, 0x01, 0xea, 0xaa, 0xfd, 0xf0, 0x9b, 0x08, 0xcc, 0x06, 0xd6, 0xef, 0xdc,
	0x7e, 0xfe, 0x39, 0xf9, 0x3a, 0xab, 0x84, 0xf9, 0xd6, 0x2c, 0xe2, 0x02, 0x97, 0xc9, 0x89, 0x52,
	0xb5, 0xe0, 0x02, 0x74, 0x52, 0xe1, 0xdb, 0x56, 0x84, 0xcc, 0x46, 0xc1, 0x2a, 0xb9, 0x37, 0x53,
	0xff, 0x08, 0x59, 0xf4, 0x5d, 0x98, 0xde, 0xff, 0x0f, 0x00, 0x00, 0xff, 0xff, 0x79, 0xd9, 0x46,
	0xc7, 0xb0, 0x02, 0x00, 0x00,
}

func (m *NodeDef) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalTo(dAtA)
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *NodeDef) MarshalTo(dAtA []byte) (int, error) {
	var i int
	_ = i
	var l int
	_ = l
	if len(m.Name) > 0 {
		dAtA[i] = 0xa
		i++
		i = encodeVarintNodeDef(dAtA, i, uint64(len(m.Name)))
		i += copy(dAtA[i:], m.Name)
	}
	if len(m.Op) > 0 {
		dAtA[i] = 0x12
		i++
		i = encodeVarintNodeDef(dAtA, i, uint64(len(m.Op)))
		i += copy(dAtA[i:], m.Op)
	}
	if len(m.Input) > 0 {
		for _, s := range m.Input {
			dAtA[i] = 0x1a
			i++
			l = len(s)
			for l >= 1<<7 {
				dAtA[i] = uint8(uint64(l)&0x7f | 0x80)
				l >>= 7
				i++
			}
			dAtA[i] = uint8(l)
			i++
			i += copy(dAtA[i:], s)
		}
	}
	if len(m.Device) > 0 {
		dAtA[i] = 0x22
		i++
		i = encodeVarintNodeDef(dAtA, i, uint64(len(m.Device)))
		i += copy(dAtA[i:], m.Device)
	}
	if len(m.Attr) > 0 {
		for k, _ := range m.Attr {
			dAtA[i] = 0x2a
			i++
			v := m.Attr[k]
			msgSize := 0
			if v != nil {
				msgSize = v.Size()
				msgSize += 1 + sovNodeDef(uint64(msgSize))
			}
			mapSize := 1 + len(k) + sovNodeDef(uint64(len(k))) + msgSize
			i = encodeVarintNodeDef(dAtA, i, uint64(mapSize))
			dAtA[i] = 0xa
			i++
			i = encodeVarintNodeDef(dAtA, i, uint64(len(k)))
			i += copy(dAtA[i:], k)
			if v != nil {
				dAtA[i] = 0x12
				i++
				i = encodeVarintNodeDef(dAtA, i, uint64(v.Size()))
				n1, err := v.MarshalTo(dAtA[i:])
				if err != nil {
					return 0, err
				}
				i += n1
			}
		}
	}
	if m.ExperimentalDebugInfo != nil {
		dAtA[i] = 0x32
		i++
		i = encodeVarintNodeDef(dAtA, i, uint64(m.ExperimentalDebugInfo.Size()))
		n2, err := m.ExperimentalDebugInfo.MarshalTo(dAtA[i:])
		if err != nil {
			return 0, err
		}
		i += n2
	}
	return i, nil
}

func (m *NodeDef_ExperimentalDebugInfo) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalTo(dAtA)
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *NodeDef_ExperimentalDebugInfo) MarshalTo(dAtA []byte) (int, error) {
	var i int
	_ = i
	var l int
	_ = l
	if len(m.OriginalNodeNames) > 0 {
		for _, s := range m.OriginalNodeNames {
			dAtA[i] = 0xa
			i++
			l = len(s)
			for l >= 1<<7 {
				dAtA[i] = uint8(uint64(l)&0x7f | 0x80)
				l >>= 7
				i++
			}
			dAtA[i] = uint8(l)
			i++
			i += copy(dAtA[i:], s)
		}
	}
	if len(m.OriginalFuncNames) > 0 {
		for _, s := range m.OriginalFuncNames {
			dAtA[i] = 0x12
			i++
			l = len(s)
			for l >= 1<<7 {
				dAtA[i] = uint8(uint64(l)&0x7f | 0x80)
				l >>= 7
				i++
			}
			dAtA[i] = uint8(l)
			i++
			i += copy(dAtA[i:], s)
		}
	}
	return i, nil
}

func encodeVarintNodeDef(dAtA []byte, offset int, v uint64) int {
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return offset + 1
}
func (m *NodeDef) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	l = len(m.Name)
	if l > 0 {
		n += 1 + l + sovNodeDef(uint64(l))
	}
	l = len(m.Op)
	if l > 0 {
		n += 1 + l + sovNodeDef(uint64(l))
	}
	if len(m.Input) > 0 {
		for _, s := range m.Input {
			l = len(s)
			n += 1 + l + sovNodeDef(uint64(l))
		}
	}
	l = len(m.Device)
	if l > 0 {
		n += 1 + l + sovNodeDef(uint64(l))
	}
	if len(m.Attr) > 0 {
		for k, v := range m.Attr {
			_ = k
			_ = v
			l = 0
			if v != nil {
				l = v.Size()
				l += 1 + sovNodeDef(uint64(l))
			}
			mapEntrySize := 1 + len(k) + sovNodeDef(uint64(len(k))) + l
			n += mapEntrySize + 1 + sovNodeDef(uint64(mapEntrySize))
		}
	}
	if m.ExperimentalDebugInfo != nil {
		l = m.ExperimentalDebugInfo.Size()
		n += 1 + l + sovNodeDef(uint64(l))
	}
	return n
}

func (m *NodeDef_ExperimentalDebugInfo) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if len(m.OriginalNodeNames) > 0 {
		for _, s := range m.OriginalNodeNames {
			l = len(s)
			n += 1 + l + sovNodeDef(uint64(l))
		}
	}
	if len(m.OriginalFuncNames) > 0 {
		for _, s := range m.OriginalFuncNames {
			l = len(s)
			n += 1 + l + sovNodeDef(uint64(l))
		}
	}
	return n
}

func sovNodeDef(x uint64) (n int) {
	for {
		n++
		x >>= 7
		if x == 0 {
			break
		}
	}
	return n
}
func sozNodeDef(x uint64) (n int) {
	return sovNodeDef(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (m *NodeDef) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowNodeDef
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
			return fmt.Errorf("proto: NodeDef: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: NodeDef: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Name", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowNodeDef
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
				return ErrInvalidLengthNodeDef
			}
			postIndex := iNdEx + intStringLen
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.Name = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		case 2:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Op", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowNodeDef
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
				return ErrInvalidLengthNodeDef
			}
			postIndex := iNdEx + intStringLen
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.Op = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		case 3:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Input", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowNodeDef
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
				return ErrInvalidLengthNodeDef
			}
			postIndex := iNdEx + intStringLen
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.Input = append(m.Input, string(dAtA[iNdEx:postIndex]))
			iNdEx = postIndex
		case 4:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Device", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowNodeDef
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
				return ErrInvalidLengthNodeDef
			}
			postIndex := iNdEx + intStringLen
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.Device = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		case 5:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Attr", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowNodeDef
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
				return ErrInvalidLengthNodeDef
			}
			postIndex := iNdEx + msglen
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			if m.Attr == nil {
				m.Attr = make(map[string]*AttrValue)
			}
			var mapkey string
			var mapvalue *AttrValue
			for iNdEx < postIndex {
				entryPreIndex := iNdEx
				var wire uint64
				for shift := uint(0); ; shift += 7 {
					if shift >= 64 {
						return ErrIntOverflowNodeDef
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
				if fieldNum == 1 {
					var stringLenmapkey uint64
					for shift := uint(0); ; shift += 7 {
						if shift >= 64 {
							return ErrIntOverflowNodeDef
						}
						if iNdEx >= l {
							return io.ErrUnexpectedEOF
						}
						b := dAtA[iNdEx]
						iNdEx++
						stringLenmapkey |= (uint64(b) & 0x7F) << shift
						if b < 0x80 {
							break
						}
					}
					intStringLenmapkey := int(stringLenmapkey)
					if intStringLenmapkey < 0 {
						return ErrInvalidLengthNodeDef
					}
					postStringIndexmapkey := iNdEx + intStringLenmapkey
					if postStringIndexmapkey > l {
						return io.ErrUnexpectedEOF
					}
					mapkey = string(dAtA[iNdEx:postStringIndexmapkey])
					iNdEx = postStringIndexmapkey
				} else if fieldNum == 2 {
					var mapmsglen int
					for shift := uint(0); ; shift += 7 {
						if shift >= 64 {
							return ErrIntOverflowNodeDef
						}
						if iNdEx >= l {
							return io.ErrUnexpectedEOF
						}
						b := dAtA[iNdEx]
						iNdEx++
						mapmsglen |= (int(b) & 0x7F) << shift
						if b < 0x80 {
							break
						}
					}
					if mapmsglen < 0 {
						return ErrInvalidLengthNodeDef
					}
					postmsgIndex := iNdEx + mapmsglen
					if mapmsglen < 0 {
						return ErrInvalidLengthNodeDef
					}
					if postmsgIndex > l {
						return io.ErrUnexpectedEOF
					}
					mapvalue = &AttrValue{}
					if err := mapvalue.Unmarshal(dAtA[iNdEx:postmsgIndex]); err != nil {
						return err
					}
					iNdEx = postmsgIndex
				} else {
					iNdEx = entryPreIndex
					skippy, err := skipNodeDef(dAtA[iNdEx:])
					if err != nil {
						return err
					}
					if skippy < 0 {
						return ErrInvalidLengthNodeDef
					}
					if (iNdEx + skippy) > postIndex {
						return io.ErrUnexpectedEOF
					}
					iNdEx += skippy
				}
			}
			m.Attr[mapkey] = mapvalue
			iNdEx = postIndex
		case 6:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field ExperimentalDebugInfo", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowNodeDef
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
				return ErrInvalidLengthNodeDef
			}
			postIndex := iNdEx + msglen
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			if m.ExperimentalDebugInfo == nil {
				m.ExperimentalDebugInfo = &NodeDef_ExperimentalDebugInfo{}
			}
			if err := m.ExperimentalDebugInfo.Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipNodeDef(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if skippy < 0 {
				return ErrInvalidLengthNodeDef
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
func (m *NodeDef_ExperimentalDebugInfo) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowNodeDef
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
			return fmt.Errorf("proto: ExperimentalDebugInfo: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: ExperimentalDebugInfo: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field OriginalNodeNames", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowNodeDef
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
				return ErrInvalidLengthNodeDef
			}
			postIndex := iNdEx + intStringLen
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.OriginalNodeNames = append(m.OriginalNodeNames, string(dAtA[iNdEx:postIndex]))
			iNdEx = postIndex
		case 2:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field OriginalFuncNames", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowNodeDef
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
				return ErrInvalidLengthNodeDef
			}
			postIndex := iNdEx + intStringLen
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.OriginalFuncNames = append(m.OriginalFuncNames, string(dAtA[iNdEx:postIndex]))
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipNodeDef(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if skippy < 0 {
				return ErrInvalidLengthNodeDef
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
func skipNodeDef(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowNodeDef
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
					return 0, ErrIntOverflowNodeDef
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
					return 0, ErrIntOverflowNodeDef
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
				return 0, ErrInvalidLengthNodeDef
			}
			return iNdEx, nil
		case 3:
			for {
				var innerWire uint64
				var start int = iNdEx
				for shift := uint(0); ; shift += 7 {
					if shift >= 64 {
						return 0, ErrIntOverflowNodeDef
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
				next, err := skipNodeDef(dAtA[start:])
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
	ErrInvalidLengthNodeDef = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowNodeDef   = fmt.Errorf("proto: integer overflow")
)
