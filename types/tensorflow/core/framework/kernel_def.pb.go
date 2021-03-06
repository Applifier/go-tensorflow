// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: tensorflow/core/framework/kernel_def.proto

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

type KernelDef struct {
	// Must match the name of an Op.
	Op string `protobuf:"bytes,1,opt,name=op,proto3" json:"op,omitempty"`
	// Type of device this kernel runs on.
	DeviceType string                      `protobuf:"bytes,2,opt,name=device_type,json=deviceType,proto3" json:"device_type,omitempty"`
	Constraint []*KernelDef_AttrConstraint `protobuf:"bytes,3,rep,name=constraint,proto3" json:"constraint,omitempty"`
	// Names of the Op's input_/output_args that reside in host memory
	// instead of device memory.
	HostMemoryArg []string `protobuf:"bytes,4,rep,name=host_memory_arg,json=hostMemoryArg,proto3" json:"host_memory_arg,omitempty"`
	// This allows experimental kernels to be registered for an op that
	// won't be used unless the user specifies a "_kernel" attr with
	// value matching this.
	Label string `protobuf:"bytes,5,opt,name=label,proto3" json:"label,omitempty"`
	// Prioritization of kernel amongst different devices. By default we assume
	// priority is 0. The higher the priority the better. By default (i.e. if
	// this is not set), we prefer GPU kernels over CPU.
	Priority int32 `protobuf:"varint,6,opt,name=priority,proto3" json:"priority,omitempty"`
}

func (m *KernelDef) Reset()         { *m = KernelDef{} }
func (m *KernelDef) String() string { return proto.CompactTextString(m) }
func (*KernelDef) ProtoMessage()    {}
func (*KernelDef) Descriptor() ([]byte, []int) {
	return fileDescriptor_18794e085ea7671a, []int{0}
}
func (m *KernelDef) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *KernelDef) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_KernelDef.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *KernelDef) XXX_Merge(src proto.Message) {
	xxx_messageInfo_KernelDef.Merge(m, src)
}
func (m *KernelDef) XXX_Size() int {
	return m.Size()
}
func (m *KernelDef) XXX_DiscardUnknown() {
	xxx_messageInfo_KernelDef.DiscardUnknown(m)
}

var xxx_messageInfo_KernelDef proto.InternalMessageInfo

func (m *KernelDef) GetOp() string {
	if m != nil {
		return m.Op
	}
	return ""
}

func (m *KernelDef) GetDeviceType() string {
	if m != nil {
		return m.DeviceType
	}
	return ""
}

func (m *KernelDef) GetConstraint() []*KernelDef_AttrConstraint {
	if m != nil {
		return m.Constraint
	}
	return nil
}

func (m *KernelDef) GetHostMemoryArg() []string {
	if m != nil {
		return m.HostMemoryArg
	}
	return nil
}

func (m *KernelDef) GetLabel() string {
	if m != nil {
		return m.Label
	}
	return ""
}

func (m *KernelDef) GetPriority() int32 {
	if m != nil {
		return m.Priority
	}
	return 0
}

type KernelDef_AttrConstraint struct {
	// Name of an attr from the Op.
	Name string `protobuf:"bytes,1,opt,name=name,proto3" json:"name,omitempty"`
	// A list of values that this kernel supports for this attr.
	// Like OpDef.AttrDef.allowed_values, except for kernels instead of Ops.
	AllowedValues *AttrValue `protobuf:"bytes,2,opt,name=allowed_values,json=allowedValues,proto3" json:"allowed_values,omitempty"`
}

func (m *KernelDef_AttrConstraint) Reset()         { *m = KernelDef_AttrConstraint{} }
func (m *KernelDef_AttrConstraint) String() string { return proto.CompactTextString(m) }
func (*KernelDef_AttrConstraint) ProtoMessage()    {}
func (*KernelDef_AttrConstraint) Descriptor() ([]byte, []int) {
	return fileDescriptor_18794e085ea7671a, []int{0, 0}
}
func (m *KernelDef_AttrConstraint) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *KernelDef_AttrConstraint) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_KernelDef_AttrConstraint.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *KernelDef_AttrConstraint) XXX_Merge(src proto.Message) {
	xxx_messageInfo_KernelDef_AttrConstraint.Merge(m, src)
}
func (m *KernelDef_AttrConstraint) XXX_Size() int {
	return m.Size()
}
func (m *KernelDef_AttrConstraint) XXX_DiscardUnknown() {
	xxx_messageInfo_KernelDef_AttrConstraint.DiscardUnknown(m)
}

var xxx_messageInfo_KernelDef_AttrConstraint proto.InternalMessageInfo

func (m *KernelDef_AttrConstraint) GetName() string {
	if m != nil {
		return m.Name
	}
	return ""
}

func (m *KernelDef_AttrConstraint) GetAllowedValues() *AttrValue {
	if m != nil {
		return m.AllowedValues
	}
	return nil
}

// A collection of KernelDefs
type KernelList struct {
	Kernel []*KernelDef `protobuf:"bytes,1,rep,name=kernel,proto3" json:"kernel,omitempty"`
}

func (m *KernelList) Reset()         { *m = KernelList{} }
func (m *KernelList) String() string { return proto.CompactTextString(m) }
func (*KernelList) ProtoMessage()    {}
func (*KernelList) Descriptor() ([]byte, []int) {
	return fileDescriptor_18794e085ea7671a, []int{1}
}
func (m *KernelList) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *KernelList) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_KernelList.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *KernelList) XXX_Merge(src proto.Message) {
	xxx_messageInfo_KernelList.Merge(m, src)
}
func (m *KernelList) XXX_Size() int {
	return m.Size()
}
func (m *KernelList) XXX_DiscardUnknown() {
	xxx_messageInfo_KernelList.DiscardUnknown(m)
}

var xxx_messageInfo_KernelList proto.InternalMessageInfo

func (m *KernelList) GetKernel() []*KernelDef {
	if m != nil {
		return m.Kernel
	}
	return nil
}

func init() {
	proto.RegisterType((*KernelDef)(nil), "tensorflow.KernelDef")
	proto.RegisterType((*KernelDef_AttrConstraint)(nil), "tensorflow.KernelDef.AttrConstraint")
	proto.RegisterType((*KernelList)(nil), "tensorflow.KernelList")
}

func init() {
	proto.RegisterFile("tensorflow/core/framework/kernel_def.proto", fileDescriptor_18794e085ea7671a)
}

var fileDescriptor_18794e085ea7671a = []byte{
	// 405 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x7c, 0x52, 0xc1, 0x6a, 0xdb, 0x40,
	0x10, 0xf5, 0xda, 0xb1, 0xa9, 0xc7, 0xc4, 0x81, 0xa5, 0x01, 0xe1, 0x83, 0x6a, 0x42, 0x29, 0xa6,
	0x10, 0x09, 0xd2, 0x63, 0x7b, 0xb1, 0x9b, 0x5b, 0x5b, 0x08, 0xa2, 0xf4, 0xd0, 0x8b, 0x90, 0xed,
	0x91, 0x22, 0xb2, 0xd2, 0x2c, 0xb3, 0x9b, 0x18, 0xff, 0x43, 0x0f, 0xfd, 0xa4, 0x1e, 0x7b, 0xcc,
	0xb1, 0xc7, 0x62, 0xff, 0x44, 0x8f, 0x45, 0x2b, 0x23, 0xab, 0xd0, 0xf6, 0xa6, 0x99, 0x7d, 0xef,
	0x8d, 0xde, 0xbc, 0x81, 0x97, 0x16, 0x4b, 0x43, 0x9c, 0x2a, 0xda, 0x84, 0x2b, 0x62, 0x0c, 0x53,
	0x4e, 0x0a, 0xdc, 0x10, 0xdf, 0x85, 0x77, 0xc8, 0x25, 0xaa, 0x78, 0x8d, 0x69, 0xa0, 0x99, 0x2c,
	0x49, 0x38, 0x62, 0x27, 0xff, 0xe1, 0x25, 0xd6, 0x72, 0xfc, 0x90, 0xa8, 0x7b, 0xac, 0x79, 0x17,
	0xdf, 0xba, 0x30, 0x7c, 0xe7, 0xc4, 0xae, 0x31, 0x95, 0x63, 0xe8, 0x92, 0xf6, 0xc4, 0x54, 0xcc,
	0x86, 0x51, 0x97, 0xb4, 0x7c, 0x06, 0xa3, 0x35, 0x3e, 0xe4, 0x2b, 0x8c, 0xed, 0x56, 0xa3, 0xd7,
	0x75, 0x0f, 0x50, 0xb7, 0x3e, 0x6e, 0x35, 0xca, 0x6b, 0x80, 0x15, 0x95, 0xc6, 0x72, 0x92, 0x97,
	0xd6, 0xeb, 0x4d, 0x7b, 0xb3, 0xd1, 0xd5, 0xf3, 0xe0, 0x38, 0x3f, 0x68, 0xb4, 0x83, 0xb9, 0xb5,
	0xfc, 0xb6, 0xc1, 0x46, 0x2d, 0x9e, 0x7c, 0x01, 0x67, 0xb7, 0x64, 0x6c, 0x5c, 0x60, 0x41, 0xbc,
	0x8d, 0x13, 0xce, 0xbc, 0x93, 0x69, 0x6f, 0x36, 0x8c, 0x4e, 0xab, 0xf6, 0x07, 0xd7, 0x9d, 0x73,
	0x26, 0x9f, 0x42, 0x5f, 0x25, 0x4b, 0x54, 0x5e, 0xdf, 0xfd, 0x48, 0x5d, 0xc8, 0x09, 0x3c, 0xd1,
	0x9c, 0x13, 0xe7, 0x76, 0xeb, 0x0d, 0xa6, 0x62, 0xd6, 0x8f, 0x9a, 0x7a, 0xb2, 0x84, 0xf1, 0x9f,
	0x73, 0xa5, 0x84, 0x93, 0x32, 0x29, 0xf0, 0x60, 0xd2, 0x7d, 0xcb, 0x37, 0x30, 0x4e, 0x94, 0xa2,
	0x0d, 0xae, 0xeb, 0xdd, 0x18, 0xe7, 0x74, 0x74, 0x75, 0xde, 0x76, 0x52, 0xe9, 0x7c, 0xaa, 0x5e,
	0xa3, 0xd3, 0x03, 0xd8, 0x55, 0xe6, 0xe2, 0x35, 0x40, 0xed, 0xf2, 0x7d, 0x6e, 0xac, 0xbc, 0x84,
	0x41, 0x1d, 0x8e, 0x27, 0xdc, 0x36, 0xce, 0xff, 0xba, 0x8d, 0xe8, 0x00, 0x5a, 0x7c, 0x11, 0xdf,
	0x77, 0xbe, 0x78, 0xdc, 0xf9, 0xe2, 0xe7, 0xce, 0x17, 0x5f, 0xf7, 0x7e, 0xe7, 0x71, 0xef, 0x77,
	0x7e, 0xec, 0xfd, 0x0e, 0x78, 0xc4, 0x59, 0x9b, 0xdc, 0xa4, 0xb8, 0x38, 0x6b, 0x74, 0x6e, 0xaa,
	0x10, 0xcd, 0x8d, 0xf8, 0xbc, 0xc8, 0x72, 0x7b, 0x7b, 0xbf, 0x0c, 0x56, 0x54, 0x84, 0x73, 0xad,
	0x55, 0x9e, 0xe6, 0xc8, 0x61, 0x46, 0x97, 0xad, 0x5b, 0xa8, 0x32, 0x34, 0xe1, 0x3f, 0x8f, 0xe3,
	0x97, 0x10, 0xcb, 0x81, 0xbb, 0x8a, 0x57, 0xbf, 0x03, 0x00, 0x00, 0xff, 0xff, 0x9c, 0xb2, 0x3a,
	0x4e, 0x7b, 0x02, 0x00, 0x00,
}

func (m *KernelDef) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *KernelDef) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *KernelDef) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if m.Priority != 0 {
		i = encodeVarintKernelDef(dAtA, i, uint64(m.Priority))
		i--
		dAtA[i] = 0x30
	}
	if len(m.Label) > 0 {
		i -= len(m.Label)
		copy(dAtA[i:], m.Label)
		i = encodeVarintKernelDef(dAtA, i, uint64(len(m.Label)))
		i--
		dAtA[i] = 0x2a
	}
	if len(m.HostMemoryArg) > 0 {
		for iNdEx := len(m.HostMemoryArg) - 1; iNdEx >= 0; iNdEx-- {
			i -= len(m.HostMemoryArg[iNdEx])
			copy(dAtA[i:], m.HostMemoryArg[iNdEx])
			i = encodeVarintKernelDef(dAtA, i, uint64(len(m.HostMemoryArg[iNdEx])))
			i--
			dAtA[i] = 0x22
		}
	}
	if len(m.Constraint) > 0 {
		for iNdEx := len(m.Constraint) - 1; iNdEx >= 0; iNdEx-- {
			{
				size, err := m.Constraint[iNdEx].MarshalToSizedBuffer(dAtA[:i])
				if err != nil {
					return 0, err
				}
				i -= size
				i = encodeVarintKernelDef(dAtA, i, uint64(size))
			}
			i--
			dAtA[i] = 0x1a
		}
	}
	if len(m.DeviceType) > 0 {
		i -= len(m.DeviceType)
		copy(dAtA[i:], m.DeviceType)
		i = encodeVarintKernelDef(dAtA, i, uint64(len(m.DeviceType)))
		i--
		dAtA[i] = 0x12
	}
	if len(m.Op) > 0 {
		i -= len(m.Op)
		copy(dAtA[i:], m.Op)
		i = encodeVarintKernelDef(dAtA, i, uint64(len(m.Op)))
		i--
		dAtA[i] = 0xa
	}
	return len(dAtA) - i, nil
}

func (m *KernelDef_AttrConstraint) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *KernelDef_AttrConstraint) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *KernelDef_AttrConstraint) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if m.AllowedValues != nil {
		{
			size, err := m.AllowedValues.MarshalToSizedBuffer(dAtA[:i])
			if err != nil {
				return 0, err
			}
			i -= size
			i = encodeVarintKernelDef(dAtA, i, uint64(size))
		}
		i--
		dAtA[i] = 0x12
	}
	if len(m.Name) > 0 {
		i -= len(m.Name)
		copy(dAtA[i:], m.Name)
		i = encodeVarintKernelDef(dAtA, i, uint64(len(m.Name)))
		i--
		dAtA[i] = 0xa
	}
	return len(dAtA) - i, nil
}

func (m *KernelList) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *KernelList) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *KernelList) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if len(m.Kernel) > 0 {
		for iNdEx := len(m.Kernel) - 1; iNdEx >= 0; iNdEx-- {
			{
				size, err := m.Kernel[iNdEx].MarshalToSizedBuffer(dAtA[:i])
				if err != nil {
					return 0, err
				}
				i -= size
				i = encodeVarintKernelDef(dAtA, i, uint64(size))
			}
			i--
			dAtA[i] = 0xa
		}
	}
	return len(dAtA) - i, nil
}

func encodeVarintKernelDef(dAtA []byte, offset int, v uint64) int {
	offset -= sovKernelDef(v)
	base := offset
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return base
}
func (m *KernelDef) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	l = len(m.Op)
	if l > 0 {
		n += 1 + l + sovKernelDef(uint64(l))
	}
	l = len(m.DeviceType)
	if l > 0 {
		n += 1 + l + sovKernelDef(uint64(l))
	}
	if len(m.Constraint) > 0 {
		for _, e := range m.Constraint {
			l = e.Size()
			n += 1 + l + sovKernelDef(uint64(l))
		}
	}
	if len(m.HostMemoryArg) > 0 {
		for _, s := range m.HostMemoryArg {
			l = len(s)
			n += 1 + l + sovKernelDef(uint64(l))
		}
	}
	l = len(m.Label)
	if l > 0 {
		n += 1 + l + sovKernelDef(uint64(l))
	}
	if m.Priority != 0 {
		n += 1 + sovKernelDef(uint64(m.Priority))
	}
	return n
}

func (m *KernelDef_AttrConstraint) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	l = len(m.Name)
	if l > 0 {
		n += 1 + l + sovKernelDef(uint64(l))
	}
	if m.AllowedValues != nil {
		l = m.AllowedValues.Size()
		n += 1 + l + sovKernelDef(uint64(l))
	}
	return n
}

func (m *KernelList) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if len(m.Kernel) > 0 {
		for _, e := range m.Kernel {
			l = e.Size()
			n += 1 + l + sovKernelDef(uint64(l))
		}
	}
	return n
}

func sovKernelDef(x uint64) (n int) {
	return (math_bits.Len64(x|1) + 6) / 7
}
func sozKernelDef(x uint64) (n int) {
	return sovKernelDef(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (m *KernelDef) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowKernelDef
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
			return fmt.Errorf("proto: KernelDef: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: KernelDef: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Op", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowKernelDef
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
				return ErrInvalidLengthKernelDef
			}
			postIndex := iNdEx + intStringLen
			if postIndex < 0 {
				return ErrInvalidLengthKernelDef
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.Op = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		case 2:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field DeviceType", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowKernelDef
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
				return ErrInvalidLengthKernelDef
			}
			postIndex := iNdEx + intStringLen
			if postIndex < 0 {
				return ErrInvalidLengthKernelDef
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.DeviceType = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		case 3:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Constraint", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowKernelDef
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
				return ErrInvalidLengthKernelDef
			}
			postIndex := iNdEx + msglen
			if postIndex < 0 {
				return ErrInvalidLengthKernelDef
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.Constraint = append(m.Constraint, &KernelDef_AttrConstraint{})
			if err := m.Constraint[len(m.Constraint)-1].Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		case 4:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field HostMemoryArg", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowKernelDef
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
				return ErrInvalidLengthKernelDef
			}
			postIndex := iNdEx + intStringLen
			if postIndex < 0 {
				return ErrInvalidLengthKernelDef
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.HostMemoryArg = append(m.HostMemoryArg, string(dAtA[iNdEx:postIndex]))
			iNdEx = postIndex
		case 5:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Label", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowKernelDef
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
				return ErrInvalidLengthKernelDef
			}
			postIndex := iNdEx + intStringLen
			if postIndex < 0 {
				return ErrInvalidLengthKernelDef
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.Label = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		case 6:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field Priority", wireType)
			}
			m.Priority = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowKernelDef
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.Priority |= int32(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		default:
			iNdEx = preIndex
			skippy, err := skipKernelDef(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if skippy < 0 {
				return ErrInvalidLengthKernelDef
			}
			if (iNdEx + skippy) < 0 {
				return ErrInvalidLengthKernelDef
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
func (m *KernelDef_AttrConstraint) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowKernelDef
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
			return fmt.Errorf("proto: AttrConstraint: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: AttrConstraint: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Name", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowKernelDef
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
				return ErrInvalidLengthKernelDef
			}
			postIndex := iNdEx + intStringLen
			if postIndex < 0 {
				return ErrInvalidLengthKernelDef
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.Name = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		case 2:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field AllowedValues", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowKernelDef
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
				return ErrInvalidLengthKernelDef
			}
			postIndex := iNdEx + msglen
			if postIndex < 0 {
				return ErrInvalidLengthKernelDef
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			if m.AllowedValues == nil {
				m.AllowedValues = &AttrValue{}
			}
			if err := m.AllowedValues.Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipKernelDef(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if skippy < 0 {
				return ErrInvalidLengthKernelDef
			}
			if (iNdEx + skippy) < 0 {
				return ErrInvalidLengthKernelDef
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
func (m *KernelList) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowKernelDef
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
			return fmt.Errorf("proto: KernelList: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: KernelList: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Kernel", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowKernelDef
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
				return ErrInvalidLengthKernelDef
			}
			postIndex := iNdEx + msglen
			if postIndex < 0 {
				return ErrInvalidLengthKernelDef
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.Kernel = append(m.Kernel, &KernelDef{})
			if err := m.Kernel[len(m.Kernel)-1].Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipKernelDef(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if skippy < 0 {
				return ErrInvalidLengthKernelDef
			}
			if (iNdEx + skippy) < 0 {
				return ErrInvalidLengthKernelDef
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
func skipKernelDef(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	depth := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowKernelDef
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
					return 0, ErrIntOverflowKernelDef
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
					return 0, ErrIntOverflowKernelDef
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
				return 0, ErrInvalidLengthKernelDef
			}
			iNdEx += length
		case 3:
			depth++
		case 4:
			if depth == 0 {
				return 0, ErrUnexpectedEndOfGroupKernelDef
			}
			depth--
		case 5:
			iNdEx += 4
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
		if iNdEx < 0 {
			return 0, ErrInvalidLengthKernelDef
		}
		if depth == 0 {
			return iNdEx, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}

var (
	ErrInvalidLengthKernelDef        = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowKernelDef          = fmt.Errorf("proto: integer overflow")
	ErrUnexpectedEndOfGroupKernelDef = fmt.Errorf("proto: unexpected end of group")
)
