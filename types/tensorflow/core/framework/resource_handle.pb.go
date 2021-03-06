// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: tensorflow/core/framework/resource_handle.proto

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

// Protocol buffer representing a handle to a tensorflow resource. Handles are
// not valid across executions, but can be serialized back and forth from within
// a single run.
type ResourceHandleProto struct {
	// Unique name for the device containing the resource.
	Device string `protobuf:"bytes,1,opt,name=device,proto3" json:"device,omitempty"`
	// Container in which this resource is placed.
	Container string `protobuf:"bytes,2,opt,name=container,proto3" json:"container,omitempty"`
	// Unique name of this resource.
	Name string `protobuf:"bytes,3,opt,name=name,proto3" json:"name,omitempty"`
	// Hash code for the type of the resource. Is only valid in the same device
	// and in the same execution.
	HashCode uint64 `protobuf:"varint,4,opt,name=hash_code,json=hashCode,proto3" json:"hash_code,omitempty"`
	// For debug-only, the name of the type pointed to by this handle, if
	// available.
	MaybeTypeName string `protobuf:"bytes,5,opt,name=maybe_type_name,json=maybeTypeName,proto3" json:"maybe_type_name,omitempty"`
	// Data types and shapes for the underlying resource.
	DtypesAndShapes []*ResourceHandleProto_DtypeAndShape `protobuf:"bytes,6,rep,name=dtypes_and_shapes,json=dtypesAndShapes,proto3" json:"dtypes_and_shapes,omitempty"`
	// A set of devices containing the resource. If empty, the resource only
	// exists on `device`.
	AllowedDevices []string `protobuf:"bytes,7,rep,name=allowed_devices,json=allowedDevices,proto3" json:"allowed_devices,omitempty"`
}

func (m *ResourceHandleProto) Reset()         { *m = ResourceHandleProto{} }
func (m *ResourceHandleProto) String() string { return proto.CompactTextString(m) }
func (*ResourceHandleProto) ProtoMessage()    {}
func (*ResourceHandleProto) Descriptor() ([]byte, []int) {
	return fileDescriptor_a36024d2bd9a2afd, []int{0}
}
func (m *ResourceHandleProto) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *ResourceHandleProto) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_ResourceHandleProto.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *ResourceHandleProto) XXX_Merge(src proto.Message) {
	xxx_messageInfo_ResourceHandleProto.Merge(m, src)
}
func (m *ResourceHandleProto) XXX_Size() int {
	return m.Size()
}
func (m *ResourceHandleProto) XXX_DiscardUnknown() {
	xxx_messageInfo_ResourceHandleProto.DiscardUnknown(m)
}

var xxx_messageInfo_ResourceHandleProto proto.InternalMessageInfo

func (m *ResourceHandleProto) GetDevice() string {
	if m != nil {
		return m.Device
	}
	return ""
}

func (m *ResourceHandleProto) GetContainer() string {
	if m != nil {
		return m.Container
	}
	return ""
}

func (m *ResourceHandleProto) GetName() string {
	if m != nil {
		return m.Name
	}
	return ""
}

func (m *ResourceHandleProto) GetHashCode() uint64 {
	if m != nil {
		return m.HashCode
	}
	return 0
}

func (m *ResourceHandleProto) GetMaybeTypeName() string {
	if m != nil {
		return m.MaybeTypeName
	}
	return ""
}

func (m *ResourceHandleProto) GetDtypesAndShapes() []*ResourceHandleProto_DtypeAndShape {
	if m != nil {
		return m.DtypesAndShapes
	}
	return nil
}

func (m *ResourceHandleProto) GetAllowedDevices() []string {
	if m != nil {
		return m.AllowedDevices
	}
	return nil
}

// Protocol buffer representing a pair of (data type, tensor shape).
type ResourceHandleProto_DtypeAndShape struct {
	Dtype DataType          `protobuf:"varint,1,opt,name=dtype,proto3,enum=tensorflow.DataType" json:"dtype,omitempty"`
	Shape *TensorShapeProto `protobuf:"bytes,2,opt,name=shape,proto3" json:"shape,omitempty"`
}

func (m *ResourceHandleProto_DtypeAndShape) Reset()         { *m = ResourceHandleProto_DtypeAndShape{} }
func (m *ResourceHandleProto_DtypeAndShape) String() string { return proto.CompactTextString(m) }
func (*ResourceHandleProto_DtypeAndShape) ProtoMessage()    {}
func (*ResourceHandleProto_DtypeAndShape) Descriptor() ([]byte, []int) {
	return fileDescriptor_a36024d2bd9a2afd, []int{0, 0}
}
func (m *ResourceHandleProto_DtypeAndShape) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *ResourceHandleProto_DtypeAndShape) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_ResourceHandleProto_DtypeAndShape.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *ResourceHandleProto_DtypeAndShape) XXX_Merge(src proto.Message) {
	xxx_messageInfo_ResourceHandleProto_DtypeAndShape.Merge(m, src)
}
func (m *ResourceHandleProto_DtypeAndShape) XXX_Size() int {
	return m.Size()
}
func (m *ResourceHandleProto_DtypeAndShape) XXX_DiscardUnknown() {
	xxx_messageInfo_ResourceHandleProto_DtypeAndShape.DiscardUnknown(m)
}

var xxx_messageInfo_ResourceHandleProto_DtypeAndShape proto.InternalMessageInfo

func (m *ResourceHandleProto_DtypeAndShape) GetDtype() DataType {
	if m != nil {
		return m.Dtype
	}
	return DataType_DT_INVALID
}

func (m *ResourceHandleProto_DtypeAndShape) GetShape() *TensorShapeProto {
	if m != nil {
		return m.Shape
	}
	return nil
}

func init() {
	proto.RegisterType((*ResourceHandleProto)(nil), "tensorflow.ResourceHandleProto")
	proto.RegisterType((*ResourceHandleProto_DtypeAndShape)(nil), "tensorflow.ResourceHandleProto.DtypeAndShape")
}

func init() {
	proto.RegisterFile("tensorflow/core/framework/resource_handle.proto", fileDescriptor_a36024d2bd9a2afd)
}

var fileDescriptor_a36024d2bd9a2afd = []byte{
	// 416 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x7c, 0x52, 0xb1, 0x8e, 0xd3, 0x40,
	0x10, 0xcd, 0xe2, 0x24, 0xe0, 0x3d, 0x5d, 0x22, 0x16, 0x84, 0xac, 0x70, 0xb2, 0x2c, 0x24, 0xc0,
	0x42, 0x9c, 0x2d, 0x99, 0x2f, 0xb8, 0x90, 0x82, 0x0a, 0x21, 0x73, 0x0d, 0x34, 0xd6, 0xc6, 0x3b,
	0x89, 0x2d, 0x6c, 0xaf, 0xb5, 0xeb, 0x23, 0xca, 0x37, 0xd0, 0xd0, 0xf2, 0x47, 0x94, 0x29, 0x29,
	0x51, 0xf2, 0x13, 0x94, 0xc8, 0xb3, 0x86, 0x18, 0x44, 0xe8, 0xbc, 0x6f, 0xde, 0x7b, 0xf3, 0x66,
	0x3c, 0x34, 0x6c, 0xa0, 0xd2, 0x52, 0xad, 0x0a, 0xb9, 0x09, 0x53, 0xa9, 0x20, 0x5c, 0x29, 0x5e,
	0xc2, 0x46, 0xaa, 0x0f, 0xa1, 0x02, 0x2d, 0x6f, 0x54, 0x0a, 0x49, 0xc6, 0x2b, 0x51, 0x40, 0x50,
	0x2b, 0xd9, 0x48, 0x46, 0x8f, 0x82, 0xd9, 0xf3, 0xd3, 0x62, 0x53, 0x49, 0x74, 0xc6, 0xeb, 0x4e,
	0x39, 0x7b, 0xfc, 0x1f, 0xf6, 0xb6, 0x06, 0x6d, 0x68, 0x8f, 0xbe, 0x58, 0xf4, 0x5e, 0xdc, 0xb5,
	0x7e, 0x85, 0x9d, 0xdf, 0x60, 0xe3, 0x07, 0x74, 0x2c, 0xe0, 0x63, 0x9e, 0x82, 0x43, 0x3c, 0xe2,
	0xdb, 0x71, 0xf7, 0x62, 0x17, 0xd4, 0x4e, 0x65, 0xd5, 0xf0, 0xbc, 0x02, 0xe5, 0xdc, 0xc2, 0xd2,
	0x11, 0x60, 0x8c, 0x0e, 0x2b, 0x5e, 0x82, 0x63, 0x61, 0x01, 0xbf, 0xd9, 0x43, 0x6a, 0x67, 0x5c,
	0x67, 0x49, 0x2a, 0x05, 0x38, 0x43, 0x8f, 0xf8, 0xc3, 0xf8, 0x4e, 0x0b, 0xbc, 0x94, 0x02, 0xd8,
	0x13, 0x3a, 0x2d, 0xf9, 0x76, 0x09, 0x49, 0x9b, 0x29, 0x41, 0xed, 0x08, 0xb5, 0xe7, 0x08, 0x5f,
	0x6f, 0x6b, 0x78, 0xdd, 0x9a, 0xbc, 0xa3, 0x77, 0x05, 0xc6, 0x4e, 0x78, 0x25, 0xcc, 0x9c, 0xda,
	0x19, 0x7b, 0x96, 0x7f, 0x16, 0x5d, 0x06, 0xc7, 0x49, 0x83, 0x7f, 0x8c, 0x12, 0x2c, 0x5a, 0xe1,
	0x55, 0x25, 0xde, 0xb6, 0xaa, 0x78, 0x6a, 0x7c, 0x7e, 0xbd, 0x35, 0x7b, 0x4a, 0xa7, 0xbc, 0x28,
	0xe4, 0x06, 0x44, 0x62, 0x66, 0xd4, 0xce, 0x6d, 0xcf, 0xf2, 0xed, 0x78, 0xd2, 0xc1, 0x0b, 0x83,
	0xce, 0x24, 0x3d, 0xff, 0xc3, 0x8a, 0x3d, 0xa3, 0x23, 0x34, 0xc3, 0x15, 0x4d, 0xa2, 0xfb, 0xfd,
	0x20, 0x0b, 0xde, 0xf0, 0x36, 0x7d, 0x6c, 0x28, 0x2c, 0xa2, 0x23, 0x4c, 0x8d, 0x3b, 0x3b, 0x8b,
	0x2e, 0xfa, 0xdc, 0x6b, 0xfc, 0x44, 0x4f, 0x4c, 0x1c, 0x1b, 0xea, 0xfc, 0x13, 0xf9, 0xba, 0x77,
	0xc9, 0x6e, 0xef, 0x92, 0xef, 0x7b, 0x97, 0x7c, 0x3e, 0xb8, 0x83, 0xdd, 0xc1, 0x1d, 0x7c, 0x3b,
	0xb8, 0x03, 0xea, 0x48, 0xb5, 0xee, 0x5b, 0xfc, 0xfe, 0xb9, 0xf3, 0xc9, 0x5f, 0x2b, 0x20, 0xef,
	0xe7, 0xeb, 0xbc, 0xc9, 0x6e, 0x96, 0x41, 0x2a, 0xcb, 0xf0, 0xaa, 0xae, 0x8b, 0x7c, 0x95, 0x83,
	0x0a, 0xd7, 0xf2, 0xb2, 0x77, 0x21, 0xb8, 0x93, 0xd3, 0xd7, 0xf9, 0x83, 0x90, 0xe5, 0x18, 0x0f,
	0xe6, 0xc5, 0xcf, 0x00, 0x00, 0x00, 0xff, 0xff, 0xbd, 0x35, 0xc8, 0x93, 0xc4, 0x02, 0x00, 0x00,
}

func (m *ResourceHandleProto) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *ResourceHandleProto) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *ResourceHandleProto) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if len(m.AllowedDevices) > 0 {
		for iNdEx := len(m.AllowedDevices) - 1; iNdEx >= 0; iNdEx-- {
			i -= len(m.AllowedDevices[iNdEx])
			copy(dAtA[i:], m.AllowedDevices[iNdEx])
			i = encodeVarintResourceHandle(dAtA, i, uint64(len(m.AllowedDevices[iNdEx])))
			i--
			dAtA[i] = 0x3a
		}
	}
	if len(m.DtypesAndShapes) > 0 {
		for iNdEx := len(m.DtypesAndShapes) - 1; iNdEx >= 0; iNdEx-- {
			{
				size, err := m.DtypesAndShapes[iNdEx].MarshalToSizedBuffer(dAtA[:i])
				if err != nil {
					return 0, err
				}
				i -= size
				i = encodeVarintResourceHandle(dAtA, i, uint64(size))
			}
			i--
			dAtA[i] = 0x32
		}
	}
	if len(m.MaybeTypeName) > 0 {
		i -= len(m.MaybeTypeName)
		copy(dAtA[i:], m.MaybeTypeName)
		i = encodeVarintResourceHandle(dAtA, i, uint64(len(m.MaybeTypeName)))
		i--
		dAtA[i] = 0x2a
	}
	if m.HashCode != 0 {
		i = encodeVarintResourceHandle(dAtA, i, uint64(m.HashCode))
		i--
		dAtA[i] = 0x20
	}
	if len(m.Name) > 0 {
		i -= len(m.Name)
		copy(dAtA[i:], m.Name)
		i = encodeVarintResourceHandle(dAtA, i, uint64(len(m.Name)))
		i--
		dAtA[i] = 0x1a
	}
	if len(m.Container) > 0 {
		i -= len(m.Container)
		copy(dAtA[i:], m.Container)
		i = encodeVarintResourceHandle(dAtA, i, uint64(len(m.Container)))
		i--
		dAtA[i] = 0x12
	}
	if len(m.Device) > 0 {
		i -= len(m.Device)
		copy(dAtA[i:], m.Device)
		i = encodeVarintResourceHandle(dAtA, i, uint64(len(m.Device)))
		i--
		dAtA[i] = 0xa
	}
	return len(dAtA) - i, nil
}

func (m *ResourceHandleProto_DtypeAndShape) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *ResourceHandleProto_DtypeAndShape) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *ResourceHandleProto_DtypeAndShape) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if m.Shape != nil {
		{
			size, err := m.Shape.MarshalToSizedBuffer(dAtA[:i])
			if err != nil {
				return 0, err
			}
			i -= size
			i = encodeVarintResourceHandle(dAtA, i, uint64(size))
		}
		i--
		dAtA[i] = 0x12
	}
	if m.Dtype != 0 {
		i = encodeVarintResourceHandle(dAtA, i, uint64(m.Dtype))
		i--
		dAtA[i] = 0x8
	}
	return len(dAtA) - i, nil
}

func encodeVarintResourceHandle(dAtA []byte, offset int, v uint64) int {
	offset -= sovResourceHandle(v)
	base := offset
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return base
}
func (m *ResourceHandleProto) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	l = len(m.Device)
	if l > 0 {
		n += 1 + l + sovResourceHandle(uint64(l))
	}
	l = len(m.Container)
	if l > 0 {
		n += 1 + l + sovResourceHandle(uint64(l))
	}
	l = len(m.Name)
	if l > 0 {
		n += 1 + l + sovResourceHandle(uint64(l))
	}
	if m.HashCode != 0 {
		n += 1 + sovResourceHandle(uint64(m.HashCode))
	}
	l = len(m.MaybeTypeName)
	if l > 0 {
		n += 1 + l + sovResourceHandle(uint64(l))
	}
	if len(m.DtypesAndShapes) > 0 {
		for _, e := range m.DtypesAndShapes {
			l = e.Size()
			n += 1 + l + sovResourceHandle(uint64(l))
		}
	}
	if len(m.AllowedDevices) > 0 {
		for _, s := range m.AllowedDevices {
			l = len(s)
			n += 1 + l + sovResourceHandle(uint64(l))
		}
	}
	return n
}

func (m *ResourceHandleProto_DtypeAndShape) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if m.Dtype != 0 {
		n += 1 + sovResourceHandle(uint64(m.Dtype))
	}
	if m.Shape != nil {
		l = m.Shape.Size()
		n += 1 + l + sovResourceHandle(uint64(l))
	}
	return n
}

func sovResourceHandle(x uint64) (n int) {
	return (math_bits.Len64(x|1) + 6) / 7
}
func sozResourceHandle(x uint64) (n int) {
	return sovResourceHandle(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (m *ResourceHandleProto) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowResourceHandle
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
			return fmt.Errorf("proto: ResourceHandleProto: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: ResourceHandleProto: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Device", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowResourceHandle
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
				return ErrInvalidLengthResourceHandle
			}
			postIndex := iNdEx + intStringLen
			if postIndex < 0 {
				return ErrInvalidLengthResourceHandle
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.Device = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		case 2:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Container", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowResourceHandle
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
				return ErrInvalidLengthResourceHandle
			}
			postIndex := iNdEx + intStringLen
			if postIndex < 0 {
				return ErrInvalidLengthResourceHandle
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.Container = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		case 3:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Name", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowResourceHandle
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
				return ErrInvalidLengthResourceHandle
			}
			postIndex := iNdEx + intStringLen
			if postIndex < 0 {
				return ErrInvalidLengthResourceHandle
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.Name = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		case 4:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field HashCode", wireType)
			}
			m.HashCode = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowResourceHandle
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.HashCode |= uint64(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		case 5:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field MaybeTypeName", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowResourceHandle
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
				return ErrInvalidLengthResourceHandle
			}
			postIndex := iNdEx + intStringLen
			if postIndex < 0 {
				return ErrInvalidLengthResourceHandle
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.MaybeTypeName = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		case 6:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field DtypesAndShapes", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowResourceHandle
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
				return ErrInvalidLengthResourceHandle
			}
			postIndex := iNdEx + msglen
			if postIndex < 0 {
				return ErrInvalidLengthResourceHandle
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.DtypesAndShapes = append(m.DtypesAndShapes, &ResourceHandleProto_DtypeAndShape{})
			if err := m.DtypesAndShapes[len(m.DtypesAndShapes)-1].Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		case 7:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field AllowedDevices", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowResourceHandle
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
				return ErrInvalidLengthResourceHandle
			}
			postIndex := iNdEx + intStringLen
			if postIndex < 0 {
				return ErrInvalidLengthResourceHandle
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.AllowedDevices = append(m.AllowedDevices, string(dAtA[iNdEx:postIndex]))
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipResourceHandle(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if skippy < 0 {
				return ErrInvalidLengthResourceHandle
			}
			if (iNdEx + skippy) < 0 {
				return ErrInvalidLengthResourceHandle
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
func (m *ResourceHandleProto_DtypeAndShape) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowResourceHandle
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
			return fmt.Errorf("proto: DtypeAndShape: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: DtypeAndShape: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field Dtype", wireType)
			}
			m.Dtype = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowResourceHandle
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.Dtype |= DataType(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		case 2:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Shape", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowResourceHandle
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
				return ErrInvalidLengthResourceHandle
			}
			postIndex := iNdEx + msglen
			if postIndex < 0 {
				return ErrInvalidLengthResourceHandle
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			if m.Shape == nil {
				m.Shape = &TensorShapeProto{}
			}
			if err := m.Shape.Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipResourceHandle(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if skippy < 0 {
				return ErrInvalidLengthResourceHandle
			}
			if (iNdEx + skippy) < 0 {
				return ErrInvalidLengthResourceHandle
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
func skipResourceHandle(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	depth := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowResourceHandle
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
					return 0, ErrIntOverflowResourceHandle
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
					return 0, ErrIntOverflowResourceHandle
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
				return 0, ErrInvalidLengthResourceHandle
			}
			iNdEx += length
		case 3:
			depth++
		case 4:
			if depth == 0 {
				return 0, ErrUnexpectedEndOfGroupResourceHandle
			}
			depth--
		case 5:
			iNdEx += 4
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
		if iNdEx < 0 {
			return 0, ErrInvalidLengthResourceHandle
		}
		if depth == 0 {
			return iNdEx, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}

var (
	ErrInvalidLengthResourceHandle        = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowResourceHandle          = fmt.Errorf("proto: integer overflow")
	ErrUnexpectedEndOfGroupResourceHandle = fmt.Errorf("proto: unexpected end of group")
)
