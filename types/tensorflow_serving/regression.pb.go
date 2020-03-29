// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: tensorflow_serving/regression.proto

package tensorflow_serving

import (
	encoding_binary "encoding/binary"
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

// Regression result for a single item (tensorflow.Example).
type Regression struct {
	Value float32 `protobuf:"fixed32,1,opt,name=value,proto3" json:"value,omitempty"`
}

func (m *Regression) Reset()         { *m = Regression{} }
func (m *Regression) String() string { return proto.CompactTextString(m) }
func (*Regression) ProtoMessage()    {}
func (*Regression) Descriptor() ([]byte, []int) {
	return fileDescriptor_878511867b21c925, []int{0}
}
func (m *Regression) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *Regression) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_Regression.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *Regression) XXX_Merge(src proto.Message) {
	xxx_messageInfo_Regression.Merge(m, src)
}
func (m *Regression) XXX_Size() int {
	return m.Size()
}
func (m *Regression) XXX_DiscardUnknown() {
	xxx_messageInfo_Regression.DiscardUnknown(m)
}

var xxx_messageInfo_Regression proto.InternalMessageInfo

func (m *Regression) GetValue() float32 {
	if m != nil {
		return m.Value
	}
	return 0
}

// Contains one result per input example, in the same order as the input in
// RegressionRequest.
type RegressionResult struct {
	Regressions []*Regression `protobuf:"bytes,1,rep,name=regressions,proto3" json:"regressions,omitempty"`
}

func (m *RegressionResult) Reset()         { *m = RegressionResult{} }
func (m *RegressionResult) String() string { return proto.CompactTextString(m) }
func (*RegressionResult) ProtoMessage()    {}
func (*RegressionResult) Descriptor() ([]byte, []int) {
	return fileDescriptor_878511867b21c925, []int{1}
}
func (m *RegressionResult) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *RegressionResult) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_RegressionResult.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *RegressionResult) XXX_Merge(src proto.Message) {
	xxx_messageInfo_RegressionResult.Merge(m, src)
}
func (m *RegressionResult) XXX_Size() int {
	return m.Size()
}
func (m *RegressionResult) XXX_DiscardUnknown() {
	xxx_messageInfo_RegressionResult.DiscardUnknown(m)
}

var xxx_messageInfo_RegressionResult proto.InternalMessageInfo

func (m *RegressionResult) GetRegressions() []*Regression {
	if m != nil {
		return m.Regressions
	}
	return nil
}

type RegressionRequest struct {
	// Model Specification. If version is not specified, will use the latest
	// (numerical) version.
	ModelSpec *ModelSpec `protobuf:"bytes,1,opt,name=model_spec,json=modelSpec,proto3" json:"model_spec,omitempty"`
	// Input data.
	Input *Input `protobuf:"bytes,2,opt,name=input,proto3" json:"input,omitempty"`
}

func (m *RegressionRequest) Reset()         { *m = RegressionRequest{} }
func (m *RegressionRequest) String() string { return proto.CompactTextString(m) }
func (*RegressionRequest) ProtoMessage()    {}
func (*RegressionRequest) Descriptor() ([]byte, []int) {
	return fileDescriptor_878511867b21c925, []int{2}
}
func (m *RegressionRequest) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *RegressionRequest) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_RegressionRequest.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *RegressionRequest) XXX_Merge(src proto.Message) {
	xxx_messageInfo_RegressionRequest.Merge(m, src)
}
func (m *RegressionRequest) XXX_Size() int {
	return m.Size()
}
func (m *RegressionRequest) XXX_DiscardUnknown() {
	xxx_messageInfo_RegressionRequest.DiscardUnknown(m)
}

var xxx_messageInfo_RegressionRequest proto.InternalMessageInfo

func (m *RegressionRequest) GetModelSpec() *ModelSpec {
	if m != nil {
		return m.ModelSpec
	}
	return nil
}

func (m *RegressionRequest) GetInput() *Input {
	if m != nil {
		return m.Input
	}
	return nil
}

type RegressionResponse struct {
	// Effective Model Specification used for regression.
	ModelSpec *ModelSpec        `protobuf:"bytes,2,opt,name=model_spec,json=modelSpec,proto3" json:"model_spec,omitempty"`
	Result    *RegressionResult `protobuf:"bytes,1,opt,name=result,proto3" json:"result,omitempty"`
}

func (m *RegressionResponse) Reset()         { *m = RegressionResponse{} }
func (m *RegressionResponse) String() string { return proto.CompactTextString(m) }
func (*RegressionResponse) ProtoMessage()    {}
func (*RegressionResponse) Descriptor() ([]byte, []int) {
	return fileDescriptor_878511867b21c925, []int{3}
}
func (m *RegressionResponse) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *RegressionResponse) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_RegressionResponse.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *RegressionResponse) XXX_Merge(src proto.Message) {
	xxx_messageInfo_RegressionResponse.Merge(m, src)
}
func (m *RegressionResponse) XXX_Size() int {
	return m.Size()
}
func (m *RegressionResponse) XXX_DiscardUnknown() {
	xxx_messageInfo_RegressionResponse.DiscardUnknown(m)
}

var xxx_messageInfo_RegressionResponse proto.InternalMessageInfo

func (m *RegressionResponse) GetModelSpec() *ModelSpec {
	if m != nil {
		return m.ModelSpec
	}
	return nil
}

func (m *RegressionResponse) GetResult() *RegressionResult {
	if m != nil {
		return m.Result
	}
	return nil
}

func init() {
	proto.RegisterType((*Regression)(nil), "tensorflow.serving.Regression")
	proto.RegisterType((*RegressionResult)(nil), "tensorflow.serving.RegressionResult")
	proto.RegisterType((*RegressionRequest)(nil), "tensorflow.serving.RegressionRequest")
	proto.RegisterType((*RegressionResponse)(nil), "tensorflow.serving.RegressionResponse")
}

func init() {
	proto.RegisterFile("tensorflow_serving/regression.proto", fileDescriptor_878511867b21c925)
}

var fileDescriptor_878511867b21c925 = []byte{
	// 283 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0xe2, 0x52, 0x2e, 0x49, 0xcd, 0x2b,
	0xce, 0x2f, 0x4a, 0xcb, 0xc9, 0x2f, 0x8f, 0x2f, 0x4e, 0x2d, 0x2a, 0xcb, 0xcc, 0x4b, 0xd7, 0x2f,
	0x4a, 0x4d, 0x2f, 0x4a, 0x2d, 0x2e, 0xce, 0xcc, 0xcf, 0xd3, 0x2b, 0x28, 0xca, 0x2f, 0xc9, 0x17,
	0x12, 0x42, 0x28, 0xd2, 0x83, 0x2a, 0x92, 0x92, 0xc3, 0xa2, 0x31, 0x33, 0xaf, 0xa0, 0xb4, 0x04,
	0xa2, 0x07, 0xab, 0x7c, 0x6e, 0x7e, 0x4a, 0x6a, 0x0e, 0x44, 0x5e, 0x49, 0x89, 0x8b, 0x2b, 0x08,
	0x6e, 0x8f, 0x90, 0x08, 0x17, 0x6b, 0x59, 0x62, 0x4e, 0x69, 0xaa, 0x04, 0xa3, 0x02, 0xa3, 0x06,
	0x53, 0x10, 0x84, 0xa3, 0x14, 0xc2, 0x25, 0x80, 0x50, 0x13, 0x94, 0x5a, 0x5c, 0x9a, 0x53, 0x22,
	0xe4, 0xc0, 0xc5, 0x8d, 0x70, 0x5f, 0xb1, 0x04, 0xa3, 0x02, 0xb3, 0x06, 0xb7, 0x91, 0x9c, 0x1e,
	0xa6, 0x0b, 0xf5, 0x90, 0xb4, 0x22, 0x6b, 0x51, 0x6a, 0x62, 0xe4, 0x12, 0x44, 0x36, 0xb6, 0xb0,
	0x34, 0xb5, 0xb8, 0x44, 0xc8, 0x86, 0x8b, 0x0b, 0xec, 0xbc, 0xf8, 0xe2, 0x82, 0xd4, 0x64, 0xb0,
	0x33, 0xb8, 0x8d, 0x64, 0xb1, 0x19, 0xeb, 0x0b, 0x52, 0x15, 0x5c, 0x90, 0x9a, 0x1c, 0xc4, 0x99,
	0x0b, 0x63, 0x0a, 0xe9, 0x73, 0xb1, 0x82, 0x3d, 0x2f, 0xc1, 0x04, 0xd6, 0x28, 0x89, 0x4d, 0xa3,
	0x27, 0x48, 0x41, 0x10, 0x44, 0x9d, 0xd2, 0x04, 0x46, 0x2e, 0x21, 0x14, 0xbf, 0x15, 0xe4, 0xe7,
	0x15, 0xa7, 0xa2, 0xb9, 0x82, 0x89, 0x44, 0x57, 0xd8, 0x70, 0xb1, 0x15, 0x81, 0x43, 0x09, 0xea,
	0x7e, 0x15, 0x02, 0xc1, 0x02, 0x56, 0x1b, 0x04, 0xd5, 0xe3, 0x24, 0x7d, 0xe2, 0x91, 0x1c, 0xe3,
	0x85, 0x47, 0x72, 0x8c, 0x0f, 0x1e, 0xc9, 0x31, 0x4e, 0x78, 0x2c, 0xc7, 0x70, 0xe1, 0xb1, 0x1c,
	0xc3, 0x8d, 0xc7, 0x72, 0x0c, 0x3f, 0x18, 0x19, 0x93, 0xd8, 0xc0, 0xb1, 0x66, 0x0c, 0x08, 0x00,
	0x00, 0xff, 0xff, 0x16, 0x34, 0x1f, 0x12, 0x30, 0x02, 0x00, 0x00,
}

func (m *Regression) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *Regression) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *Regression) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if m.Value != 0 {
		i -= 4
		encoding_binary.LittleEndian.PutUint32(dAtA[i:], uint32(math.Float32bits(float32(m.Value))))
		i--
		dAtA[i] = 0xd
	}
	return len(dAtA) - i, nil
}

func (m *RegressionResult) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *RegressionResult) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *RegressionResult) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if len(m.Regressions) > 0 {
		for iNdEx := len(m.Regressions) - 1; iNdEx >= 0; iNdEx-- {
			{
				size, err := m.Regressions[iNdEx].MarshalToSizedBuffer(dAtA[:i])
				if err != nil {
					return 0, err
				}
				i -= size
				i = encodeVarintRegression(dAtA, i, uint64(size))
			}
			i--
			dAtA[i] = 0xa
		}
	}
	return len(dAtA) - i, nil
}

func (m *RegressionRequest) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *RegressionRequest) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *RegressionRequest) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if m.Input != nil {
		{
			size, err := m.Input.MarshalToSizedBuffer(dAtA[:i])
			if err != nil {
				return 0, err
			}
			i -= size
			i = encodeVarintRegression(dAtA, i, uint64(size))
		}
		i--
		dAtA[i] = 0x12
	}
	if m.ModelSpec != nil {
		{
			size, err := m.ModelSpec.MarshalToSizedBuffer(dAtA[:i])
			if err != nil {
				return 0, err
			}
			i -= size
			i = encodeVarintRegression(dAtA, i, uint64(size))
		}
		i--
		dAtA[i] = 0xa
	}
	return len(dAtA) - i, nil
}

func (m *RegressionResponse) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *RegressionResponse) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *RegressionResponse) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if m.ModelSpec != nil {
		{
			size, err := m.ModelSpec.MarshalToSizedBuffer(dAtA[:i])
			if err != nil {
				return 0, err
			}
			i -= size
			i = encodeVarintRegression(dAtA, i, uint64(size))
		}
		i--
		dAtA[i] = 0x12
	}
	if m.Result != nil {
		{
			size, err := m.Result.MarshalToSizedBuffer(dAtA[:i])
			if err != nil {
				return 0, err
			}
			i -= size
			i = encodeVarintRegression(dAtA, i, uint64(size))
		}
		i--
		dAtA[i] = 0xa
	}
	return len(dAtA) - i, nil
}

func encodeVarintRegression(dAtA []byte, offset int, v uint64) int {
	offset -= sovRegression(v)
	base := offset
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return base
}
func (m *Regression) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if m.Value != 0 {
		n += 5
	}
	return n
}

func (m *RegressionResult) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if len(m.Regressions) > 0 {
		for _, e := range m.Regressions {
			l = e.Size()
			n += 1 + l + sovRegression(uint64(l))
		}
	}
	return n
}

func (m *RegressionRequest) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if m.ModelSpec != nil {
		l = m.ModelSpec.Size()
		n += 1 + l + sovRegression(uint64(l))
	}
	if m.Input != nil {
		l = m.Input.Size()
		n += 1 + l + sovRegression(uint64(l))
	}
	return n
}

func (m *RegressionResponse) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if m.Result != nil {
		l = m.Result.Size()
		n += 1 + l + sovRegression(uint64(l))
	}
	if m.ModelSpec != nil {
		l = m.ModelSpec.Size()
		n += 1 + l + sovRegression(uint64(l))
	}
	return n
}

func sovRegression(x uint64) (n int) {
	return (math_bits.Len64(x|1) + 6) / 7
}
func sozRegression(x uint64) (n int) {
	return sovRegression(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (m *Regression) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowRegression
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
			return fmt.Errorf("proto: Regression: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: Regression: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 5 {
				return fmt.Errorf("proto: wrong wireType = %d for field Value", wireType)
			}
			var v uint32
			if (iNdEx + 4) > l {
				return io.ErrUnexpectedEOF
			}
			v = uint32(encoding_binary.LittleEndian.Uint32(dAtA[iNdEx:]))
			iNdEx += 4
			m.Value = float32(math.Float32frombits(v))
		default:
			iNdEx = preIndex
			skippy, err := skipRegression(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if skippy < 0 {
				return ErrInvalidLengthRegression
			}
			if (iNdEx + skippy) < 0 {
				return ErrInvalidLengthRegression
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
func (m *RegressionResult) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowRegression
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
			return fmt.Errorf("proto: RegressionResult: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: RegressionResult: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Regressions", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowRegression
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
				return ErrInvalidLengthRegression
			}
			postIndex := iNdEx + msglen
			if postIndex < 0 {
				return ErrInvalidLengthRegression
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.Regressions = append(m.Regressions, &Regression{})
			if err := m.Regressions[len(m.Regressions)-1].Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipRegression(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if skippy < 0 {
				return ErrInvalidLengthRegression
			}
			if (iNdEx + skippy) < 0 {
				return ErrInvalidLengthRegression
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
func (m *RegressionRequest) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowRegression
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
			return fmt.Errorf("proto: RegressionRequest: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: RegressionRequest: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field ModelSpec", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowRegression
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
				return ErrInvalidLengthRegression
			}
			postIndex := iNdEx + msglen
			if postIndex < 0 {
				return ErrInvalidLengthRegression
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			if m.ModelSpec == nil {
				m.ModelSpec = &ModelSpec{}
			}
			if err := m.ModelSpec.Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		case 2:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Input", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowRegression
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
				return ErrInvalidLengthRegression
			}
			postIndex := iNdEx + msglen
			if postIndex < 0 {
				return ErrInvalidLengthRegression
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			if m.Input == nil {
				m.Input = &Input{}
			}
			if err := m.Input.Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipRegression(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if skippy < 0 {
				return ErrInvalidLengthRegression
			}
			if (iNdEx + skippy) < 0 {
				return ErrInvalidLengthRegression
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
func (m *RegressionResponse) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowRegression
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
			return fmt.Errorf("proto: RegressionResponse: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: RegressionResponse: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Result", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowRegression
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
				return ErrInvalidLengthRegression
			}
			postIndex := iNdEx + msglen
			if postIndex < 0 {
				return ErrInvalidLengthRegression
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			if m.Result == nil {
				m.Result = &RegressionResult{}
			}
			if err := m.Result.Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		case 2:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field ModelSpec", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowRegression
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
				return ErrInvalidLengthRegression
			}
			postIndex := iNdEx + msglen
			if postIndex < 0 {
				return ErrInvalidLengthRegression
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			if m.ModelSpec == nil {
				m.ModelSpec = &ModelSpec{}
			}
			if err := m.ModelSpec.Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipRegression(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if skippy < 0 {
				return ErrInvalidLengthRegression
			}
			if (iNdEx + skippy) < 0 {
				return ErrInvalidLengthRegression
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
func skipRegression(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	depth := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowRegression
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
					return 0, ErrIntOverflowRegression
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
					return 0, ErrIntOverflowRegression
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
				return 0, ErrInvalidLengthRegression
			}
			iNdEx += length
		case 3:
			depth++
		case 4:
			if depth == 0 {
				return 0, ErrUnexpectedEndOfGroupRegression
			}
			depth--
		case 5:
			iNdEx += 4
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
		if iNdEx < 0 {
			return 0, ErrInvalidLengthRegression
		}
		if depth == 0 {
			return iNdEx, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}

var (
	ErrInvalidLengthRegression        = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowRegression          = fmt.Errorf("proto: integer overflow")
	ErrUnexpectedEndOfGroupRegression = fmt.Errorf("proto: unexpected end of group")
)
