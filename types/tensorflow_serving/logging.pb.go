// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: tensorflow_serving/logging.proto

package tensorflow_serving

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

// Metadata logged along with the request logs.
type LogMetadata struct {
	ModelSpec      *ModelSpec      `protobuf:"bytes,1,opt,name=model_spec,json=modelSpec,proto3" json:"model_spec,omitempty"`
	SamplingConfig *SamplingConfig `protobuf:"bytes,2,opt,name=sampling_config,json=samplingConfig,proto3" json:"sampling_config,omitempty"`
	// List of tags used to load the relevant MetaGraphDef from SavedModel.
	SavedModelTags []string `protobuf:"bytes,3,rep,name=saved_model_tags,json=savedModelTags,proto3" json:"saved_model_tags,omitempty"`
}

func (m *LogMetadata) Reset()         { *m = LogMetadata{} }
func (m *LogMetadata) String() string { return proto.CompactTextString(m) }
func (*LogMetadata) ProtoMessage()    {}
func (*LogMetadata) Descriptor() ([]byte, []int) {
	return fileDescriptor_e2383633b45aa1b8, []int{0}
}
func (m *LogMetadata) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *LogMetadata) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_LogMetadata.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *LogMetadata) XXX_Merge(src proto.Message) {
	xxx_messageInfo_LogMetadata.Merge(m, src)
}
func (m *LogMetadata) XXX_Size() int {
	return m.Size()
}
func (m *LogMetadata) XXX_DiscardUnknown() {
	xxx_messageInfo_LogMetadata.DiscardUnknown(m)
}

var xxx_messageInfo_LogMetadata proto.InternalMessageInfo

func (m *LogMetadata) GetModelSpec() *ModelSpec {
	if m != nil {
		return m.ModelSpec
	}
	return nil
}

func (m *LogMetadata) GetSamplingConfig() *SamplingConfig {
	if m != nil {
		return m.SamplingConfig
	}
	return nil
}

func (m *LogMetadata) GetSavedModelTags() []string {
	if m != nil {
		return m.SavedModelTags
	}
	return nil
}

func init() {
	proto.RegisterType((*LogMetadata)(nil), "tensorflow.serving.LogMetadata")
}

func init() { proto.RegisterFile("tensorflow_serving/logging.proto", fileDescriptor_e2383633b45aa1b8) }

var fileDescriptor_e2383633b45aa1b8 = []byte{
	// 239 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0xe2, 0x52, 0x28, 0x49, 0xcd, 0x2b,
	0xce, 0x2f, 0x4a, 0xcb, 0xc9, 0x2f, 0x8f, 0x2f, 0x4e, 0x2d, 0x2a, 0xcb, 0xcc, 0x4b, 0xd7, 0xcf,
	0xc9, 0x4f, 0x4f, 0xcf, 0xcc, 0x4b, 0xd7, 0x2b, 0x28, 0xca, 0x2f, 0xc9, 0x17, 0x12, 0x42, 0xa8,
	0xd0, 0x83, 0xaa, 0x90, 0x92, 0xc3, 0xa2, 0x2b, 0x37, 0x3f, 0x25, 0x35, 0x07, 0xa2, 0x47, 0x4a,
	0x1d, 0xb7, 0xa9, 0xf1, 0xc9, 0xf9, 0x79, 0x69, 0x99, 0x50, 0xc3, 0x95, 0x0e, 0x31, 0x72, 0x71,
	0xfb, 0xe4, 0xa7, 0xfb, 0xa6, 0x96, 0x24, 0xa6, 0x24, 0x96, 0x24, 0x0a, 0xd9, 0x70, 0x71, 0x81,
	0xcd, 0x89, 0x2f, 0x2e, 0x48, 0x4d, 0x96, 0x60, 0x54, 0x60, 0xd4, 0xe0, 0x36, 0x92, 0xd5, 0xc3,
	0x74, 0x81, 0x9e, 0x2f, 0x48, 0x55, 0x70, 0x41, 0x6a, 0x72, 0x10, 0x67, 0x2e, 0x8c, 0x29, 0xe4,
	0xcd, 0xc5, 0x5f, 0x9c, 0x98, 0x5b, 0x90, 0x83, 0xb0, 0x46, 0x82, 0x09, 0x6c, 0x84, 0x12, 0x36,
	0x23, 0x82, 0xa1, 0x4a, 0x9d, 0xc1, 0x2a, 0x83, 0xf8, 0x8a, 0x51, 0xf8, 0x42, 0x1a, 0x5c, 0x02,
	0xc5, 0x89, 0x65, 0xa9, 0x29, 0xf1, 0x10, 0x07, 0x95, 0x24, 0xa6, 0x17, 0x4b, 0x30, 0x2b, 0x30,
	0x6b, 0x70, 0x82, 0x54, 0x96, 0xa5, 0xa6, 0x80, 0x5d, 0x10, 0x92, 0x98, 0x5e, 0xec, 0x24, 0x7d,
	0xe2, 0x91, 0x1c, 0xe3, 0x85, 0x47, 0x72, 0x8c, 0x0f, 0x1e, 0xc9, 0x31, 0x4e, 0x78, 0x2c, 0xc7,
	0x70, 0xe1, 0xb1, 0x1c, 0xc3, 0x8d, 0xc7, 0x72, 0x0c, 0x3f, 0x18, 0x19, 0x93, 0xd8, 0xc0, 0x1e,
	0x35, 0x06, 0x04, 0x00, 0x00, 0xff, 0xff, 0xd0, 0xf7, 0xfe, 0x94, 0x69, 0x01, 0x00, 0x00,
}

func (m *LogMetadata) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *LogMetadata) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *LogMetadata) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if len(m.SavedModelTags) > 0 {
		for iNdEx := len(m.SavedModelTags) - 1; iNdEx >= 0; iNdEx-- {
			i -= len(m.SavedModelTags[iNdEx])
			copy(dAtA[i:], m.SavedModelTags[iNdEx])
			i = encodeVarintLogging(dAtA, i, uint64(len(m.SavedModelTags[iNdEx])))
			i--
			dAtA[i] = 0x1a
		}
	}
	if m.SamplingConfig != nil {
		{
			size, err := m.SamplingConfig.MarshalToSizedBuffer(dAtA[:i])
			if err != nil {
				return 0, err
			}
			i -= size
			i = encodeVarintLogging(dAtA, i, uint64(size))
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
			i = encodeVarintLogging(dAtA, i, uint64(size))
		}
		i--
		dAtA[i] = 0xa
	}
	return len(dAtA) - i, nil
}

func encodeVarintLogging(dAtA []byte, offset int, v uint64) int {
	offset -= sovLogging(v)
	base := offset
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return base
}
func (m *LogMetadata) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if m.ModelSpec != nil {
		l = m.ModelSpec.Size()
		n += 1 + l + sovLogging(uint64(l))
	}
	if m.SamplingConfig != nil {
		l = m.SamplingConfig.Size()
		n += 1 + l + sovLogging(uint64(l))
	}
	if len(m.SavedModelTags) > 0 {
		for _, s := range m.SavedModelTags {
			l = len(s)
			n += 1 + l + sovLogging(uint64(l))
		}
	}
	return n
}

func sovLogging(x uint64) (n int) {
	return (math_bits.Len64(x|1) + 6) / 7
}
func sozLogging(x uint64) (n int) {
	return sovLogging(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (m *LogMetadata) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowLogging
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
			return fmt.Errorf("proto: LogMetadata: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: LogMetadata: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field ModelSpec", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowLogging
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
				return ErrInvalidLengthLogging
			}
			postIndex := iNdEx + msglen
			if postIndex < 0 {
				return ErrInvalidLengthLogging
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
				return fmt.Errorf("proto: wrong wireType = %d for field SamplingConfig", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowLogging
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
				return ErrInvalidLengthLogging
			}
			postIndex := iNdEx + msglen
			if postIndex < 0 {
				return ErrInvalidLengthLogging
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			if m.SamplingConfig == nil {
				m.SamplingConfig = &SamplingConfig{}
			}
			if err := m.SamplingConfig.Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		case 3:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field SavedModelTags", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowLogging
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
				return ErrInvalidLengthLogging
			}
			postIndex := iNdEx + intStringLen
			if postIndex < 0 {
				return ErrInvalidLengthLogging
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.SavedModelTags = append(m.SavedModelTags, string(dAtA[iNdEx:postIndex]))
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipLogging(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if skippy < 0 {
				return ErrInvalidLengthLogging
			}
			if (iNdEx + skippy) < 0 {
				return ErrInvalidLengthLogging
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
func skipLogging(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	depth := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowLogging
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
					return 0, ErrIntOverflowLogging
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
					return 0, ErrIntOverflowLogging
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
				return 0, ErrInvalidLengthLogging
			}
			iNdEx += length
		case 3:
			depth++
		case 4:
			if depth == 0 {
				return 0, ErrUnexpectedEndOfGroupLogging
			}
			depth--
		case 5:
			iNdEx += 4
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
		if iNdEx < 0 {
			return 0, ErrInvalidLengthLogging
		}
		if depth == 0 {
			return iNdEx, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}

var (
	ErrInvalidLengthLogging        = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowLogging          = fmt.Errorf("proto: integer overflow")
	ErrUnexpectedEndOfGroupLogging = fmt.Errorf("proto: unexpected end of group")
)
