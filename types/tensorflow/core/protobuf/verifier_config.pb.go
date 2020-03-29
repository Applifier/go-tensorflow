// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: tensorflow/core/protobuf/verifier_config.proto

package protobuf

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

type VerifierConfig_Toggle int32

const (
	VerifierConfig_DEFAULT VerifierConfig_Toggle = 0
	VerifierConfig_ON      VerifierConfig_Toggle = 1
	VerifierConfig_OFF     VerifierConfig_Toggle = 2
)

var VerifierConfig_Toggle_name = map[int32]string{
	0: "DEFAULT",
	1: "ON",
	2: "OFF",
}

var VerifierConfig_Toggle_value = map[string]int32{
	"DEFAULT": 0,
	"ON":      1,
	"OFF":     2,
}

func (x VerifierConfig_Toggle) String() string {
	return proto.EnumName(VerifierConfig_Toggle_name, int32(x))
}

func (VerifierConfig_Toggle) EnumDescriptor() ([]byte, []int) {
	return fileDescriptor_5049fcf5d8bb3c3c, []int{0, 0}
}

// The config for graph verifiers.
type VerifierConfig struct {
	// Deadline for completion of all verification i.e. all the Toggle ON
	// verifiers must complete execution within this time.
	VerificationTimeoutInMs int64 `protobuf:"varint,1,opt,name=verification_timeout_in_ms,json=verificationTimeoutInMs,proto3" json:"verification_timeout_in_ms,omitempty"`
	// Perform structural validation on a tensorflow graph. Default is OFF.
	StructureVerifier VerifierConfig_Toggle `protobuf:"varint,2,opt,name=structure_verifier,json=structureVerifier,proto3,enum=tensorflow.VerifierConfig_Toggle" json:"structure_verifier,omitempty"`
}

func (m *VerifierConfig) Reset()         { *m = VerifierConfig{} }
func (m *VerifierConfig) String() string { return proto.CompactTextString(m) }
func (*VerifierConfig) ProtoMessage()    {}
func (*VerifierConfig) Descriptor() ([]byte, []int) {
	return fileDescriptor_5049fcf5d8bb3c3c, []int{0}
}
func (m *VerifierConfig) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *VerifierConfig) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_VerifierConfig.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *VerifierConfig) XXX_Merge(src proto.Message) {
	xxx_messageInfo_VerifierConfig.Merge(m, src)
}
func (m *VerifierConfig) XXX_Size() int {
	return m.Size()
}
func (m *VerifierConfig) XXX_DiscardUnknown() {
	xxx_messageInfo_VerifierConfig.DiscardUnknown(m)
}

var xxx_messageInfo_VerifierConfig proto.InternalMessageInfo

func (m *VerifierConfig) GetVerificationTimeoutInMs() int64 {
	if m != nil {
		return m.VerificationTimeoutInMs
	}
	return 0
}

func (m *VerifierConfig) GetStructureVerifier() VerifierConfig_Toggle {
	if m != nil {
		return m.StructureVerifier
	}
	return VerifierConfig_DEFAULT
}

func init() {
	proto.RegisterEnum("tensorflow.VerifierConfig_Toggle", VerifierConfig_Toggle_name, VerifierConfig_Toggle_value)
	proto.RegisterType((*VerifierConfig)(nil), "tensorflow.VerifierConfig")
}

func init() {
	proto.RegisterFile("tensorflow/core/protobuf/verifier_config.proto", fileDescriptor_5049fcf5d8bb3c3c)
}

var fileDescriptor_5049fcf5d8bb3c3c = []byte{
	// 300 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x74, 0x90, 0x31, 0x4b, 0xc3, 0x40,
	0x18, 0x86, 0xf3, 0xb5, 0xd0, 0xc2, 0x09, 0xa5, 0x1e, 0x82, 0xc5, 0xe1, 0xa8, 0x1d, 0xa4, 0x8b,
	0x39, 0xd0, 0xd1, 0xa9, 0x55, 0x0b, 0x82, 0xda, 0x52, 0xaa, 0x83, 0x4b, 0x68, 0xc3, 0xe5, 0x3c,
	0x4c, 0xf2, 0x85, 0xbb, 0x8b, 0xc5, 0x7f, 0xd1, 0x7f, 0xa5, 0x63, 0x47, 0x47, 0x49, 0xfe, 0x84,
	0xa3, 0x98, 0x58, 0x4d, 0x07, 0xc7, 0x8f, 0xf7, 0xf9, 0x78, 0x5f, 0x1e, 0xe2, 0x5a, 0x11, 0x1b,
	0xd4, 0x41, 0x88, 0x4b, 0xee, 0xa3, 0x16, 0x3c, 0xd1, 0x68, 0x71, 0x91, 0x06, 0xfc, 0x59, 0x68,
	0x15, 0x28, 0xa1, 0x3d, 0x1f, 0xe3, 0x40, 0x49, 0xb7, 0x08, 0x28, 0xf9, 0xe3, 0x7b, 0xaf, 0x40,
	0x5a, 0xf7, 0x3f, 0xd4, 0x79, 0x01, 0xd1, 0x33, 0x72, 0x50, 0xfe, 0xf9, 0x73, 0xab, 0x30, 0xf6,
	0xac, 0x8a, 0x04, 0xa6, 0xd6, 0x53, 0xb1, 0x17, 0x99, 0x0e, 0x74, 0xa1, 0x5f, 0x9f, 0xee, 0x57,
	0x89, 0x59, 0x09, 0x5c, 0xc5, 0x37, 0x86, 0x4e, 0x08, 0x35, 0x56, 0xa7, 0xbe, 0x4d, 0xb5, 0xf0,
	0x36, 0xf5, 0x9d, 0x5a, 0x17, 0xfa, 0xad, 0x93, 0xc3, 0xca, 0x50, 0x77, 0xbb, 0xd4, 0x9d, 0xa1,
	0x94, 0xa1, 0x98, 0xee, 0xfe, 0x3e, 0x6f, 0xf2, 0xde, 0x11, 0x69, 0x94, 0x21, 0xdd, 0x21, 0xcd,
	0x8b, 0xcb, 0xd1, 0xe0, 0xee, 0x7a, 0xd6, 0x76, 0x68, 0x83, 0xd4, 0xc6, 0xb7, 0x6d, 0xa0, 0x4d,
	0x52, 0x1f, 0x8f, 0x46, 0xed, 0xda, 0x70, 0x05, 0x6f, 0x19, 0x83, 0x75, 0xc6, 0xe0, 0x23, 0x63,
	0xb0, 0xca, 0x99, 0xb3, 0xce, 0x99, 0xf3, 0x9e, 0x33, 0x87, 0x74, 0x50, 0xcb, 0x6a, 0x77, 0xa0,
	0xe7, 0x91, 0x58, 0xa2, 0x7e, 0x1a, 0xee, 0x6d, 0xcf, 0x98, 0x7c, 0xfb, 0x31, 0x13, 0x78, 0x18,
	0x48, 0x65, 0x1f, 0xd3, 0x85, 0xeb, 0x63, 0xc4, 0x07, 0x49, 0x12, 0x16, 0x0c, 0x97, 0x78, 0x5c,
	0x51, 0x6d, 0x5f, 0x12, 0x61, 0xf8, 0x7f, 0xee, 0x3f, 0x01, 0x16, 0x8d, 0xe2, 0x38, 0xfd, 0x0a,
	0x00, 0x00, 0xff, 0xff, 0xfc, 0xce, 0x9d, 0x9c, 0xa1, 0x01, 0x00, 0x00,
}

func (m *VerifierConfig) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *VerifierConfig) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *VerifierConfig) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if m.StructureVerifier != 0 {
		i = encodeVarintVerifierConfig(dAtA, i, uint64(m.StructureVerifier))
		i--
		dAtA[i] = 0x10
	}
	if m.VerificationTimeoutInMs != 0 {
		i = encodeVarintVerifierConfig(dAtA, i, uint64(m.VerificationTimeoutInMs))
		i--
		dAtA[i] = 0x8
	}
	return len(dAtA) - i, nil
}

func encodeVarintVerifierConfig(dAtA []byte, offset int, v uint64) int {
	offset -= sovVerifierConfig(v)
	base := offset
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return base
}
func (m *VerifierConfig) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if m.VerificationTimeoutInMs != 0 {
		n += 1 + sovVerifierConfig(uint64(m.VerificationTimeoutInMs))
	}
	if m.StructureVerifier != 0 {
		n += 1 + sovVerifierConfig(uint64(m.StructureVerifier))
	}
	return n
}

func sovVerifierConfig(x uint64) (n int) {
	return (math_bits.Len64(x|1) + 6) / 7
}
func sozVerifierConfig(x uint64) (n int) {
	return sovVerifierConfig(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (m *VerifierConfig) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowVerifierConfig
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
			return fmt.Errorf("proto: VerifierConfig: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: VerifierConfig: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field VerificationTimeoutInMs", wireType)
			}
			m.VerificationTimeoutInMs = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowVerifierConfig
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.VerificationTimeoutInMs |= int64(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		case 2:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field StructureVerifier", wireType)
			}
			m.StructureVerifier = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowVerifierConfig
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.StructureVerifier |= VerifierConfig_Toggle(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		default:
			iNdEx = preIndex
			skippy, err := skipVerifierConfig(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if skippy < 0 {
				return ErrInvalidLengthVerifierConfig
			}
			if (iNdEx + skippy) < 0 {
				return ErrInvalidLengthVerifierConfig
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
func skipVerifierConfig(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	depth := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowVerifierConfig
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
					return 0, ErrIntOverflowVerifierConfig
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
					return 0, ErrIntOverflowVerifierConfig
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
				return 0, ErrInvalidLengthVerifierConfig
			}
			iNdEx += length
		case 3:
			depth++
		case 4:
			if depth == 0 {
				return 0, ErrUnexpectedEndOfGroupVerifierConfig
			}
			depth--
		case 5:
			iNdEx += 4
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
		if iNdEx < 0 {
			return 0, ErrInvalidLengthVerifierConfig
		}
		if depth == 0 {
			return iNdEx, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}

var (
	ErrInvalidLengthVerifierConfig        = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowVerifierConfig          = fmt.Errorf("proto: integer overflow")
	ErrUnexpectedEndOfGroupVerifierConfig = fmt.Errorf("proto: unexpected end of group")
)
