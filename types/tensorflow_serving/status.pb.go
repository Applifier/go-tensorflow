// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: tensorflow_serving/status.proto

package tensorflow_serving

import proto "github.com/gogo/protobuf/proto"
import fmt "fmt"
import math "math"
import tensorflow_error "github.com/Applifier/go-tensorflow/types/tensorflow/core/lib/core"

import io "io"

// Reference imports to suppress errors if they are not otherwise used.
var _ = proto.Marshal
var _ = fmt.Errorf
var _ = math.Inf

// Status that corresponds to Status in
// third_party/tensorflow/core/lib/core/status.h.
type StatusProto struct {
	// Error code.
	ErrorCode tensorflow_error.Code `protobuf:"varint,1,opt,name=error_code,json=errorCode,proto3,enum=tensorflow.error.Code" json:"error_code,omitempty"`
	// Error message. Will only be set if an error was encountered.
	ErrorMessage string `protobuf:"bytes,2,opt,name=error_message,json=errorMessage,proto3" json:"error_message,omitempty"`
}

func (m *StatusProto) Reset()                    { *m = StatusProto{} }
func (m *StatusProto) String() string            { return proto.CompactTextString(m) }
func (*StatusProto) ProtoMessage()               {}
func (*StatusProto) Descriptor() ([]byte, []int) { return fileDescriptorStatus, []int{0} }

func (m *StatusProto) GetErrorCode() tensorflow_error.Code {
	if m != nil {
		return m.ErrorCode
	}
	return tensorflow_error.Code_OK
}

func (m *StatusProto) GetErrorMessage() string {
	if m != nil {
		return m.ErrorMessage
	}
	return ""
}

func init() {
	proto.RegisterType((*StatusProto)(nil), "tensorflow.serving.StatusProto")
}
func (m *StatusProto) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalTo(dAtA)
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *StatusProto) MarshalTo(dAtA []byte) (int, error) {
	var i int
	_ = i
	var l int
	_ = l
	if m.ErrorCode != 0 {
		dAtA[i] = 0x8
		i++
		i = encodeVarintStatus(dAtA, i, uint64(m.ErrorCode))
	}
	if len(m.ErrorMessage) > 0 {
		dAtA[i] = 0x12
		i++
		i = encodeVarintStatus(dAtA, i, uint64(len(m.ErrorMessage)))
		i += copy(dAtA[i:], m.ErrorMessage)
	}
	return i, nil
}

func encodeVarintStatus(dAtA []byte, offset int, v uint64) int {
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return offset + 1
}
func (m *StatusProto) Size() (n int) {
	var l int
	_ = l
	if m.ErrorCode != 0 {
		n += 1 + sovStatus(uint64(m.ErrorCode))
	}
	l = len(m.ErrorMessage)
	if l > 0 {
		n += 1 + l + sovStatus(uint64(l))
	}
	return n
}

func sovStatus(x uint64) (n int) {
	for {
		n++
		x >>= 7
		if x == 0 {
			break
		}
	}
	return n
}
func sozStatus(x uint64) (n int) {
	return sovStatus(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (m *StatusProto) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowStatus
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
			return fmt.Errorf("proto: StatusProto: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: StatusProto: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field ErrorCode", wireType)
			}
			m.ErrorCode = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowStatus
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.ErrorCode |= (tensorflow_error.Code(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		case 2:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field ErrorMessage", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowStatus
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
				return ErrInvalidLengthStatus
			}
			postIndex := iNdEx + intStringLen
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.ErrorMessage = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipStatus(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if skippy < 0 {
				return ErrInvalidLengthStatus
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
func skipStatus(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowStatus
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
					return 0, ErrIntOverflowStatus
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
					return 0, ErrIntOverflowStatus
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
				return 0, ErrInvalidLengthStatus
			}
			return iNdEx, nil
		case 3:
			for {
				var innerWire uint64
				var start int = iNdEx
				for shift := uint(0); ; shift += 7 {
					if shift >= 64 {
						return 0, ErrIntOverflowStatus
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
				next, err := skipStatus(dAtA[start:])
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
	ErrInvalidLengthStatus = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowStatus   = fmt.Errorf("proto: integer overflow")
)

func init() { proto.RegisterFile("tensorflow_serving/status.proto", fileDescriptorStatus) }

var fileDescriptorStatus = []byte{
	// 183 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0xe2, 0x92, 0x2f, 0x49, 0xcd, 0x2b,
	0xce, 0x2f, 0x4a, 0xcb, 0xc9, 0x2f, 0x8f, 0x2f, 0x4e, 0x2d, 0x2a, 0xcb, 0xcc, 0x4b, 0xd7, 0x2f,
	0x2e, 0x49, 0x2c, 0x29, 0x2d, 0xd6, 0x2b, 0x28, 0xca, 0x2f, 0xc9, 0x17, 0x12, 0x42, 0x28, 0xd0,
	0x83, 0x2a, 0x90, 0xd2, 0x42, 0x88, 0xe9, 0x27, 0xe7, 0x17, 0xa5, 0xea, 0xe7, 0x64, 0x26, 0x41,
	0x18, 0xa9, 0x45, 0x45, 0xf9, 0x45, 0xf1, 0xc9, 0xf9, 0x29, 0xa9, 0x50, 0xfd, 0x4a, 0x99, 0x5c,
	0xdc, 0xc1, 0x60, 0xf3, 0x02, 0xc0, 0xc6, 0x99, 0x72, 0x71, 0x21, 0xd4, 0x48, 0x30, 0x2a, 0x30,
	0x6a, 0xf0, 0x19, 0x89, 0xe9, 0x21, 0xd9, 0x01, 0x96, 0xd5, 0x73, 0xce, 0x4f, 0x49, 0x0d, 0xe2,
	0x04, 0xb3, 0x41, 0x4c, 0x21, 0x65, 0x2e, 0x5e, 0x88, 0xb6, 0xdc, 0xd4, 0xe2, 0xe2, 0xc4, 0xf4,
	0x54, 0x09, 0x26, 0x05, 0x46, 0x0d, 0xce, 0x20, 0x1e, 0xb0, 0xa0, 0x2f, 0x44, 0xcc, 0x49, 0xf8,
	0xc4, 0x23, 0x39, 0xc6, 0x0b, 0x8f, 0xe4, 0x18, 0x1f, 0x3c, 0x92, 0x63, 0x9c, 0xf0, 0x58, 0x8e,
	0xe1, 0x07, 0x23, 0x63, 0x12, 0x1b, 0xd8, 0x19, 0xc6, 0x80, 0x00, 0x00, 0x00, 0xff, 0xff, 0xe8,
	0x11, 0x4a, 0x1c, 0xe9, 0x00, 0x00, 0x00,
}
