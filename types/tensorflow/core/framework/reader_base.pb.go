// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: tensorflow/core/framework/reader_base.proto

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

// For serializing and restoring the state of ReaderBase, see
// reader_base.h for details.
type ReaderBaseState struct {
	WorkStarted        int64  `protobuf:"varint,1,opt,name=work_started,json=workStarted,proto3" json:"work_started,omitempty"`
	WorkFinished       int64  `protobuf:"varint,2,opt,name=work_finished,json=workFinished,proto3" json:"work_finished,omitempty"`
	NumRecordsProduced int64  `protobuf:"varint,3,opt,name=num_records_produced,json=numRecordsProduced,proto3" json:"num_records_produced,omitempty"`
	CurrentWork        []byte `protobuf:"bytes,4,opt,name=current_work,json=currentWork,proto3" json:"current_work,omitempty"`
}

func (m *ReaderBaseState) Reset()         { *m = ReaderBaseState{} }
func (m *ReaderBaseState) String() string { return proto.CompactTextString(m) }
func (*ReaderBaseState) ProtoMessage()    {}
func (*ReaderBaseState) Descriptor() ([]byte, []int) {
	return fileDescriptor_9d8282e7620a01b6, []int{0}
}
func (m *ReaderBaseState) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *ReaderBaseState) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_ReaderBaseState.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *ReaderBaseState) XXX_Merge(src proto.Message) {
	xxx_messageInfo_ReaderBaseState.Merge(m, src)
}
func (m *ReaderBaseState) XXX_Size() int {
	return m.Size()
}
func (m *ReaderBaseState) XXX_DiscardUnknown() {
	xxx_messageInfo_ReaderBaseState.DiscardUnknown(m)
}

var xxx_messageInfo_ReaderBaseState proto.InternalMessageInfo

func (m *ReaderBaseState) GetWorkStarted() int64 {
	if m != nil {
		return m.WorkStarted
	}
	return 0
}

func (m *ReaderBaseState) GetWorkFinished() int64 {
	if m != nil {
		return m.WorkFinished
	}
	return 0
}

func (m *ReaderBaseState) GetNumRecordsProduced() int64 {
	if m != nil {
		return m.NumRecordsProduced
	}
	return 0
}

func (m *ReaderBaseState) GetCurrentWork() []byte {
	if m != nil {
		return m.CurrentWork
	}
	return nil
}

func init() {
	proto.RegisterType((*ReaderBaseState)(nil), "tensorflow.ReaderBaseState")
}

func init() {
	proto.RegisterFile("tensorflow/core/framework/reader_base.proto", fileDescriptor_9d8282e7620a01b6)
}

var fileDescriptor_9d8282e7620a01b6 = []byte{
	// 289 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x74, 0x90, 0xb1, 0x4e, 0xc3, 0x30,
	0x10, 0x40, 0x6b, 0x8a, 0x18, 0xdc, 0x22, 0x50, 0xc4, 0x90, 0xc9, 0x2a, 0xb0, 0x54, 0x42, 0xc4,
	0x48, 0x7c, 0x01, 0x19, 0x98, 0xab, 0x74, 0x40, 0x62, 0x89, 0xdc, 0xf8, 0x92, 0x46, 0x6d, 0x72,
	0xd1, 0xd9, 0x51, 0xc5, 0x47, 0x20, 0xf1, 0x25, 0x7c, 0x07, 0x63, 0x47, 0x46, 0x94, 0xfc, 0x04,
	0x23, 0x8a, 0x1b, 0x11, 0x16, 0xd6, 0x77, 0xef, 0x6c, 0xdd, 0xe3, 0x37, 0x16, 0x4a, 0x83, 0x94,
	0x6e, 0x71, 0x27, 0x13, 0x24, 0x90, 0x29, 0xa9, 0x02, 0x76, 0x48, 0x1b, 0x49, 0xa0, 0x34, 0x50,
	0xbc, 0x52, 0x06, 0x82, 0x8a, 0xd0, 0xa2, 0xc7, 0x07, 0xf9, 0xea, 0x9d, 0xf1, 0xb3, 0xc8, 0x19,
	0xa1, 0x32, 0xb0, 0xb4, 0xca, 0x82, 0x77, 0xc9, 0xa7, 0xdd, 0x66, 0x6c, 0xac, 0x22, 0x0b, 0xda,
	0x67, 0x33, 0x36, 0x1f, 0x47, 0x93, 0x8e, 0x2d, 0x0f, 0xc8, 0xbb, 0xe6, 0xa7, 0x4e, 0x49, 0xf3,
	0x32, 0x37, 0x6b, 0xd0, 0xfe, 0x91, 0x73, 0xdc, 0xde, 0x63, 0xcf, 0xbc, 0x3b, 0x7e, 0x51, 0xd6,
	0x45, 0x4c, 0x90, 0x20, 0x69, 0x13, 0x57, 0x84, 0xba, 0x4e, 0x40, 0xfb, 0x63, 0xe7, 0x7a, 0x65,
	0x5d, 0x44, 0x87, 0xd1, 0xa2, 0x9f, 0x74, 0x3f, 0x27, 0x35, 0x11, 0x94, 0x36, 0xee, 0x5e, 0xf2,
	0x8f, 0x67, 0x6c, 0x3e, 0x8d, 0x26, 0x3d, 0x7b, 0x42, 0xda, 0x84, 0xaf, 0xec, 0xa3, 0x11, 0x6c,
	0xdf, 0x08, 0xf6, 0xd5, 0x08, 0xf6, 0xd6, 0x8a, 0xd1, 0xbe, 0x15, 0xa3, 0xcf, 0x56, 0x8c, 0xb8,
	0x8f, 0x94, 0x05, 0xc3, 0x6d, 0xc1, 0x6f, 0x83, 0xf0, 0x7c, 0x38, 0x71, 0xd1, 0x25, 0x30, 0x0b,
	0xf6, 0x1c, 0x66, 0xb9, 0x5d, 0xd7, 0xab, 0x20, 0xc1, 0x42, 0x3e, 0x54, 0xd5, 0x36, 0x4f, 0x73,
	0x20, 0x99, 0xe1, 0xed, 0x9f, 0x94, 0xf6, 0xa5, 0x02, 0x23, 0xff, 0x6d, 0xfb, 0xcd, 0xd8, 0xea,
	0xc4, 0x35, 0xbd, 0xff, 0x09, 0x00, 0x00, 0xff, 0xff, 0x8a, 0xca, 0xa7, 0x54, 0x82, 0x01, 0x00,
	0x00,
}

func (m *ReaderBaseState) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *ReaderBaseState) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *ReaderBaseState) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if len(m.CurrentWork) > 0 {
		i -= len(m.CurrentWork)
		copy(dAtA[i:], m.CurrentWork)
		i = encodeVarintReaderBase(dAtA, i, uint64(len(m.CurrentWork)))
		i--
		dAtA[i] = 0x22
	}
	if m.NumRecordsProduced != 0 {
		i = encodeVarintReaderBase(dAtA, i, uint64(m.NumRecordsProduced))
		i--
		dAtA[i] = 0x18
	}
	if m.WorkFinished != 0 {
		i = encodeVarintReaderBase(dAtA, i, uint64(m.WorkFinished))
		i--
		dAtA[i] = 0x10
	}
	if m.WorkStarted != 0 {
		i = encodeVarintReaderBase(dAtA, i, uint64(m.WorkStarted))
		i--
		dAtA[i] = 0x8
	}
	return len(dAtA) - i, nil
}

func encodeVarintReaderBase(dAtA []byte, offset int, v uint64) int {
	offset -= sovReaderBase(v)
	base := offset
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return base
}
func (m *ReaderBaseState) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if m.WorkStarted != 0 {
		n += 1 + sovReaderBase(uint64(m.WorkStarted))
	}
	if m.WorkFinished != 0 {
		n += 1 + sovReaderBase(uint64(m.WorkFinished))
	}
	if m.NumRecordsProduced != 0 {
		n += 1 + sovReaderBase(uint64(m.NumRecordsProduced))
	}
	l = len(m.CurrentWork)
	if l > 0 {
		n += 1 + l + sovReaderBase(uint64(l))
	}
	return n
}

func sovReaderBase(x uint64) (n int) {
	return (math_bits.Len64(x|1) + 6) / 7
}
func sozReaderBase(x uint64) (n int) {
	return sovReaderBase(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (m *ReaderBaseState) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowReaderBase
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
			return fmt.Errorf("proto: ReaderBaseState: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: ReaderBaseState: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field WorkStarted", wireType)
			}
			m.WorkStarted = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowReaderBase
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.WorkStarted |= int64(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		case 2:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field WorkFinished", wireType)
			}
			m.WorkFinished = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowReaderBase
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.WorkFinished |= int64(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		case 3:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field NumRecordsProduced", wireType)
			}
			m.NumRecordsProduced = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowReaderBase
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.NumRecordsProduced |= int64(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		case 4:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field CurrentWork", wireType)
			}
			var byteLen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowReaderBase
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				byteLen |= int(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			if byteLen < 0 {
				return ErrInvalidLengthReaderBase
			}
			postIndex := iNdEx + byteLen
			if postIndex < 0 {
				return ErrInvalidLengthReaderBase
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.CurrentWork = append(m.CurrentWork[:0], dAtA[iNdEx:postIndex]...)
			if m.CurrentWork == nil {
				m.CurrentWork = []byte{}
			}
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipReaderBase(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if skippy < 0 {
				return ErrInvalidLengthReaderBase
			}
			if (iNdEx + skippy) < 0 {
				return ErrInvalidLengthReaderBase
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
func skipReaderBase(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	depth := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowReaderBase
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
					return 0, ErrIntOverflowReaderBase
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
					return 0, ErrIntOverflowReaderBase
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
				return 0, ErrInvalidLengthReaderBase
			}
			iNdEx += length
		case 3:
			depth++
		case 4:
			if depth == 0 {
				return 0, ErrUnexpectedEndOfGroupReaderBase
			}
			depth--
		case 5:
			iNdEx += 4
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
		if iNdEx < 0 {
			return 0, ErrInvalidLengthReaderBase
		}
		if depth == 0 {
			return iNdEx, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}

var (
	ErrInvalidLengthReaderBase        = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowReaderBase          = fmt.Errorf("proto: integer overflow")
	ErrUnexpectedEndOfGroupReaderBase = fmt.Errorf("proto: unexpected end of group")
)
