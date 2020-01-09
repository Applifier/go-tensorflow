// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: tensorflow/core/protobuf/saved_model.proto

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

// SavedModel is the high level serialization format for TensorFlow Models.
// See [todo: doc links, similar to session_bundle] for more information.
type SavedModel struct {
	// The schema version of the SavedModel instance. Used for versioning when
	// making future changes to the specification/implementation. Initial value
	// at release will be 1.
	SavedModelSchemaVersion int64 `protobuf:"varint,1,opt,name=saved_model_schema_version,json=savedModelSchemaVersion,proto3" json:"saved_model_schema_version,omitempty"`
	// One or more MetaGraphs.
	MetaGraphs []*MetaGraphDef `protobuf:"bytes,2,rep,name=meta_graphs,json=metaGraphs,proto3" json:"meta_graphs,omitempty"`
}

func (m *SavedModel) Reset()         { *m = SavedModel{} }
func (m *SavedModel) String() string { return proto.CompactTextString(m) }
func (*SavedModel) ProtoMessage()    {}
func (*SavedModel) Descriptor() ([]byte, []int) {
	return fileDescriptor_537826d0bcc2f334, []int{0}
}
func (m *SavedModel) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *SavedModel) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_SavedModel.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalTo(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *SavedModel) XXX_Merge(src proto.Message) {
	xxx_messageInfo_SavedModel.Merge(m, src)
}
func (m *SavedModel) XXX_Size() int {
	return m.Size()
}
func (m *SavedModel) XXX_DiscardUnknown() {
	xxx_messageInfo_SavedModel.DiscardUnknown(m)
}

var xxx_messageInfo_SavedModel proto.InternalMessageInfo

func (m *SavedModel) GetSavedModelSchemaVersion() int64 {
	if m != nil {
		return m.SavedModelSchemaVersion
	}
	return 0
}

func (m *SavedModel) GetMetaGraphs() []*MetaGraphDef {
	if m != nil {
		return m.MetaGraphs
	}
	return nil
}

func init() {
	proto.RegisterType((*SavedModel)(nil), "tensorflow.SavedModel")
}

func init() {
	proto.RegisterFile("tensorflow/core/protobuf/saved_model.proto", fileDescriptor_537826d0bcc2f334)
}

var fileDescriptor_537826d0bcc2f334 = []byte{
	// 270 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0xe2, 0xd2, 0x2a, 0x49, 0xcd, 0x2b,
	0xce, 0x2f, 0x4a, 0xcb, 0xc9, 0x2f, 0xd7, 0x4f, 0xce, 0x2f, 0x4a, 0xd5, 0x2f, 0x28, 0xca, 0x2f,
	0xc9, 0x4f, 0x2a, 0x4d, 0xd3, 0x2f, 0x4e, 0x2c, 0x4b, 0x4d, 0x89, 0xcf, 0xcd, 0x4f, 0x49, 0xcd,
	0xd1, 0x03, 0x0b, 0x0a, 0x71, 0x21, 0xd4, 0x4a, 0x69, 0xe2, 0xd4, 0x97, 0x9b, 0x5a, 0x92, 0x18,
	0x9f, 0x5e, 0x94, 0x58, 0x90, 0x01, 0xd1, 0xa6, 0xd4, 0xc2, 0xc8, 0xc5, 0x15, 0x0c, 0x32, 0xcc,
	0x17, 0x64, 0x96, 0x90, 0x35, 0x97, 0x14, 0x92, 0xd1, 0xf1, 0xc5, 0xc9, 0x19, 0xa9, 0xb9, 0x89,
	0xf1, 0x65, 0xa9, 0x45, 0xc5, 0x99, 0xf9, 0x79, 0x12, 0x8c, 0x0a, 0x8c, 0x1a, 0xcc, 0x41, 0xe2,
	0xc5, 0x70, 0xf5, 0xc1, 0x60, 0xf9, 0x30, 0x88, 0xb4, 0x90, 0x25, 0x17, 0x37, 0xc2, 0xfc, 0x62,
	0x09, 0x26, 0x05, 0x66, 0x0d, 0x6e, 0x23, 0x09, 0x3d, 0x84, 0x63, 0xf4, 0x7c, 0x53, 0x4b, 0x12,
	0xdd, 0x41, 0xb2, 0x2e, 0xa9, 0x69, 0x41, 0x5c, 0xb9, 0x30, 0x5e, 0xb1, 0x53, 0x0f, 0xe3, 0x89,
	0x47, 0x72, 0x8c, 0x17, 0x1e, 0xc9, 0x31, 0x3e, 0x78, 0x24, 0xc7, 0x38, 0xe1, 0xb1, 0x1c, 0xc3,
	0x85, 0xc7, 0x72, 0x0c, 0x37, 0x1e, 0xcb, 0x31, 0x70, 0x49, 0xe4, 0x17, 0xa5, 0x23, 0x9b, 0x91,
	0x56, 0x94, 0x98, 0x9b, 0x5a, 0x9e, 0x5f, 0x94, 0xed, 0x24, 0x80, 0x70, 0x78, 0x00, 0xc8, 0x33,
	0xc5, 0x01, 0x8c, 0x51, 0x8e, 0xe9, 0x99, 0x25, 0x19, 0xa5, 0x49, 0x7a, 0xc9, 0xf9, 0xb9, 0xfa,
	0x8e, 0x05, 0x05, 0x39, 0x99, 0x69, 0x99, 0xa9, 0x45, 0xfa, 0xe9, 0xf9, 0xba, 0x48, 0x41, 0x52,
	0x52, 0x59, 0x90, 0x5a, 0xac, 0x8f, 0x2b, 0x8c, 0x7e, 0x30, 0x32, 0x26, 0xb1, 0x81, 0x39, 0xc6,
	0x80, 0x00, 0x00, 0x00, 0xff, 0xff, 0x27, 0x14, 0xb1, 0x6c, 0x81, 0x01, 0x00, 0x00,
}

func (m *SavedModel) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalTo(dAtA)
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *SavedModel) MarshalTo(dAtA []byte) (int, error) {
	var i int
	_ = i
	var l int
	_ = l
	if m.SavedModelSchemaVersion != 0 {
		dAtA[i] = 0x8
		i++
		i = encodeVarintSavedModel(dAtA, i, uint64(m.SavedModelSchemaVersion))
	}
	if len(m.MetaGraphs) > 0 {
		for _, msg := range m.MetaGraphs {
			dAtA[i] = 0x12
			i++
			i = encodeVarintSavedModel(dAtA, i, uint64(msg.Size()))
			n, err := msg.MarshalTo(dAtA[i:])
			if err != nil {
				return 0, err
			}
			i += n
		}
	}
	return i, nil
}

func encodeVarintSavedModel(dAtA []byte, offset int, v uint64) int {
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return offset + 1
}
func (m *SavedModel) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if m.SavedModelSchemaVersion != 0 {
		n += 1 + sovSavedModel(uint64(m.SavedModelSchemaVersion))
	}
	if len(m.MetaGraphs) > 0 {
		for _, e := range m.MetaGraphs {
			l = e.Size()
			n += 1 + l + sovSavedModel(uint64(l))
		}
	}
	return n
}

func sovSavedModel(x uint64) (n int) {
	for {
		n++
		x >>= 7
		if x == 0 {
			break
		}
	}
	return n
}
func sozSavedModel(x uint64) (n int) {
	return sovSavedModel(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (m *SavedModel) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowSavedModel
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
			return fmt.Errorf("proto: SavedModel: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: SavedModel: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field SavedModelSchemaVersion", wireType)
			}
			m.SavedModelSchemaVersion = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowSavedModel
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.SavedModelSchemaVersion |= (int64(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		case 2:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field MetaGraphs", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowSavedModel
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
				return ErrInvalidLengthSavedModel
			}
			postIndex := iNdEx + msglen
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.MetaGraphs = append(m.MetaGraphs, &MetaGraphDef{})
			if err := m.MetaGraphs[len(m.MetaGraphs)-1].Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipSavedModel(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if skippy < 0 {
				return ErrInvalidLengthSavedModel
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
func skipSavedModel(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowSavedModel
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
					return 0, ErrIntOverflowSavedModel
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
					return 0, ErrIntOverflowSavedModel
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
				return 0, ErrInvalidLengthSavedModel
			}
			return iNdEx, nil
		case 3:
			for {
				var innerWire uint64
				var start int = iNdEx
				for shift := uint(0); ; shift += 7 {
					if shift >= 64 {
						return 0, ErrIntOverflowSavedModel
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
				next, err := skipSavedModel(dAtA[start:])
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
	ErrInvalidLengthSavedModel = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowSavedModel   = fmt.Errorf("proto: integer overflow")
)
