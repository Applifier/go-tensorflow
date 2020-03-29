// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: tensorflow/core/protobuf/saver.proto

package core

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

// A version number that identifies a different on-disk checkpoint format.
// Usually, each subclass of BaseSaverBuilder works with a particular
// version/format.  However, it is possible that the same builder may be
// upgraded to support a newer checkpoint format in the future.
type SaverDef_CheckpointFormatVersion int32

const (
	// Internal legacy format.
	SaverDef_LEGACY SaverDef_CheckpointFormatVersion = 0
	// Deprecated format: tf.Saver() which works with tensorflow::table::Table.
	SaverDef_V1 SaverDef_CheckpointFormatVersion = 1
	// Current format: more efficient.
	SaverDef_V2 SaverDef_CheckpointFormatVersion = 2
)

var SaverDef_CheckpointFormatVersion_name = map[int32]string{
	0: "LEGACY",
	1: "V1",
	2: "V2",
}

var SaverDef_CheckpointFormatVersion_value = map[string]int32{
	"LEGACY": 0,
	"V1":     1,
	"V2":     2,
}

func (x SaverDef_CheckpointFormatVersion) String() string {
	return proto.EnumName(SaverDef_CheckpointFormatVersion_name, int32(x))
}

func (SaverDef_CheckpointFormatVersion) EnumDescriptor() ([]byte, []int) {
	return fileDescriptor_5551ea1a7581c104, []int{0, 0}
}

// Protocol buffer representing the configuration of a Saver.
type SaverDef struct {
	// The name of the tensor in which to specify the filename when saving or
	// restoring a model checkpoint.
	FilenameTensorName string `protobuf:"bytes,1,opt,name=filename_tensor_name,json=filenameTensorName,proto3" json:"filename_tensor_name,omitempty"`
	// The operation to run when saving a model checkpoint.
	SaveTensorName string `protobuf:"bytes,2,opt,name=save_tensor_name,json=saveTensorName,proto3" json:"save_tensor_name,omitempty"`
	// The operation to run when restoring a model checkpoint.
	RestoreOpName string `protobuf:"bytes,3,opt,name=restore_op_name,json=restoreOpName,proto3" json:"restore_op_name,omitempty"`
	// Maximum number of checkpoints to keep.  If 0, no checkpoints are deleted.
	MaxToKeep int32 `protobuf:"varint,4,opt,name=max_to_keep,json=maxToKeep,proto3" json:"max_to_keep,omitempty"`
	// Shard the save files, one per device that has Variable nodes.
	Sharded bool `protobuf:"varint,5,opt,name=sharded,proto3" json:"sharded,omitempty"`
	// How often to keep an additional checkpoint. If not specified, only the last
	// "max_to_keep" checkpoints are kept; if specified, in addition to keeping
	// the last "max_to_keep" checkpoints, an additional checkpoint will be kept
	// for every n hours of training.
	KeepCheckpointEveryNHours float32                          `protobuf:"fixed32,6,opt,name=keep_checkpoint_every_n_hours,json=keepCheckpointEveryNHours,proto3" json:"keep_checkpoint_every_n_hours,omitempty"`
	Version                   SaverDef_CheckpointFormatVersion `protobuf:"varint,7,opt,name=version,proto3,enum=tensorflow.SaverDef_CheckpointFormatVersion" json:"version,omitempty"`
}

func (m *SaverDef) Reset()         { *m = SaverDef{} }
func (m *SaverDef) String() string { return proto.CompactTextString(m) }
func (*SaverDef) ProtoMessage()    {}
func (*SaverDef) Descriptor() ([]byte, []int) {
	return fileDescriptor_5551ea1a7581c104, []int{0}
}
func (m *SaverDef) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *SaverDef) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_SaverDef.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *SaverDef) XXX_Merge(src proto.Message) {
	xxx_messageInfo_SaverDef.Merge(m, src)
}
func (m *SaverDef) XXX_Size() int {
	return m.Size()
}
func (m *SaverDef) XXX_DiscardUnknown() {
	xxx_messageInfo_SaverDef.DiscardUnknown(m)
}

var xxx_messageInfo_SaverDef proto.InternalMessageInfo

func (m *SaverDef) GetFilenameTensorName() string {
	if m != nil {
		return m.FilenameTensorName
	}
	return ""
}

func (m *SaverDef) GetSaveTensorName() string {
	if m != nil {
		return m.SaveTensorName
	}
	return ""
}

func (m *SaverDef) GetRestoreOpName() string {
	if m != nil {
		return m.RestoreOpName
	}
	return ""
}

func (m *SaverDef) GetMaxToKeep() int32 {
	if m != nil {
		return m.MaxToKeep
	}
	return 0
}

func (m *SaverDef) GetSharded() bool {
	if m != nil {
		return m.Sharded
	}
	return false
}

func (m *SaverDef) GetKeepCheckpointEveryNHours() float32 {
	if m != nil {
		return m.KeepCheckpointEveryNHours
	}
	return 0
}

func (m *SaverDef) GetVersion() SaverDef_CheckpointFormatVersion {
	if m != nil {
		return m.Version
	}
	return SaverDef_LEGACY
}

func init() {
	proto.RegisterEnum("tensorflow.SaverDef_CheckpointFormatVersion", SaverDef_CheckpointFormatVersion_name, SaverDef_CheckpointFormatVersion_value)
	proto.RegisterType((*SaverDef)(nil), "tensorflow.SaverDef")
}

func init() {
	proto.RegisterFile("tensorflow/core/protobuf/saver.proto", fileDescriptor_5551ea1a7581c104)
}

var fileDescriptor_5551ea1a7581c104 = []byte{
	// 405 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x74, 0x92, 0x41, 0x6b, 0xd4, 0x40,
	0x14, 0xc7, 0x77, 0x52, 0x9b, 0x6d, 0xa7, 0x58, 0xc3, 0x28, 0x18, 0x0f, 0x86, 0x50, 0x44, 0x72,
	0xd0, 0x44, 0x2b, 0x82, 0x47, 0xdb, 0xda, 0x2a, 0x28, 0xb5, 0xc4, 0x52, 0xd0, 0xcb, 0x90, 0x4d,
	0x5f, 0x36, 0xa1, 0x49, 0xde, 0x30, 0x33, 0x59, 0xbb, 0x1f, 0xc1, 0x9b, 0x1f, 0xcb, 0xe3, 0x1e,
	0x3d, 0xca, 0xee, 0x97, 0xf0, 0x28, 0x33, 0x6b, 0xdc, 0x55, 0xf0, 0x34, 0xf3, 0xde, 0xff, 0xf7,
	0x1f, 0x1e, 0x6f, 0xfe, 0xf4, 0x81, 0x86, 0x56, 0xa1, 0x2c, 0x6a, 0xfc, 0x9c, 0xe4, 0x28, 0x21,
	0x11, 0x12, 0x35, 0x8e, 0xba, 0x22, 0x51, 0xd9, 0x04, 0x64, 0x6c, 0x4b, 0x46, 0x57, 0xd4, 0xde,
	0x97, 0x0d, 0xba, 0xf5, 0xc1, 0x68, 0xaf, 0xa0, 0x60, 0x4f, 0xe8, 0x9d, 0xa2, 0xaa, 0xa1, 0xcd,
	0x1a, 0xe0, 0x4b, 0x86, 0x9b, 0xbb, 0x4f, 0x42, 0x12, 0x6d, 0xa7, 0xac, 0xd7, 0xce, 0xad, 0x74,
	0x9a, 0x35, 0xc0, 0x22, 0xea, 0x99, 0x97, 0xff, 0xa2, 0x1d, 0x4b, 0xef, 0x9a, 0xfe, 0x1a, 0xf9,
	0x90, 0xde, 0x92, 0xa0, 0x34, 0x4a, 0xe0, 0x28, 0x96, 0xe0, 0x86, 0x05, 0x6f, 0xfe, 0x6e, 0xbf,
	0x17, 0x96, 0x0b, 0xe8, 0x4e, 0x93, 0x5d, 0x73, 0x8d, 0xfc, 0x0a, 0x40, 0xf8, 0x37, 0x42, 0x12,
	0x6d, 0xa6, 0xdb, 0x4d, 0x76, 0x7d, 0x8e, 0x6f, 0x01, 0x04, 0xf3, 0xe9, 0x50, 0x95, 0x99, 0xbc,
	0x84, 0x4b, 0x7f, 0x33, 0x24, 0xd1, 0x56, 0xda, 0x97, 0xec, 0x25, 0xbd, 0x6f, 0x2c, 0x3c, 0x2f,
	0x21, 0xbf, 0x12, 0x58, 0xb5, 0x9a, 0xc3, 0x04, 0xe4, 0x94, 0xb7, 0xbc, 0xc4, 0x4e, 0x2a, 0xdf,
	0x0d, 0x49, 0xe4, 0xa4, 0xf7, 0x0c, 0x74, 0xf4, 0x87, 0x39, 0x36, 0xc8, 0xe9, 0x1b, 0x03, 0xb0,
	0x13, 0x3a, 0x9c, 0x80, 0x54, 0x15, 0xb6, 0xfe, 0x30, 0x24, 0xd1, 0xee, 0xfe, 0xa3, 0x78, 0xb5,
	0xaa, 0xb8, 0x5f, 0x53, 0xbc, 0x32, 0x9f, 0xa0, 0x6c, 0x32, 0x7d, 0xb1, 0xf4, 0xa4, 0xbd, 0x79,
	0xef, 0x39, 0xbd, 0xfb, 0x1f, 0x86, 0x51, 0xea, 0xbe, 0x3b, 0x7e, 0x7d, 0x70, 0xf4, 0xd1, 0x1b,
	0x30, 0x97, 0x3a, 0x17, 0x4f, 0x3d, 0x62, 0xcf, 0x7d, 0xcf, 0x39, 0x9c, 0x7e, 0x9b, 0x07, 0x64,
	0x36, 0x0f, 0xc8, 0x8f, 0x79, 0x40, 0xbe, 0x2e, 0x82, 0xc1, 0x6c, 0x11, 0x0c, 0xbe, 0x2f, 0x82,
	0x01, 0xbd, 0x8d, 0x72, 0xbc, 0x3e, 0x4a, 0xa7, 0xab, 0xfa, 0x70, 0xc7, 0x0e, 0x74, 0x66, 0xbe,
	0x54, 0x9d, 0x91, 0x4f, 0x2f, 0xc6, 0x95, 0x2e, 0xbb, 0x51, 0x9c, 0x63, 0x93, 0x1c, 0x08, 0x51,
	0x57, 0x45, 0x05, 0x32, 0x19, 0xe3, 0xe3, 0xb5, 0x4c, 0xe8, 0xa9, 0x00, 0x95, 0xfc, 0x13, 0x92,
	0x9f, 0x84, 0x8c, 0x5c, 0x9b, 0x8c, 0x67, 0xbf, 0x02, 0x00, 0x00, 0xff, 0xff, 0xf9, 0x66, 0x84,
	0x8b, 0x41, 0x02, 0x00, 0x00,
}

func (m *SaverDef) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *SaverDef) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *SaverDef) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if m.Version != 0 {
		i = encodeVarintSaver(dAtA, i, uint64(m.Version))
		i--
		dAtA[i] = 0x38
	}
	if m.KeepCheckpointEveryNHours != 0 {
		i -= 4
		encoding_binary.LittleEndian.PutUint32(dAtA[i:], uint32(math.Float32bits(float32(m.KeepCheckpointEveryNHours))))
		i--
		dAtA[i] = 0x35
	}
	if m.Sharded {
		i--
		if m.Sharded {
			dAtA[i] = 1
		} else {
			dAtA[i] = 0
		}
		i--
		dAtA[i] = 0x28
	}
	if m.MaxToKeep != 0 {
		i = encodeVarintSaver(dAtA, i, uint64(m.MaxToKeep))
		i--
		dAtA[i] = 0x20
	}
	if len(m.RestoreOpName) > 0 {
		i -= len(m.RestoreOpName)
		copy(dAtA[i:], m.RestoreOpName)
		i = encodeVarintSaver(dAtA, i, uint64(len(m.RestoreOpName)))
		i--
		dAtA[i] = 0x1a
	}
	if len(m.SaveTensorName) > 0 {
		i -= len(m.SaveTensorName)
		copy(dAtA[i:], m.SaveTensorName)
		i = encodeVarintSaver(dAtA, i, uint64(len(m.SaveTensorName)))
		i--
		dAtA[i] = 0x12
	}
	if len(m.FilenameTensorName) > 0 {
		i -= len(m.FilenameTensorName)
		copy(dAtA[i:], m.FilenameTensorName)
		i = encodeVarintSaver(dAtA, i, uint64(len(m.FilenameTensorName)))
		i--
		dAtA[i] = 0xa
	}
	return len(dAtA) - i, nil
}

func encodeVarintSaver(dAtA []byte, offset int, v uint64) int {
	offset -= sovSaver(v)
	base := offset
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return base
}
func (m *SaverDef) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	l = len(m.FilenameTensorName)
	if l > 0 {
		n += 1 + l + sovSaver(uint64(l))
	}
	l = len(m.SaveTensorName)
	if l > 0 {
		n += 1 + l + sovSaver(uint64(l))
	}
	l = len(m.RestoreOpName)
	if l > 0 {
		n += 1 + l + sovSaver(uint64(l))
	}
	if m.MaxToKeep != 0 {
		n += 1 + sovSaver(uint64(m.MaxToKeep))
	}
	if m.Sharded {
		n += 2
	}
	if m.KeepCheckpointEveryNHours != 0 {
		n += 5
	}
	if m.Version != 0 {
		n += 1 + sovSaver(uint64(m.Version))
	}
	return n
}

func sovSaver(x uint64) (n int) {
	return (math_bits.Len64(x|1) + 6) / 7
}
func sozSaver(x uint64) (n int) {
	return sovSaver(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (m *SaverDef) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowSaver
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
			return fmt.Errorf("proto: SaverDef: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: SaverDef: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field FilenameTensorName", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowSaver
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
				return ErrInvalidLengthSaver
			}
			postIndex := iNdEx + intStringLen
			if postIndex < 0 {
				return ErrInvalidLengthSaver
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.FilenameTensorName = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		case 2:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field SaveTensorName", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowSaver
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
				return ErrInvalidLengthSaver
			}
			postIndex := iNdEx + intStringLen
			if postIndex < 0 {
				return ErrInvalidLengthSaver
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.SaveTensorName = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		case 3:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field RestoreOpName", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowSaver
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
				return ErrInvalidLengthSaver
			}
			postIndex := iNdEx + intStringLen
			if postIndex < 0 {
				return ErrInvalidLengthSaver
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.RestoreOpName = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		case 4:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field MaxToKeep", wireType)
			}
			m.MaxToKeep = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowSaver
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.MaxToKeep |= int32(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		case 5:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field Sharded", wireType)
			}
			var v int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowSaver
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				v |= int(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			m.Sharded = bool(v != 0)
		case 6:
			if wireType != 5 {
				return fmt.Errorf("proto: wrong wireType = %d for field KeepCheckpointEveryNHours", wireType)
			}
			var v uint32
			if (iNdEx + 4) > l {
				return io.ErrUnexpectedEOF
			}
			v = uint32(encoding_binary.LittleEndian.Uint32(dAtA[iNdEx:]))
			iNdEx += 4
			m.KeepCheckpointEveryNHours = float32(math.Float32frombits(v))
		case 7:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field Version", wireType)
			}
			m.Version = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowSaver
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.Version |= SaverDef_CheckpointFormatVersion(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		default:
			iNdEx = preIndex
			skippy, err := skipSaver(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if skippy < 0 {
				return ErrInvalidLengthSaver
			}
			if (iNdEx + skippy) < 0 {
				return ErrInvalidLengthSaver
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
func skipSaver(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	depth := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowSaver
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
					return 0, ErrIntOverflowSaver
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
					return 0, ErrIntOverflowSaver
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
				return 0, ErrInvalidLengthSaver
			}
			iNdEx += length
		case 3:
			depth++
		case 4:
			if depth == 0 {
				return 0, ErrUnexpectedEndOfGroupSaver
			}
			depth--
		case 5:
			iNdEx += 4
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
		if iNdEx < 0 {
			return 0, ErrInvalidLengthSaver
		}
		if depth == 0 {
			return iNdEx, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}

var (
	ErrInvalidLengthSaver        = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowSaver          = fmt.Errorf("proto: integer overflow")
	ErrUnexpectedEndOfGroupSaver = fmt.Errorf("proto: unexpected end of group")
)
