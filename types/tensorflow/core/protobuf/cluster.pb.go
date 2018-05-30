// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: tensorflow/core/protobuf/cluster.proto

/*
	Package protobuf is a generated protocol buffer package.

	It is generated from these files:
		tensorflow/core/protobuf/cluster.proto
		tensorflow/core/protobuf/config.proto
		tensorflow/core/protobuf/debug.proto
		tensorflow/core/protobuf/meta_graph.proto
		tensorflow/core/protobuf/named_tensor.proto
		tensorflow/core/protobuf/rewriter_config.proto
		tensorflow/core/protobuf/saved_model.proto
		tensorflow/core/protobuf/saver.proto

	It has these top-level messages:
		JobDef
		ClusterDef
		GPUOptions
		OptimizerOptions
		GraphOptions
		ThreadPoolOptionProto
		RPCOptions
		ConfigProto
		RunOptions
		RunMetadata
		TensorConnection
		CallableOptions
		DebugTensorWatch
		DebugOptions
		DebuggedSourceFile
		DebuggedSourceFiles
		MetaGraphDef
		CollectionDef
		TensorInfo
		SignatureDef
		AssetFileDef
		NamedTensorProto
		AutoParallelOptions
		RewriterConfig
		SavedModel
		SaverDef
*/
package protobuf

import proto "github.com/gogo/protobuf/proto"
import fmt "fmt"
import math "math"

import io "io"

// Reference imports to suppress errors if they are not otherwise used.
var _ = proto.Marshal
var _ = fmt.Errorf
var _ = math.Inf

// This is a compile-time assertion to ensure that this generated file
// is compatible with the proto package it is being compiled against.
// A compilation error at this line likely means your copy of the
// proto package needs to be updated.
const _ = proto.GoGoProtoPackageIsVersion2 // please upgrade the proto package

// Defines a single job in a TensorFlow cluster.
type JobDef struct {
	// The name of this job.
	Name string `protobuf:"bytes,1,opt,name=name,proto3" json:"name,omitempty"`
	// Mapping from task ID to "hostname:port" string.
	//
	// If the `name` field contains "worker", and the `tasks` map contains a
	// mapping from 7 to "example.org:2222", then the device prefix
	// "/job:worker/task:7" will be assigned to "example.org:2222".
	Tasks map[int32]string `protobuf:"bytes,2,rep,name=tasks" json:"tasks,omitempty" protobuf_key:"varint,1,opt,name=key,proto3" protobuf_val:"bytes,2,opt,name=value,proto3"`
}

func (m *JobDef) Reset()                    { *m = JobDef{} }
func (m *JobDef) String() string            { return proto.CompactTextString(m) }
func (*JobDef) ProtoMessage()               {}
func (*JobDef) Descriptor() ([]byte, []int) { return fileDescriptorCluster, []int{0} }

func (m *JobDef) GetName() string {
	if m != nil {
		return m.Name
	}
	return ""
}

func (m *JobDef) GetTasks() map[int32]string {
	if m != nil {
		return m.Tasks
	}
	return nil
}

// Defines a TensorFlow cluster as a set of jobs.
type ClusterDef struct {
	// The jobs that comprise the cluster.
	Job []*JobDef `protobuf:"bytes,1,rep,name=job" json:"job,omitempty"`
}

func (m *ClusterDef) Reset()                    { *m = ClusterDef{} }
func (m *ClusterDef) String() string            { return proto.CompactTextString(m) }
func (*ClusterDef) ProtoMessage()               {}
func (*ClusterDef) Descriptor() ([]byte, []int) { return fileDescriptorCluster, []int{1} }

func (m *ClusterDef) GetJob() []*JobDef {
	if m != nil {
		return m.Job
	}
	return nil
}

func init() {
	proto.RegisterType((*JobDef)(nil), "tensorflow.JobDef")
	proto.RegisterType((*ClusterDef)(nil), "tensorflow.ClusterDef")
}
func (m *JobDef) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalTo(dAtA)
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *JobDef) MarshalTo(dAtA []byte) (int, error) {
	var i int
	_ = i
	var l int
	_ = l
	if len(m.Name) > 0 {
		dAtA[i] = 0xa
		i++
		i = encodeVarintCluster(dAtA, i, uint64(len(m.Name)))
		i += copy(dAtA[i:], m.Name)
	}
	if len(m.Tasks) > 0 {
		for k, _ := range m.Tasks {
			dAtA[i] = 0x12
			i++
			v := m.Tasks[k]
			mapSize := 1 + sovCluster(uint64(k)) + 1 + len(v) + sovCluster(uint64(len(v)))
			i = encodeVarintCluster(dAtA, i, uint64(mapSize))
			dAtA[i] = 0x8
			i++
			i = encodeVarintCluster(dAtA, i, uint64(k))
			dAtA[i] = 0x12
			i++
			i = encodeVarintCluster(dAtA, i, uint64(len(v)))
			i += copy(dAtA[i:], v)
		}
	}
	return i, nil
}

func (m *ClusterDef) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalTo(dAtA)
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *ClusterDef) MarshalTo(dAtA []byte) (int, error) {
	var i int
	_ = i
	var l int
	_ = l
	if len(m.Job) > 0 {
		for _, msg := range m.Job {
			dAtA[i] = 0xa
			i++
			i = encodeVarintCluster(dAtA, i, uint64(msg.Size()))
			n, err := msg.MarshalTo(dAtA[i:])
			if err != nil {
				return 0, err
			}
			i += n
		}
	}
	return i, nil
}

func encodeVarintCluster(dAtA []byte, offset int, v uint64) int {
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return offset + 1
}
func (m *JobDef) Size() (n int) {
	var l int
	_ = l
	l = len(m.Name)
	if l > 0 {
		n += 1 + l + sovCluster(uint64(l))
	}
	if len(m.Tasks) > 0 {
		for k, v := range m.Tasks {
			_ = k
			_ = v
			mapEntrySize := 1 + sovCluster(uint64(k)) + 1 + len(v) + sovCluster(uint64(len(v)))
			n += mapEntrySize + 1 + sovCluster(uint64(mapEntrySize))
		}
	}
	return n
}

func (m *ClusterDef) Size() (n int) {
	var l int
	_ = l
	if len(m.Job) > 0 {
		for _, e := range m.Job {
			l = e.Size()
			n += 1 + l + sovCluster(uint64(l))
		}
	}
	return n
}

func sovCluster(x uint64) (n int) {
	for {
		n++
		x >>= 7
		if x == 0 {
			break
		}
	}
	return n
}
func sozCluster(x uint64) (n int) {
	return sovCluster(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (m *JobDef) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowCluster
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
			return fmt.Errorf("proto: JobDef: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: JobDef: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Name", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowCluster
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
				return ErrInvalidLengthCluster
			}
			postIndex := iNdEx + intStringLen
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.Name = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		case 2:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Tasks", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowCluster
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
				return ErrInvalidLengthCluster
			}
			postIndex := iNdEx + msglen
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			if m.Tasks == nil {
				m.Tasks = make(map[int32]string)
			}
			var mapkey int32
			var mapvalue string
			for iNdEx < postIndex {
				entryPreIndex := iNdEx
				var wire uint64
				for shift := uint(0); ; shift += 7 {
					if shift >= 64 {
						return ErrIntOverflowCluster
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
					for shift := uint(0); ; shift += 7 {
						if shift >= 64 {
							return ErrIntOverflowCluster
						}
						if iNdEx >= l {
							return io.ErrUnexpectedEOF
						}
						b := dAtA[iNdEx]
						iNdEx++
						mapkey |= (int32(b) & 0x7F) << shift
						if b < 0x80 {
							break
						}
					}
				} else if fieldNum == 2 {
					var stringLenmapvalue uint64
					for shift := uint(0); ; shift += 7 {
						if shift >= 64 {
							return ErrIntOverflowCluster
						}
						if iNdEx >= l {
							return io.ErrUnexpectedEOF
						}
						b := dAtA[iNdEx]
						iNdEx++
						stringLenmapvalue |= (uint64(b) & 0x7F) << shift
						if b < 0x80 {
							break
						}
					}
					intStringLenmapvalue := int(stringLenmapvalue)
					if intStringLenmapvalue < 0 {
						return ErrInvalidLengthCluster
					}
					postStringIndexmapvalue := iNdEx + intStringLenmapvalue
					if postStringIndexmapvalue > l {
						return io.ErrUnexpectedEOF
					}
					mapvalue = string(dAtA[iNdEx:postStringIndexmapvalue])
					iNdEx = postStringIndexmapvalue
				} else {
					iNdEx = entryPreIndex
					skippy, err := skipCluster(dAtA[iNdEx:])
					if err != nil {
						return err
					}
					if skippy < 0 {
						return ErrInvalidLengthCluster
					}
					if (iNdEx + skippy) > postIndex {
						return io.ErrUnexpectedEOF
					}
					iNdEx += skippy
				}
			}
			m.Tasks[mapkey] = mapvalue
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipCluster(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if skippy < 0 {
				return ErrInvalidLengthCluster
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
func (m *ClusterDef) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowCluster
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
			return fmt.Errorf("proto: ClusterDef: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: ClusterDef: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Job", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowCluster
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
				return ErrInvalidLengthCluster
			}
			postIndex := iNdEx + msglen
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.Job = append(m.Job, &JobDef{})
			if err := m.Job[len(m.Job)-1].Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipCluster(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if skippy < 0 {
				return ErrInvalidLengthCluster
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
func skipCluster(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowCluster
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
					return 0, ErrIntOverflowCluster
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
					return 0, ErrIntOverflowCluster
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
				return 0, ErrInvalidLengthCluster
			}
			return iNdEx, nil
		case 3:
			for {
				var innerWire uint64
				var start int = iNdEx
				for shift := uint(0); ; shift += 7 {
					if shift >= 64 {
						return 0, ErrIntOverflowCluster
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
				next, err := skipCluster(dAtA[start:])
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
	ErrInvalidLengthCluster = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowCluster   = fmt.Errorf("proto: integer overflow")
)

func init() { proto.RegisterFile("tensorflow/core/protobuf/cluster.proto", fileDescriptorCluster) }

var fileDescriptorCluster = []byte{
	// 285 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0xe2, 0x52, 0x2b, 0x49, 0xcd, 0x2b,
	0xce, 0x2f, 0x4a, 0xcb, 0xc9, 0x2f, 0xd7, 0x4f, 0xce, 0x2f, 0x4a, 0xd5, 0x2f, 0x28, 0xca, 0x2f,
	0xc9, 0x4f, 0x2a, 0x4d, 0xd3, 0x4f, 0xce, 0x29, 0x2d, 0x2e, 0x49, 0x2d, 0xd2, 0x03, 0x0b, 0x08,
	0x71, 0x21, 0xd4, 0x29, 0x75, 0x33, 0x72, 0xb1, 0x79, 0xe5, 0x27, 0xb9, 0xa4, 0xa6, 0x09, 0x09,
	0x71, 0xb1, 0xe4, 0x25, 0xe6, 0xa6, 0x4a, 0x30, 0x2a, 0x30, 0x6a, 0x70, 0x06, 0x81, 0xd9, 0x42,
	0xc6, 0x5c, 0xac, 0x25, 0x89, 0xc5, 0xd9, 0xc5, 0x12, 0x4c, 0x0a, 0xcc, 0x1a, 0xdc, 0x46, 0xb2,
	0x7a, 0x08, 0xad, 0x7a, 0x10, 0x6d, 0x7a, 0x21, 0x20, 0x79, 0xd7, 0xbc, 0x92, 0xa2, 0xca, 0x20,
	0x88, 0x5a, 0x29, 0x0b, 0x2e, 0x2e, 0x84, 0xa0, 0x90, 0x00, 0x17, 0x73, 0x76, 0x6a, 0x25, 0xd8,
	0x54, 0xd6, 0x20, 0x10, 0x53, 0x48, 0x84, 0x8b, 0xb5, 0x2c, 0x31, 0xa7, 0x34, 0x55, 0x82, 0x09,
	0x6c, 0x13, 0x84, 0x63, 0xc5, 0x64, 0xc1, 0xa8, 0x64, 0xc4, 0xc5, 0xe5, 0x0c, 0x71, 0x2a, 0xc8,
	0x41, 0x2a, 0x5c, 0xcc, 0x59, 0xf9, 0x49, 0x12, 0x8c, 0x60, 0xab, 0x85, 0x30, 0xad, 0x0e, 0x02,
	0x49, 0x3b, 0x35, 0x33, 0x9e, 0x78, 0x24, 0xc7, 0x78, 0xe1, 0x91, 0x1c, 0xe3, 0x83, 0x47, 0x72,
	0x8c, 0x13, 0x1e, 0xcb, 0x31, 0x70, 0x49, 0xe5, 0x17, 0xa5, 0x23, 0x2b, 0x4f, 0xc9, 0x2c, 0x2e,
	0x29, 0x2a, 0xcd, 0x2b, 0xc9, 0xcc, 0x4d, 0x75, 0xe2, 0x85, 0x5a, 0x10, 0x00, 0x0a, 0x8a, 0xe2,
	0x00, 0xc6, 0x28, 0xc7, 0xf4, 0xcc, 0x92, 0x8c, 0xd2, 0x24, 0xbd, 0xe4, 0xfc, 0x5c, 0x7d, 0xc7,
	0x82, 0x82, 0x9c, 0xcc, 0xb4, 0xcc, 0xd4, 0x22, 0xfd, 0xf4, 0x7c, 0x5d, 0xa4, 0xd0, 0x2c, 0xa9,
	0x2c, 0x48, 0x2d, 0xd6, 0xc7, 0x15, 0xbc, 0x3f, 0x18, 0x19, 0x93, 0xd8, 0xc0, 0x1c, 0x63, 0x40,
	0x00, 0x00, 0x00, 0xff, 0xff, 0x64, 0x34, 0x7b, 0xad, 0x84, 0x01, 0x00, 0x00,
}
