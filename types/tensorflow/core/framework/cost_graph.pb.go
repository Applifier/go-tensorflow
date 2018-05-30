// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: tensorflow/core/framework/cost_graph.proto

package framework

import proto "github.com/gogo/protobuf/proto"
import fmt "fmt"
import math "math"

import io "io"

// Reference imports to suppress errors if they are not otherwise used.
var _ = proto.Marshal
var _ = fmt.Errorf
var _ = math.Inf

type CostGraphDef struct {
	Node []*CostGraphDef_Node `protobuf:"bytes,1,rep,name=node" json:"node,omitempty"`
}

func (m *CostGraphDef) Reset()                    { *m = CostGraphDef{} }
func (m *CostGraphDef) String() string            { return proto.CompactTextString(m) }
func (*CostGraphDef) ProtoMessage()               {}
func (*CostGraphDef) Descriptor() ([]byte, []int) { return fileDescriptorCostGraph, []int{0} }

func (m *CostGraphDef) GetNode() []*CostGraphDef_Node {
	if m != nil {
		return m.Node
	}
	return nil
}

type CostGraphDef_Node struct {
	// The name of the node. Names are globally unique.
	Name string `protobuf:"bytes,1,opt,name=name,proto3" json:"name,omitempty"`
	// The device of the node. Can be empty if the node is mapped to the
	// default partition or partitioning hasn't been run yet.
	Device string `protobuf:"bytes,2,opt,name=device,proto3" json:"device,omitempty"`
	// The id of the node. Node ids are only unique inside a partition.
	Id         int32                           `protobuf:"varint,3,opt,name=id,proto3" json:"id,omitempty"`
	InputInfo  []*CostGraphDef_Node_InputInfo  `protobuf:"bytes,4,rep,name=input_info,json=inputInfo" json:"input_info,omitempty"`
	OutputInfo []*CostGraphDef_Node_OutputInfo `protobuf:"bytes,5,rep,name=output_info,json=outputInfo" json:"output_info,omitempty"`
	// Temporary memory used by this node.
	TemporaryMemorySize int64 `protobuf:"varint,6,opt,name=temporary_memory_size,json=temporaryMemorySize,proto3" json:"temporary_memory_size,omitempty"`
	// Persistent memory used by this node.
	PersistentMemorySize       int64 `protobuf:"varint,12,opt,name=persistent_memory_size,json=persistentMemorySize,proto3" json:"persistent_memory_size,omitempty"`
	HostTempMemorySize         int64 `protobuf:"varint,10,opt,name=host_temp_memory_size,json=hostTempMemorySize,proto3" json:"host_temp_memory_size,omitempty"`
	DeviceTempMemorySize       int64 `protobuf:"varint,11,opt,name=device_temp_memory_size,json=deviceTempMemorySize,proto3" json:"device_temp_memory_size,omitempty"`
	DevicePersistentMemorySize int64 `protobuf:"varint,16,opt,name=device_persistent_memory_size,json=devicePersistentMemorySize,proto3" json:"device_persistent_memory_size,omitempty"`
	// Estimate of the computational cost of this node, in microseconds.
	ComputeCost int64 `protobuf:"varint,9,opt,name=compute_cost,json=computeCost,proto3" json:"compute_cost,omitempty"`
	// Analytical estimate of the computational cost of this node, in
	// microseconds.
	ComputeTime int64 `protobuf:"varint,14,opt,name=compute_time,json=computeTime,proto3" json:"compute_time,omitempty"`
	// Analytical estimate of the memory access cost of this node, in
	// microseconds.
	MemoryTime int64 `protobuf:"varint,15,opt,name=memory_time,json=memoryTime,proto3" json:"memory_time,omitempty"`
	// If true, the output is permanent: it can't be discarded, because this
	// node is part of the "final output". Nodes may depend on final nodes.
	IsFinal bool `protobuf:"varint,7,opt,name=is_final,json=isFinal,proto3" json:"is_final,omitempty"`
	// Ids of the control inputs for this node.
	ControlInput []int32 `protobuf:"varint,8,rep,packed,name=control_input,json=controlInput" json:"control_input,omitempty"`
}

func (m *CostGraphDef_Node) Reset()                    { *m = CostGraphDef_Node{} }
func (m *CostGraphDef_Node) String() string            { return proto.CompactTextString(m) }
func (*CostGraphDef_Node) ProtoMessage()               {}
func (*CostGraphDef_Node) Descriptor() ([]byte, []int) { return fileDescriptorCostGraph, []int{0, 0} }

func (m *CostGraphDef_Node) GetName() string {
	if m != nil {
		return m.Name
	}
	return ""
}

func (m *CostGraphDef_Node) GetDevice() string {
	if m != nil {
		return m.Device
	}
	return ""
}

func (m *CostGraphDef_Node) GetId() int32 {
	if m != nil {
		return m.Id
	}
	return 0
}

func (m *CostGraphDef_Node) GetInputInfo() []*CostGraphDef_Node_InputInfo {
	if m != nil {
		return m.InputInfo
	}
	return nil
}

func (m *CostGraphDef_Node) GetOutputInfo() []*CostGraphDef_Node_OutputInfo {
	if m != nil {
		return m.OutputInfo
	}
	return nil
}

func (m *CostGraphDef_Node) GetTemporaryMemorySize() int64 {
	if m != nil {
		return m.TemporaryMemorySize
	}
	return 0
}

func (m *CostGraphDef_Node) GetPersistentMemorySize() int64 {
	if m != nil {
		return m.PersistentMemorySize
	}
	return 0
}

func (m *CostGraphDef_Node) GetHostTempMemorySize() int64 {
	if m != nil {
		return m.HostTempMemorySize
	}
	return 0
}

func (m *CostGraphDef_Node) GetDeviceTempMemorySize() int64 {
	if m != nil {
		return m.DeviceTempMemorySize
	}
	return 0
}

func (m *CostGraphDef_Node) GetDevicePersistentMemorySize() int64 {
	if m != nil {
		return m.DevicePersistentMemorySize
	}
	return 0
}

func (m *CostGraphDef_Node) GetComputeCost() int64 {
	if m != nil {
		return m.ComputeCost
	}
	return 0
}

func (m *CostGraphDef_Node) GetComputeTime() int64 {
	if m != nil {
		return m.ComputeTime
	}
	return 0
}

func (m *CostGraphDef_Node) GetMemoryTime() int64 {
	if m != nil {
		return m.MemoryTime
	}
	return 0
}

func (m *CostGraphDef_Node) GetIsFinal() bool {
	if m != nil {
		return m.IsFinal
	}
	return false
}

func (m *CostGraphDef_Node) GetControlInput() []int32 {
	if m != nil {
		return m.ControlInput
	}
	return nil
}

// Inputs of this node. They must be executed before this node can be
// executed. An input is a particular output of another node, specified
// by the node id and the output index.
type CostGraphDef_Node_InputInfo struct {
	PrecedingNode int32 `protobuf:"varint,1,opt,name=preceding_node,json=precedingNode,proto3" json:"preceding_node,omitempty"`
	PrecedingPort int32 `protobuf:"varint,2,opt,name=preceding_port,json=precedingPort,proto3" json:"preceding_port,omitempty"`
}

func (m *CostGraphDef_Node_InputInfo) Reset()         { *m = CostGraphDef_Node_InputInfo{} }
func (m *CostGraphDef_Node_InputInfo) String() string { return proto.CompactTextString(m) }
func (*CostGraphDef_Node_InputInfo) ProtoMessage()    {}
func (*CostGraphDef_Node_InputInfo) Descriptor() ([]byte, []int) {
	return fileDescriptorCostGraph, []int{0, 0, 0}
}

func (m *CostGraphDef_Node_InputInfo) GetPrecedingNode() int32 {
	if m != nil {
		return m.PrecedingNode
	}
	return 0
}

func (m *CostGraphDef_Node_InputInfo) GetPrecedingPort() int32 {
	if m != nil {
		return m.PrecedingPort
	}
	return 0
}

// Outputs of this node.
type CostGraphDef_Node_OutputInfo struct {
	Size_ int64 `protobuf:"varint,1,opt,name=size,proto3" json:"size,omitempty"`
	// If >= 0, the output is an alias of an input. Note that an alias input
	// may itself be an alias. The algorithm will therefore need to follow
	// those pointers.
	AliasInputPort int64             `protobuf:"varint,2,opt,name=alias_input_port,json=aliasInputPort,proto3" json:"alias_input_port,omitempty"`
	Shape          *TensorShapeProto `protobuf:"bytes,3,opt,name=shape" json:"shape,omitempty"`
	Dtype          DataType          `protobuf:"varint,4,opt,name=dtype,proto3,enum=tensorflow.DataType" json:"dtype,omitempty"`
}

func (m *CostGraphDef_Node_OutputInfo) Reset()         { *m = CostGraphDef_Node_OutputInfo{} }
func (m *CostGraphDef_Node_OutputInfo) String() string { return proto.CompactTextString(m) }
func (*CostGraphDef_Node_OutputInfo) ProtoMessage()    {}
func (*CostGraphDef_Node_OutputInfo) Descriptor() ([]byte, []int) {
	return fileDescriptorCostGraph, []int{0, 0, 1}
}

func (m *CostGraphDef_Node_OutputInfo) GetSize_() int64 {
	if m != nil {
		return m.Size_
	}
	return 0
}

func (m *CostGraphDef_Node_OutputInfo) GetAliasInputPort() int64 {
	if m != nil {
		return m.AliasInputPort
	}
	return 0
}

func (m *CostGraphDef_Node_OutputInfo) GetShape() *TensorShapeProto {
	if m != nil {
		return m.Shape
	}
	return nil
}

func (m *CostGraphDef_Node_OutputInfo) GetDtype() DataType {
	if m != nil {
		return m.Dtype
	}
	return DataType_DT_INVALID
}

func init() {
	proto.RegisterType((*CostGraphDef)(nil), "tensorflow.CostGraphDef")
	proto.RegisterType((*CostGraphDef_Node)(nil), "tensorflow.CostGraphDef.Node")
	proto.RegisterType((*CostGraphDef_Node_InputInfo)(nil), "tensorflow.CostGraphDef.Node.InputInfo")
	proto.RegisterType((*CostGraphDef_Node_OutputInfo)(nil), "tensorflow.CostGraphDef.Node.OutputInfo")
}
func (m *CostGraphDef) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalTo(dAtA)
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *CostGraphDef) MarshalTo(dAtA []byte) (int, error) {
	var i int
	_ = i
	var l int
	_ = l
	if len(m.Node) > 0 {
		for _, msg := range m.Node {
			dAtA[i] = 0xa
			i++
			i = encodeVarintCostGraph(dAtA, i, uint64(msg.Size()))
			n, err := msg.MarshalTo(dAtA[i:])
			if err != nil {
				return 0, err
			}
			i += n
		}
	}
	return i, nil
}

func (m *CostGraphDef_Node) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalTo(dAtA)
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *CostGraphDef_Node) MarshalTo(dAtA []byte) (int, error) {
	var i int
	_ = i
	var l int
	_ = l
	if len(m.Name) > 0 {
		dAtA[i] = 0xa
		i++
		i = encodeVarintCostGraph(dAtA, i, uint64(len(m.Name)))
		i += copy(dAtA[i:], m.Name)
	}
	if len(m.Device) > 0 {
		dAtA[i] = 0x12
		i++
		i = encodeVarintCostGraph(dAtA, i, uint64(len(m.Device)))
		i += copy(dAtA[i:], m.Device)
	}
	if m.Id != 0 {
		dAtA[i] = 0x18
		i++
		i = encodeVarintCostGraph(dAtA, i, uint64(m.Id))
	}
	if len(m.InputInfo) > 0 {
		for _, msg := range m.InputInfo {
			dAtA[i] = 0x22
			i++
			i = encodeVarintCostGraph(dAtA, i, uint64(msg.Size()))
			n, err := msg.MarshalTo(dAtA[i:])
			if err != nil {
				return 0, err
			}
			i += n
		}
	}
	if len(m.OutputInfo) > 0 {
		for _, msg := range m.OutputInfo {
			dAtA[i] = 0x2a
			i++
			i = encodeVarintCostGraph(dAtA, i, uint64(msg.Size()))
			n, err := msg.MarshalTo(dAtA[i:])
			if err != nil {
				return 0, err
			}
			i += n
		}
	}
	if m.TemporaryMemorySize != 0 {
		dAtA[i] = 0x30
		i++
		i = encodeVarintCostGraph(dAtA, i, uint64(m.TemporaryMemorySize))
	}
	if m.IsFinal {
		dAtA[i] = 0x38
		i++
		if m.IsFinal {
			dAtA[i] = 1
		} else {
			dAtA[i] = 0
		}
		i++
	}
	if len(m.ControlInput) > 0 {
		dAtA2 := make([]byte, len(m.ControlInput)*10)
		var j1 int
		for _, num1 := range m.ControlInput {
			num := uint64(num1)
			for num >= 1<<7 {
				dAtA2[j1] = uint8(uint64(num)&0x7f | 0x80)
				num >>= 7
				j1++
			}
			dAtA2[j1] = uint8(num)
			j1++
		}
		dAtA[i] = 0x42
		i++
		i = encodeVarintCostGraph(dAtA, i, uint64(j1))
		i += copy(dAtA[i:], dAtA2[:j1])
	}
	if m.ComputeCost != 0 {
		dAtA[i] = 0x48
		i++
		i = encodeVarintCostGraph(dAtA, i, uint64(m.ComputeCost))
	}
	if m.HostTempMemorySize != 0 {
		dAtA[i] = 0x50
		i++
		i = encodeVarintCostGraph(dAtA, i, uint64(m.HostTempMemorySize))
	}
	if m.DeviceTempMemorySize != 0 {
		dAtA[i] = 0x58
		i++
		i = encodeVarintCostGraph(dAtA, i, uint64(m.DeviceTempMemorySize))
	}
	if m.PersistentMemorySize != 0 {
		dAtA[i] = 0x60
		i++
		i = encodeVarintCostGraph(dAtA, i, uint64(m.PersistentMemorySize))
	}
	if m.ComputeTime != 0 {
		dAtA[i] = 0x70
		i++
		i = encodeVarintCostGraph(dAtA, i, uint64(m.ComputeTime))
	}
	if m.MemoryTime != 0 {
		dAtA[i] = 0x78
		i++
		i = encodeVarintCostGraph(dAtA, i, uint64(m.MemoryTime))
	}
	if m.DevicePersistentMemorySize != 0 {
		dAtA[i] = 0x80
		i++
		dAtA[i] = 0x1
		i++
		i = encodeVarintCostGraph(dAtA, i, uint64(m.DevicePersistentMemorySize))
	}
	return i, nil
}

func (m *CostGraphDef_Node_InputInfo) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalTo(dAtA)
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *CostGraphDef_Node_InputInfo) MarshalTo(dAtA []byte) (int, error) {
	var i int
	_ = i
	var l int
	_ = l
	if m.PrecedingNode != 0 {
		dAtA[i] = 0x8
		i++
		i = encodeVarintCostGraph(dAtA, i, uint64(m.PrecedingNode))
	}
	if m.PrecedingPort != 0 {
		dAtA[i] = 0x10
		i++
		i = encodeVarintCostGraph(dAtA, i, uint64(m.PrecedingPort))
	}
	return i, nil
}

func (m *CostGraphDef_Node_OutputInfo) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalTo(dAtA)
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *CostGraphDef_Node_OutputInfo) MarshalTo(dAtA []byte) (int, error) {
	var i int
	_ = i
	var l int
	_ = l
	if m.Size_ != 0 {
		dAtA[i] = 0x8
		i++
		i = encodeVarintCostGraph(dAtA, i, uint64(m.Size_))
	}
	if m.AliasInputPort != 0 {
		dAtA[i] = 0x10
		i++
		i = encodeVarintCostGraph(dAtA, i, uint64(m.AliasInputPort))
	}
	if m.Shape != nil {
		dAtA[i] = 0x1a
		i++
		i = encodeVarintCostGraph(dAtA, i, uint64(m.Shape.Size()))
		n3, err := m.Shape.MarshalTo(dAtA[i:])
		if err != nil {
			return 0, err
		}
		i += n3
	}
	if m.Dtype != 0 {
		dAtA[i] = 0x20
		i++
		i = encodeVarintCostGraph(dAtA, i, uint64(m.Dtype))
	}
	return i, nil
}

func encodeVarintCostGraph(dAtA []byte, offset int, v uint64) int {
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return offset + 1
}
func (m *CostGraphDef) Size() (n int) {
	var l int
	_ = l
	if len(m.Node) > 0 {
		for _, e := range m.Node {
			l = e.Size()
			n += 1 + l + sovCostGraph(uint64(l))
		}
	}
	return n
}

func (m *CostGraphDef_Node) Size() (n int) {
	var l int
	_ = l
	l = len(m.Name)
	if l > 0 {
		n += 1 + l + sovCostGraph(uint64(l))
	}
	l = len(m.Device)
	if l > 0 {
		n += 1 + l + sovCostGraph(uint64(l))
	}
	if m.Id != 0 {
		n += 1 + sovCostGraph(uint64(m.Id))
	}
	if len(m.InputInfo) > 0 {
		for _, e := range m.InputInfo {
			l = e.Size()
			n += 1 + l + sovCostGraph(uint64(l))
		}
	}
	if len(m.OutputInfo) > 0 {
		for _, e := range m.OutputInfo {
			l = e.Size()
			n += 1 + l + sovCostGraph(uint64(l))
		}
	}
	if m.TemporaryMemorySize != 0 {
		n += 1 + sovCostGraph(uint64(m.TemporaryMemorySize))
	}
	if m.IsFinal {
		n += 2
	}
	if len(m.ControlInput) > 0 {
		l = 0
		for _, e := range m.ControlInput {
			l += sovCostGraph(uint64(e))
		}
		n += 1 + sovCostGraph(uint64(l)) + l
	}
	if m.ComputeCost != 0 {
		n += 1 + sovCostGraph(uint64(m.ComputeCost))
	}
	if m.HostTempMemorySize != 0 {
		n += 1 + sovCostGraph(uint64(m.HostTempMemorySize))
	}
	if m.DeviceTempMemorySize != 0 {
		n += 1 + sovCostGraph(uint64(m.DeviceTempMemorySize))
	}
	if m.PersistentMemorySize != 0 {
		n += 1 + sovCostGraph(uint64(m.PersistentMemorySize))
	}
	if m.ComputeTime != 0 {
		n += 1 + sovCostGraph(uint64(m.ComputeTime))
	}
	if m.MemoryTime != 0 {
		n += 1 + sovCostGraph(uint64(m.MemoryTime))
	}
	if m.DevicePersistentMemorySize != 0 {
		n += 2 + sovCostGraph(uint64(m.DevicePersistentMemorySize))
	}
	return n
}

func (m *CostGraphDef_Node_InputInfo) Size() (n int) {
	var l int
	_ = l
	if m.PrecedingNode != 0 {
		n += 1 + sovCostGraph(uint64(m.PrecedingNode))
	}
	if m.PrecedingPort != 0 {
		n += 1 + sovCostGraph(uint64(m.PrecedingPort))
	}
	return n
}

func (m *CostGraphDef_Node_OutputInfo) Size() (n int) {
	var l int
	_ = l
	if m.Size_ != 0 {
		n += 1 + sovCostGraph(uint64(m.Size_))
	}
	if m.AliasInputPort != 0 {
		n += 1 + sovCostGraph(uint64(m.AliasInputPort))
	}
	if m.Shape != nil {
		l = m.Shape.Size()
		n += 1 + l + sovCostGraph(uint64(l))
	}
	if m.Dtype != 0 {
		n += 1 + sovCostGraph(uint64(m.Dtype))
	}
	return n
}

func sovCostGraph(x uint64) (n int) {
	for {
		n++
		x >>= 7
		if x == 0 {
			break
		}
	}
	return n
}
func sozCostGraph(x uint64) (n int) {
	return sovCostGraph(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (m *CostGraphDef) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowCostGraph
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
			return fmt.Errorf("proto: CostGraphDef: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: CostGraphDef: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Node", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowCostGraph
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
				return ErrInvalidLengthCostGraph
			}
			postIndex := iNdEx + msglen
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.Node = append(m.Node, &CostGraphDef_Node{})
			if err := m.Node[len(m.Node)-1].Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipCostGraph(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if skippy < 0 {
				return ErrInvalidLengthCostGraph
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
func (m *CostGraphDef_Node) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowCostGraph
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
			return fmt.Errorf("proto: Node: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: Node: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Name", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowCostGraph
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
				return ErrInvalidLengthCostGraph
			}
			postIndex := iNdEx + intStringLen
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.Name = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		case 2:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Device", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowCostGraph
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
				return ErrInvalidLengthCostGraph
			}
			postIndex := iNdEx + intStringLen
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.Device = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		case 3:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field Id", wireType)
			}
			m.Id = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowCostGraph
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.Id |= (int32(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		case 4:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field InputInfo", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowCostGraph
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
				return ErrInvalidLengthCostGraph
			}
			postIndex := iNdEx + msglen
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.InputInfo = append(m.InputInfo, &CostGraphDef_Node_InputInfo{})
			if err := m.InputInfo[len(m.InputInfo)-1].Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		case 5:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field OutputInfo", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowCostGraph
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
				return ErrInvalidLengthCostGraph
			}
			postIndex := iNdEx + msglen
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.OutputInfo = append(m.OutputInfo, &CostGraphDef_Node_OutputInfo{})
			if err := m.OutputInfo[len(m.OutputInfo)-1].Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		case 6:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field TemporaryMemorySize", wireType)
			}
			m.TemporaryMemorySize = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowCostGraph
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.TemporaryMemorySize |= (int64(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		case 7:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field IsFinal", wireType)
			}
			var v int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowCostGraph
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				v |= (int(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			m.IsFinal = bool(v != 0)
		case 8:
			if wireType == 0 {
				var v int32
				for shift := uint(0); ; shift += 7 {
					if shift >= 64 {
						return ErrIntOverflowCostGraph
					}
					if iNdEx >= l {
						return io.ErrUnexpectedEOF
					}
					b := dAtA[iNdEx]
					iNdEx++
					v |= (int32(b) & 0x7F) << shift
					if b < 0x80 {
						break
					}
				}
				m.ControlInput = append(m.ControlInput, v)
			} else if wireType == 2 {
				var packedLen int
				for shift := uint(0); ; shift += 7 {
					if shift >= 64 {
						return ErrIntOverflowCostGraph
					}
					if iNdEx >= l {
						return io.ErrUnexpectedEOF
					}
					b := dAtA[iNdEx]
					iNdEx++
					packedLen |= (int(b) & 0x7F) << shift
					if b < 0x80 {
						break
					}
				}
				if packedLen < 0 {
					return ErrInvalidLengthCostGraph
				}
				postIndex := iNdEx + packedLen
				if postIndex > l {
					return io.ErrUnexpectedEOF
				}
				for iNdEx < postIndex {
					var v int32
					for shift := uint(0); ; shift += 7 {
						if shift >= 64 {
							return ErrIntOverflowCostGraph
						}
						if iNdEx >= l {
							return io.ErrUnexpectedEOF
						}
						b := dAtA[iNdEx]
						iNdEx++
						v |= (int32(b) & 0x7F) << shift
						if b < 0x80 {
							break
						}
					}
					m.ControlInput = append(m.ControlInput, v)
				}
			} else {
				return fmt.Errorf("proto: wrong wireType = %d for field ControlInput", wireType)
			}
		case 9:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field ComputeCost", wireType)
			}
			m.ComputeCost = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowCostGraph
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.ComputeCost |= (int64(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		case 10:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field HostTempMemorySize", wireType)
			}
			m.HostTempMemorySize = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowCostGraph
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.HostTempMemorySize |= (int64(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		case 11:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field DeviceTempMemorySize", wireType)
			}
			m.DeviceTempMemorySize = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowCostGraph
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.DeviceTempMemorySize |= (int64(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		case 12:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field PersistentMemorySize", wireType)
			}
			m.PersistentMemorySize = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowCostGraph
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.PersistentMemorySize |= (int64(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		case 14:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field ComputeTime", wireType)
			}
			m.ComputeTime = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowCostGraph
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.ComputeTime |= (int64(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		case 15:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field MemoryTime", wireType)
			}
			m.MemoryTime = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowCostGraph
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.MemoryTime |= (int64(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		case 16:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field DevicePersistentMemorySize", wireType)
			}
			m.DevicePersistentMemorySize = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowCostGraph
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.DevicePersistentMemorySize |= (int64(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		default:
			iNdEx = preIndex
			skippy, err := skipCostGraph(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if skippy < 0 {
				return ErrInvalidLengthCostGraph
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
func (m *CostGraphDef_Node_InputInfo) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowCostGraph
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
			return fmt.Errorf("proto: InputInfo: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: InputInfo: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field PrecedingNode", wireType)
			}
			m.PrecedingNode = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowCostGraph
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.PrecedingNode |= (int32(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		case 2:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field PrecedingPort", wireType)
			}
			m.PrecedingPort = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowCostGraph
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.PrecedingPort |= (int32(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		default:
			iNdEx = preIndex
			skippy, err := skipCostGraph(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if skippy < 0 {
				return ErrInvalidLengthCostGraph
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
func (m *CostGraphDef_Node_OutputInfo) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowCostGraph
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
			return fmt.Errorf("proto: OutputInfo: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: OutputInfo: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field Size_", wireType)
			}
			m.Size_ = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowCostGraph
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.Size_ |= (int64(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		case 2:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field AliasInputPort", wireType)
			}
			m.AliasInputPort = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowCostGraph
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.AliasInputPort |= (int64(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		case 3:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Shape", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowCostGraph
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
				return ErrInvalidLengthCostGraph
			}
			postIndex := iNdEx + msglen
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
		case 4:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field Dtype", wireType)
			}
			m.Dtype = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowCostGraph
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.Dtype |= (DataType(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		default:
			iNdEx = preIndex
			skippy, err := skipCostGraph(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if skippy < 0 {
				return ErrInvalidLengthCostGraph
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
func skipCostGraph(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowCostGraph
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
					return 0, ErrIntOverflowCostGraph
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
					return 0, ErrIntOverflowCostGraph
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
				return 0, ErrInvalidLengthCostGraph
			}
			return iNdEx, nil
		case 3:
			for {
				var innerWire uint64
				var start int = iNdEx
				for shift := uint(0); ; shift += 7 {
					if shift >= 64 {
						return 0, ErrIntOverflowCostGraph
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
				next, err := skipCostGraph(dAtA[start:])
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
	ErrInvalidLengthCostGraph = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowCostGraph   = fmt.Errorf("proto: integer overflow")
)

func init() { proto.RegisterFile("tensorflow/core/framework/cost_graph.proto", fileDescriptorCostGraph) }

var fileDescriptorCostGraph = []byte{
	// 649 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x7c, 0x54, 0xcf, 0x6e, 0xd3, 0x4e,
	0x10, 0xfe, 0x6d, 0xfe, 0x34, 0xcd, 0x24, 0x4d, 0xab, 0xfd, 0xb5, 0xc5, 0x44, 0x34, 0x04, 0x50,
	0x85, 0x55, 0x41, 0x22, 0x02, 0x1c, 0x38, 0x12, 0x4a, 0x51, 0x0f, 0x40, 0xe4, 0xe6, 0x02, 0x17,
	0xcb, 0xb5, 0xd7, 0xc9, 0x8a, 0xd8, 0xbb, 0x5a, 0x6f, 0xa8, 0xd2, 0x33, 0x0f, 0xc0, 0x5b, 0x20,
	0xf1, 0x24, 0x1c, 0x79, 0x04, 0x54, 0x8e, 0xbc, 0x00, 0x47, 0xb4, 0x63, 0xe3, 0x38, 0x40, 0x7b,
	0x1b, 0xcf, 0x7c, 0xdf, 0xcc, 0xb7, 0xc9, 0x37, 0x03, 0x07, 0x9a, 0xc5, 0x89, 0x50, 0xe1, 0x4c,
	0x9c, 0xf5, 0x7d, 0xa1, 0x58, 0x3f, 0x54, 0x5e, 0xc4, 0xce, 0x84, 0x7a, 0xd7, 0xf7, 0x45, 0xa2,
	0xdd, 0x89, 0xf2, 0xe4, 0xb4, 0x27, 0x95, 0xd0, 0x82, 0xc2, 0x12, 0xdb, 0xbe, 0x77, 0x39, 0x2f,
	0xad, 0xb8, 0xc9, 0xd4, 0x93, 0x2c, 0x65, 0xb6, 0xf7, 0xaf, 0x40, 0x2f, 0x24, 0x4b, 0x52, 0xd8,
	0xed, 0x1f, 0x35, 0x68, 0x3e, 0x13, 0x89, 0x7e, 0x61, 0x86, 0x1e, 0xb2, 0x90, 0x3e, 0x80, 0x4a,
	0x2c, 0x02, 0x66, 0x91, 0x6e, 0xd9, 0x6e, 0x0c, 0xf6, 0x7a, 0xcb, 0x36, 0xbd, 0x22, 0xae, 0xf7,
	0x4a, 0x04, 0xcc, 0x41, 0x68, 0xfb, 0x53, 0x0d, 0x2a, 0xe6, 0x93, 0x52, 0xa8, 0xc4, 0x5e, 0x64,
	0xb8, 0xc4, 0xae, 0x3b, 0x18, 0xd3, 0x5d, 0x58, 0x0b, 0xd8, 0x7b, 0xee, 0x33, 0xab, 0x84, 0xd9,
	0xec, 0x8b, 0xb6, 0xa0, 0xc4, 0x03, 0xab, 0xdc, 0x25, 0x76, 0xd5, 0x29, 0xf1, 0x80, 0x1e, 0x01,
	0xf0, 0x58, 0xce, 0xb5, 0xcb, 0xe3, 0x50, 0x58, 0x15, 0x9c, 0x7e, 0xf7, 0xca, 0xe9, 0xbd, 0x63,
	0x83, 0x3f, 0x8e, 0x43, 0xe1, 0xd4, 0xf9, 0xef, 0x90, 0x1e, 0x43, 0x43, 0xcc, 0x75, 0xde, 0xa8,
	0x8a, 0x8d, 0xec, 0xab, 0x1b, 0xbd, 0x46, 0x02, 0x76, 0x02, 0x91, 0xc7, 0x74, 0x00, 0x3b, 0x9a,
	0x45, 0x52, 0x28, 0x4f, 0x2d, 0xdc, 0x88, 0x45, 0x42, 0x2d, 0xdc, 0x84, 0x9f, 0x33, 0x6b, 0xad,
	0x4b, 0xec, 0xb2, 0xf3, 0x7f, 0x5e, 0x7c, 0x89, 0xb5, 0x13, 0x7e, 0xce, 0xe8, 0x75, 0x58, 0xe7,
	0x89, 0x1b, 0xf2, 0xd8, 0x9b, 0x59, 0xb5, 0x2e, 0xb1, 0xd7, 0x9d, 0x1a, 0x4f, 0x8e, 0xcc, 0x27,
	0xbd, 0x03, 0x1b, 0xbe, 0x88, 0xb5, 0x12, 0x33, 0x17, 0xe5, 0x5a, 0xeb, 0xdd, 0xb2, 0x5d, 0x75,
	0x9a, 0x59, 0x12, 0x5f, 0x43, 0x6f, 0x41, 0xd3, 0x17, 0x91, 0x9c, 0x6b, 0xe6, 0x1a, 0x33, 0x58,
	0x75, 0x1c, 0xd5, 0xc8, 0x72, 0x46, 0x3a, 0x7d, 0x0c, 0x3b, 0x53, 0xe3, 0x13, 0x33, 0x7e, 0x45,
	0x16, 0x18, 0xec, 0xb0, 0x64, 0x11, 0x87, 0x1a, 0xc0, 0x98, 0x45, 0xb2, 0xa0, 0xec, 0x09, 0x5c,
	0x4b, 0x7f, 0xfa, 0xbf, 0x89, 0x8d, 0x9c, 0xb8, 0x9d, 0x42, 0xfe, 0xa0, 0x3e, 0x82, 0x5d, 0xc9,
	0x54, 0xc2, 0x13, 0xcd, 0x62, 0xbd, 0xc2, 0x6c, 0xa2, 0xbc, 0xed, 0x65, 0xb5, 0xc0, 0x2a, 0x3c,
	0x45, 0xf3, 0x88, 0x59, 0xad, 0x95, 0xa7, 0x8c, 0x79, 0xc4, 0xe8, 0x4d, 0x68, 0x64, 0xdd, 0x10,
	0xb1, 0x89, 0x08, 0x48, 0x53, 0x08, 0x78, 0x0e, 0x7b, 0x99, 0xe8, 0x4b, 0x04, 0x6c, 0xe5, 0xd2,
	0xdb, 0x29, 0x70, 0xf4, 0x0f, 0x29, 0xed, 0x37, 0x50, 0xcf, 0xcd, 0x42, 0xf7, 0xa1, 0x25, 0x15,
	0xf3, 0x59, 0xc0, 0xe3, 0x89, 0x9b, 0x79, 0xdd, 0xb8, 0x70, 0x23, 0xcf, 0xa2, 0x99, 0x57, 0x60,
	0x52, 0x28, 0x8d, 0x06, 0x2e, 0xc2, 0x46, 0x42, 0xe9, 0xf6, 0x67, 0x02, 0xb0, 0xf4, 0x8f, 0x59,
	0x01, 0xd4, 0x45, 0xf0, 0x29, 0x18, 0x53, 0x1b, 0xb6, 0xbc, 0x19, 0xf7, 0x92, 0xf4, 0x6f, 0x5f,
	0xf6, 0x2a, 0x3b, 0x2d, 0xcc, 0xa3, 0x34, 0xd3, 0x8c, 0x0e, 0xa0, 0x8a, 0x3b, 0x8c, 0x7b, 0xd1,
	0x18, 0xdc, 0x28, 0xda, 0x76, 0x8c, 0xe1, 0x89, 0x29, 0x8f, 0xcc, 0xea, 0x3a, 0x29, 0x94, 0x1e,
	0x40, 0x35, 0x30, 0x1b, 0x6d, 0x55, 0xba, 0xc4, 0x6e, 0x0d, 0xb6, 0x8b, 0x9c, 0x43, 0x4f, 0x7b,
	0xe3, 0x85, 0x64, 0x4e, 0x0a, 0x19, 0x7e, 0x20, 0x5f, 0x2e, 0x3a, 0xe4, 0xeb, 0x45, 0x87, 0x7c,
	0xbb, 0xe8, 0x90, 0x8f, 0xdf, 0x3b, 0xff, 0x81, 0x25, 0xd4, 0xa4, 0x48, 0xc9, 0xcf, 0xc4, 0x70,
	0x33, 0x5f, 0x14, 0x9c, 0x97, 0x8c, 0xc8, 0xdb, 0xe1, 0x84, 0xeb, 0xe9, 0xfc, 0xb4, 0xe7, 0x8b,
	0xa8, 0xff, 0x54, 0xca, 0x19, 0x0f, 0x39, 0x53, 0xfd, 0x89, 0xb8, 0x5f, 0x38, 0x36, 0x78, 0x5d,
	0xfa, 0x97, 0x5e, 0x9f, 0x9f, 0x84, 0x9c, 0xae, 0xe1, 0xed, 0x79, 0xf8, 0x2b, 0x00, 0x00, 0xff,
	0xff, 0xa5, 0xae, 0xb7, 0xd9, 0x0a, 0x05, 0x00, 0x00,
}
