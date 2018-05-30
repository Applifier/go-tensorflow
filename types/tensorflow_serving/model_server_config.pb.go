// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: tensorflow_serving/model_server_config.proto

package tensorflow_serving

import proto "github.com/gogo/protobuf/proto"
import fmt "fmt"
import math "math"
import google_protobuf "github.com/gogo/protobuf/types"

import io "io"

// Reference imports to suppress errors if they are not otherwise used.
var _ = proto.Marshal
var _ = fmt.Errorf
var _ = math.Inf

// The type of model.
// TODO(b/31336131): DEPRECATED.
type ModelType int32

const (
	ModelType_MODEL_TYPE_UNSPECIFIED ModelType = 0
	ModelType_TENSORFLOW             ModelType = 1
	ModelType_OTHER                  ModelType = 2
)

var ModelType_name = map[int32]string{
	0: "MODEL_TYPE_UNSPECIFIED",
	1: "TENSORFLOW",
	2: "OTHER",
}
var ModelType_value = map[string]int32{
	"MODEL_TYPE_UNSPECIFIED": 0,
	"TENSORFLOW":             1,
	"OTHER":                  2,
}

func (x ModelType) String() string {
	return proto.EnumName(ModelType_name, int32(x))
}
func (ModelType) EnumDescriptor() ([]byte, []int) { return fileDescriptorModelServerConfig, []int{0} }

// Common configuration for loading a model being served.
type ModelConfig struct {
	// Name of the model.
	Name string `protobuf:"bytes,1,opt,name=name,proto3" json:"name,omitempty"`
	// Base path to the model, excluding the version directory.
	// E.g> for a model at /foo/bar/my_model/123, where 123 is the version, the
	// base path is /foo/bar/my_model.
	//
	// (This can be changed once a model is in serving, *if* the underlying data
	// remains the same. Otherwise there are no guarantees about whether the old
	// or new data will be used for model versions currently loaded.)
	BasePath string `protobuf:"bytes,2,opt,name=base_path,json=basePath,proto3" json:"base_path,omitempty"`
	// Type of model.
	// TODO(b/31336131): DEPRECATED. Please use 'model_platform' instead.
	ModelType ModelType `protobuf:"varint,3,opt,name=model_type,json=modelType,proto3,enum=tensorflow.serving.ModelType" json:"model_type,omitempty"`
	// Type of model (e.g. "tensorflow").
	//
	// (This cannot be changed once a model is in serving.)
	ModelPlatform string `protobuf:"bytes,4,opt,name=model_platform,json=modelPlatform,proto3" json:"model_platform,omitempty"`
	// Version policy for the model indicating which version(s) of the model to
	// load and make available for serving simultaneously.
	// The default option is to serve only the latest version of the model.
	//
	// (This can be changed once a model is in serving.)
	ModelVersionPolicy *FileSystemStoragePathSourceConfig_ServableVersionPolicy `protobuf:"bytes,7,opt,name=model_version_policy,json=modelVersionPolicy" json:"model_version_policy,omitempty"`
	// Configures logging requests and responses, to the model.
	//
	// (This can be changed once a model is in serving.)
	LoggingConfig *LoggingConfig `protobuf:"bytes,6,opt,name=logging_config,json=loggingConfig" json:"logging_config,omitempty"`
}

func (m *ModelConfig) Reset()                    { *m = ModelConfig{} }
func (m *ModelConfig) String() string            { return proto.CompactTextString(m) }
func (*ModelConfig) ProtoMessage()               {}
func (*ModelConfig) Descriptor() ([]byte, []int) { return fileDescriptorModelServerConfig, []int{0} }

func (m *ModelConfig) GetName() string {
	if m != nil {
		return m.Name
	}
	return ""
}

func (m *ModelConfig) GetBasePath() string {
	if m != nil {
		return m.BasePath
	}
	return ""
}

func (m *ModelConfig) GetModelType() ModelType {
	if m != nil {
		return m.ModelType
	}
	return ModelType_MODEL_TYPE_UNSPECIFIED
}

func (m *ModelConfig) GetModelPlatform() string {
	if m != nil {
		return m.ModelPlatform
	}
	return ""
}

func (m *ModelConfig) GetModelVersionPolicy() *FileSystemStoragePathSourceConfig_ServableVersionPolicy {
	if m != nil {
		return m.ModelVersionPolicy
	}
	return nil
}

func (m *ModelConfig) GetLoggingConfig() *LoggingConfig {
	if m != nil {
		return m.LoggingConfig
	}
	return nil
}

// Static list of models to be loaded for serving.
type ModelConfigList struct {
	Config []*ModelConfig `protobuf:"bytes,1,rep,name=config" json:"config,omitempty"`
}

func (m *ModelConfigList) Reset()                    { *m = ModelConfigList{} }
func (m *ModelConfigList) String() string            { return proto.CompactTextString(m) }
func (*ModelConfigList) ProtoMessage()               {}
func (*ModelConfigList) Descriptor() ([]byte, []int) { return fileDescriptorModelServerConfig, []int{1} }

func (m *ModelConfigList) GetConfig() []*ModelConfig {
	if m != nil {
		return m.Config
	}
	return nil
}

// ModelServer config.
type ModelServerConfig struct {
	// ModelServer takes either a static file-based model config list or an Any
	// proto representing custom model config that is fetched dynamically at
	// runtime (through network RPC, custom service, etc.).
	//
	// Types that are valid to be assigned to Config:
	//	*ModelServerConfig_ModelConfigList
	//	*ModelServerConfig_CustomModelConfig
	Config isModelServerConfig_Config `protobuf_oneof:"config"`
}

func (m *ModelServerConfig) Reset()         { *m = ModelServerConfig{} }
func (m *ModelServerConfig) String() string { return proto.CompactTextString(m) }
func (*ModelServerConfig) ProtoMessage()    {}
func (*ModelServerConfig) Descriptor() ([]byte, []int) {
	return fileDescriptorModelServerConfig, []int{2}
}

type isModelServerConfig_Config interface {
	isModelServerConfig_Config()
	MarshalTo([]byte) (int, error)
	Size() int
}

type ModelServerConfig_ModelConfigList struct {
	ModelConfigList *ModelConfigList `protobuf:"bytes,1,opt,name=model_config_list,json=modelConfigList,oneof"`
}
type ModelServerConfig_CustomModelConfig struct {
	CustomModelConfig *google_protobuf.Any `protobuf:"bytes,2,opt,name=custom_model_config,json=customModelConfig,oneof"`
}

func (*ModelServerConfig_ModelConfigList) isModelServerConfig_Config()   {}
func (*ModelServerConfig_CustomModelConfig) isModelServerConfig_Config() {}

func (m *ModelServerConfig) GetConfig() isModelServerConfig_Config {
	if m != nil {
		return m.Config
	}
	return nil
}

func (m *ModelServerConfig) GetModelConfigList() *ModelConfigList {
	if x, ok := m.GetConfig().(*ModelServerConfig_ModelConfigList); ok {
		return x.ModelConfigList
	}
	return nil
}

func (m *ModelServerConfig) GetCustomModelConfig() *google_protobuf.Any {
	if x, ok := m.GetConfig().(*ModelServerConfig_CustomModelConfig); ok {
		return x.CustomModelConfig
	}
	return nil
}

// XXX_OneofFuncs is for the internal use of the proto package.
func (*ModelServerConfig) XXX_OneofFuncs() (func(msg proto.Message, b *proto.Buffer) error, func(msg proto.Message, tag, wire int, b *proto.Buffer) (bool, error), func(msg proto.Message) (n int), []interface{}) {
	return _ModelServerConfig_OneofMarshaler, _ModelServerConfig_OneofUnmarshaler, _ModelServerConfig_OneofSizer, []interface{}{
		(*ModelServerConfig_ModelConfigList)(nil),
		(*ModelServerConfig_CustomModelConfig)(nil),
	}
}

func _ModelServerConfig_OneofMarshaler(msg proto.Message, b *proto.Buffer) error {
	m := msg.(*ModelServerConfig)
	// config
	switch x := m.Config.(type) {
	case *ModelServerConfig_ModelConfigList:
		_ = b.EncodeVarint(1<<3 | proto.WireBytes)
		if err := b.EncodeMessage(x.ModelConfigList); err != nil {
			return err
		}
	case *ModelServerConfig_CustomModelConfig:
		_ = b.EncodeVarint(2<<3 | proto.WireBytes)
		if err := b.EncodeMessage(x.CustomModelConfig); err != nil {
			return err
		}
	case nil:
	default:
		return fmt.Errorf("ModelServerConfig.Config has unexpected type %T", x)
	}
	return nil
}

func _ModelServerConfig_OneofUnmarshaler(msg proto.Message, tag, wire int, b *proto.Buffer) (bool, error) {
	m := msg.(*ModelServerConfig)
	switch tag {
	case 1: // config.model_config_list
		if wire != proto.WireBytes {
			return true, proto.ErrInternalBadWireType
		}
		msg := new(ModelConfigList)
		err := b.DecodeMessage(msg)
		m.Config = &ModelServerConfig_ModelConfigList{msg}
		return true, err
	case 2: // config.custom_model_config
		if wire != proto.WireBytes {
			return true, proto.ErrInternalBadWireType
		}
		msg := new(google_protobuf.Any)
		err := b.DecodeMessage(msg)
		m.Config = &ModelServerConfig_CustomModelConfig{msg}
		return true, err
	default:
		return false, nil
	}
}

func _ModelServerConfig_OneofSizer(msg proto.Message) (n int) {
	m := msg.(*ModelServerConfig)
	// config
	switch x := m.Config.(type) {
	case *ModelServerConfig_ModelConfigList:
		s := proto.Size(x.ModelConfigList)
		n += proto.SizeVarint(1<<3 | proto.WireBytes)
		n += proto.SizeVarint(uint64(s))
		n += s
	case *ModelServerConfig_CustomModelConfig:
		s := proto.Size(x.CustomModelConfig)
		n += proto.SizeVarint(2<<3 | proto.WireBytes)
		n += proto.SizeVarint(uint64(s))
		n += s
	case nil:
	default:
		panic(fmt.Sprintf("proto: unexpected type %T in oneof", x))
	}
	return n
}

func init() {
	proto.RegisterType((*ModelConfig)(nil), "tensorflow.serving.ModelConfig")
	proto.RegisterType((*ModelConfigList)(nil), "tensorflow.serving.ModelConfigList")
	proto.RegisterType((*ModelServerConfig)(nil), "tensorflow.serving.ModelServerConfig")
	proto.RegisterEnum("tensorflow.serving.ModelType", ModelType_name, ModelType_value)
}
func (m *ModelConfig) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalTo(dAtA)
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *ModelConfig) MarshalTo(dAtA []byte) (int, error) {
	var i int
	_ = i
	var l int
	_ = l
	if len(m.Name) > 0 {
		dAtA[i] = 0xa
		i++
		i = encodeVarintModelServerConfig(dAtA, i, uint64(len(m.Name)))
		i += copy(dAtA[i:], m.Name)
	}
	if len(m.BasePath) > 0 {
		dAtA[i] = 0x12
		i++
		i = encodeVarintModelServerConfig(dAtA, i, uint64(len(m.BasePath)))
		i += copy(dAtA[i:], m.BasePath)
	}
	if m.ModelType != 0 {
		dAtA[i] = 0x18
		i++
		i = encodeVarintModelServerConfig(dAtA, i, uint64(m.ModelType))
	}
	if len(m.ModelPlatform) > 0 {
		dAtA[i] = 0x22
		i++
		i = encodeVarintModelServerConfig(dAtA, i, uint64(len(m.ModelPlatform)))
		i += copy(dAtA[i:], m.ModelPlatform)
	}
	if m.LoggingConfig != nil {
		dAtA[i] = 0x32
		i++
		i = encodeVarintModelServerConfig(dAtA, i, uint64(m.LoggingConfig.Size()))
		n1, err := m.LoggingConfig.MarshalTo(dAtA[i:])
		if err != nil {
			return 0, err
		}
		i += n1
	}
	if m.ModelVersionPolicy != nil {
		dAtA[i] = 0x3a
		i++
		i = encodeVarintModelServerConfig(dAtA, i, uint64(m.ModelVersionPolicy.Size()))
		n2, err := m.ModelVersionPolicy.MarshalTo(dAtA[i:])
		if err != nil {
			return 0, err
		}
		i += n2
	}
	return i, nil
}

func (m *ModelConfigList) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalTo(dAtA)
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *ModelConfigList) MarshalTo(dAtA []byte) (int, error) {
	var i int
	_ = i
	var l int
	_ = l
	if len(m.Config) > 0 {
		for _, msg := range m.Config {
			dAtA[i] = 0xa
			i++
			i = encodeVarintModelServerConfig(dAtA, i, uint64(msg.Size()))
			n, err := msg.MarshalTo(dAtA[i:])
			if err != nil {
				return 0, err
			}
			i += n
		}
	}
	return i, nil
}

func (m *ModelServerConfig) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalTo(dAtA)
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *ModelServerConfig) MarshalTo(dAtA []byte) (int, error) {
	var i int
	_ = i
	var l int
	_ = l
	if m.Config != nil {
		nn3, err := m.Config.MarshalTo(dAtA[i:])
		if err != nil {
			return 0, err
		}
		i += nn3
	}
	return i, nil
}

func (m *ModelServerConfig_ModelConfigList) MarshalTo(dAtA []byte) (int, error) {
	i := 0
	if m.ModelConfigList != nil {
		dAtA[i] = 0xa
		i++
		i = encodeVarintModelServerConfig(dAtA, i, uint64(m.ModelConfigList.Size()))
		n4, err := m.ModelConfigList.MarshalTo(dAtA[i:])
		if err != nil {
			return 0, err
		}
		i += n4
	}
	return i, nil
}
func (m *ModelServerConfig_CustomModelConfig) MarshalTo(dAtA []byte) (int, error) {
	i := 0
	if m.CustomModelConfig != nil {
		dAtA[i] = 0x12
		i++
		i = encodeVarintModelServerConfig(dAtA, i, uint64(m.CustomModelConfig.Size()))
		n5, err := m.CustomModelConfig.MarshalTo(dAtA[i:])
		if err != nil {
			return 0, err
		}
		i += n5
	}
	return i, nil
}
func encodeVarintModelServerConfig(dAtA []byte, offset int, v uint64) int {
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return offset + 1
}
func (m *ModelConfig) Size() (n int) {
	var l int
	_ = l
	l = len(m.Name)
	if l > 0 {
		n += 1 + l + sovModelServerConfig(uint64(l))
	}
	l = len(m.BasePath)
	if l > 0 {
		n += 1 + l + sovModelServerConfig(uint64(l))
	}
	if m.ModelType != 0 {
		n += 1 + sovModelServerConfig(uint64(m.ModelType))
	}
	l = len(m.ModelPlatform)
	if l > 0 {
		n += 1 + l + sovModelServerConfig(uint64(l))
	}
	if m.LoggingConfig != nil {
		l = m.LoggingConfig.Size()
		n += 1 + l + sovModelServerConfig(uint64(l))
	}
	if m.ModelVersionPolicy != nil {
		l = m.ModelVersionPolicy.Size()
		n += 1 + l + sovModelServerConfig(uint64(l))
	}
	return n
}

func (m *ModelConfigList) Size() (n int) {
	var l int
	_ = l
	if len(m.Config) > 0 {
		for _, e := range m.Config {
			l = e.Size()
			n += 1 + l + sovModelServerConfig(uint64(l))
		}
	}
	return n
}

func (m *ModelServerConfig) Size() (n int) {
	var l int
	_ = l
	if m.Config != nil {
		n += m.Config.Size()
	}
	return n
}

func (m *ModelServerConfig_ModelConfigList) Size() (n int) {
	var l int
	_ = l
	if m.ModelConfigList != nil {
		l = m.ModelConfigList.Size()
		n += 1 + l + sovModelServerConfig(uint64(l))
	}
	return n
}
func (m *ModelServerConfig_CustomModelConfig) Size() (n int) {
	var l int
	_ = l
	if m.CustomModelConfig != nil {
		l = m.CustomModelConfig.Size()
		n += 1 + l + sovModelServerConfig(uint64(l))
	}
	return n
}

func sovModelServerConfig(x uint64) (n int) {
	for {
		n++
		x >>= 7
		if x == 0 {
			break
		}
	}
	return n
}
func sozModelServerConfig(x uint64) (n int) {
	return sovModelServerConfig(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (m *ModelConfig) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowModelServerConfig
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
			return fmt.Errorf("proto: ModelConfig: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: ModelConfig: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Name", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowModelServerConfig
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
				return ErrInvalidLengthModelServerConfig
			}
			postIndex := iNdEx + intStringLen
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.Name = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		case 2:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field BasePath", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowModelServerConfig
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
				return ErrInvalidLengthModelServerConfig
			}
			postIndex := iNdEx + intStringLen
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.BasePath = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		case 3:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field ModelType", wireType)
			}
			m.ModelType = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowModelServerConfig
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.ModelType |= (ModelType(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		case 4:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field ModelPlatform", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowModelServerConfig
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
				return ErrInvalidLengthModelServerConfig
			}
			postIndex := iNdEx + intStringLen
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.ModelPlatform = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		case 6:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field LoggingConfig", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowModelServerConfig
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
				return ErrInvalidLengthModelServerConfig
			}
			postIndex := iNdEx + msglen
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			if m.LoggingConfig == nil {
				m.LoggingConfig = &LoggingConfig{}
			}
			if err := m.LoggingConfig.Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		case 7:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field ModelVersionPolicy", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowModelServerConfig
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
				return ErrInvalidLengthModelServerConfig
			}
			postIndex := iNdEx + msglen
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			if m.ModelVersionPolicy == nil {
				m.ModelVersionPolicy = &FileSystemStoragePathSourceConfig_ServableVersionPolicy{}
			}
			if err := m.ModelVersionPolicy.Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipModelServerConfig(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if skippy < 0 {
				return ErrInvalidLengthModelServerConfig
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
func (m *ModelConfigList) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowModelServerConfig
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
			return fmt.Errorf("proto: ModelConfigList: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: ModelConfigList: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Config", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowModelServerConfig
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
				return ErrInvalidLengthModelServerConfig
			}
			postIndex := iNdEx + msglen
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.Config = append(m.Config, &ModelConfig{})
			if err := m.Config[len(m.Config)-1].Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipModelServerConfig(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if skippy < 0 {
				return ErrInvalidLengthModelServerConfig
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
func (m *ModelServerConfig) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowModelServerConfig
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
			return fmt.Errorf("proto: ModelServerConfig: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: ModelServerConfig: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field ModelConfigList", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowModelServerConfig
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
				return ErrInvalidLengthModelServerConfig
			}
			postIndex := iNdEx + msglen
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			v := &ModelConfigList{}
			if err := v.Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			m.Config = &ModelServerConfig_ModelConfigList{v}
			iNdEx = postIndex
		case 2:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field CustomModelConfig", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowModelServerConfig
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
				return ErrInvalidLengthModelServerConfig
			}
			postIndex := iNdEx + msglen
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			v := &google_protobuf.Any{}
			if err := v.Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			m.Config = &ModelServerConfig_CustomModelConfig{v}
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipModelServerConfig(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if skippy < 0 {
				return ErrInvalidLengthModelServerConfig
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
func skipModelServerConfig(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowModelServerConfig
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
					return 0, ErrIntOverflowModelServerConfig
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
					return 0, ErrIntOverflowModelServerConfig
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
				return 0, ErrInvalidLengthModelServerConfig
			}
			return iNdEx, nil
		case 3:
			for {
				var innerWire uint64
				var start int = iNdEx
				for shift := uint(0); ; shift += 7 {
					if shift >= 64 {
						return 0, ErrIntOverflowModelServerConfig
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
				next, err := skipModelServerConfig(dAtA[start:])
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
	ErrInvalidLengthModelServerConfig = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowModelServerConfig   = fmt.Errorf("proto: integer overflow")
)

func init() {
	proto.RegisterFile("tensorflow_serving/model_server_config.proto", fileDescriptorModelServerConfig)
}

var fileDescriptorModelServerConfig = []byte{
	// 534 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x84, 0x92, 0xd1, 0x8e, 0xd2, 0x40,
	0x14, 0x86, 0x19, 0x96, 0x45, 0x18, 0x02, 0x0b, 0xb3, 0x1b, 0x53, 0x31, 0x22, 0x62, 0x8c, 0xc4,
	0x98, 0x92, 0xe0, 0x85, 0x5e, 0x2a, 0xbb, 0x25, 0xec, 0xca, 0x42, 0x6d, 0x51, 0xe3, 0xd5, 0xa4,
	0xe0, 0xb4, 0xdb, 0x64, 0xda, 0x69, 0x3a, 0x03, 0xa6, 0x17, 0xbe, 0x83, 0x8f, 0xe3, 0x23, 0x78,
	0xe9, 0x23, 0x18, 0x7c, 0x07, 0xe3, 0xa5, 0xe9, 0x4c, 0x51, 0x50, 0xcc, 0xde, 0x71, 0x7e, 0xfe,
	0xf9, 0xce, 0xf9, 0x4f, 0x0f, 0x7c, 0x2c, 0x48, 0xc8, 0x59, 0xec, 0x52, 0xf6, 0x01, 0x73, 0x12,
	0xaf, 0xfc, 0xd0, 0xeb, 0x05, 0xec, 0x3d, 0xa1, 0xb2, 0x22, 0x31, 0x5e, 0xb0, 0xd0, 0xf5, 0x3d,
	0x3d, 0x8a, 0x99, 0x60, 0x08, 0xfd, 0x71, 0xeb, 0x99, 0xbb, 0x79, 0xcb, 0x63, 0xcc, 0xa3, 0xa4,
	0x27, 0x1d, 0xf3, 0xa5, 0xdb, 0x73, 0xc2, 0x44, 0xd9, 0x9b, 0x0f, 0xf7, 0xc0, 0x29, 0xf3, 0x3c,
	0x3f, 0xf4, 0x76, 0xb8, 0xcd, 0x67, 0x7b, 0x8c, 0xae, 0x4f, 0x09, 0xe6, 0x09, 0x17, 0x24, 0xc0,
	0x5c, 0xb0, 0xd8, 0xf1, 0x08, 0x8e, 0x1c, 0x71, 0x85, 0x39, 0x5b, 0xc6, 0x0b, 0xa2, 0x5e, 0x76,
	0x7e, 0xe4, 0x61, 0xe5, 0x32, 0x9d, 0xf7, 0x54, 0xf2, 0x10, 0x82, 0x85, 0xd0, 0x09, 0x88, 0x06,
	0xda, 0xa0, 0x5b, 0xb6, 0xe4, 0x6f, 0x74, 0x1b, 0x96, 0xe7, 0x0e, 0x57, 0xaf, 0xb5, 0xbc, 0xfc,
	0xa3, 0x94, 0x0a, 0xa6, 0x23, 0xae, 0xd0, 0x73, 0x08, 0x55, 0x5e, 0x91, 0x44, 0x44, 0x3b, 0x68,
	0x83, 0x6e, 0xad, 0x7f, 0x47, 0xff, 0x37, 0xa7, 0x2e, 0xbb, 0xcc, 0x92, 0x88, 0x0c, 0xf2, 0x1a,
	0xb0, 0xca, 0xc1, 0xa6, 0x44, 0x0f, 0x60, 0x4d, 0x11, 0x22, 0xea, 0x08, 0x97, 0xc5, 0x81, 0x56,
	0x90, 0x3d, 0xaa, 0x52, 0x35, 0x33, 0x11, 0x8d, 0x60, 0x6d, 0x37, 0xbb, 0x56, 0x6c, 0x83, 0x6e,
	0xa5, 0x7f, 0x6f, 0x5f, 0xb3, 0xb1, 0x72, 0xaa, 0x50, 0x56, 0x95, 0x6e, 0x97, 0xe8, 0x23, 0x3c,
	0x51, 0x0d, 0x57, 0x24, 0xe6, 0x3e, 0x0b, 0x71, 0xc4, 0xa8, 0xbf, 0x48, 0xb4, 0x1b, 0x92, 0xf7,
	0x72, 0x1f, 0x6f, 0xe8, 0x53, 0x62, 0xcb, 0x5d, 0xda, 0x6a, 0x95, 0x69, 0x76, 0x5b, 0x2e, 0x52,
	0x41, 0x75, 0x9b, 0xc4, 0x2b, 0x67, 0x4e, 0xc9, 0x1b, 0xc5, 0x34, 0x25, 0xd2, 0x42, 0xb2, 0xd1,
	0x8e, 0x76, 0x51, 0x28, 0x1d, 0xd6, 0x8b, 0x9d, 0x0b, 0x78, 0xb4, 0xb5, 0xf7, 0xb1, 0xcf, 0x05,
	0x7a, 0x0a, 0x8b, 0x59, 0x32, 0xd0, 0x3e, 0xe8, 0x56, 0xfa, 0x77, 0xff, 0xbb, 0xc6, 0x2c, 0x57,
	0x66, 0xef, 0x7c, 0x06, 0xb0, 0x21, 0x75, 0x5b, 0xde, 0x5c, 0x16, 0xf3, 0x15, 0x6c, 0xa8, 0x98,
	0xca, 0x85, 0xa9, 0xcf, 0x85, 0xfc, 0xae, 0x95, 0xfe, 0xfd, 0x6b, 0xc8, 0xe9, 0x38, 0xa3, 0x9c,
	0x75, 0x14, 0xfc, 0x35, 0xe1, 0x10, 0x1e, 0x2f, 0x96, 0x5c, 0xb0, 0x00, 0x6f, 0x93, 0xe5, 0x4d,
	0x54, 0xfa, 0x27, 0xba, 0xba, 0x64, 0x7d, 0x73, 0xc9, 0xfa, 0x8b, 0x30, 0x19, 0xe5, 0xac, 0x86,
	0x7a, 0xb2, 0x85, 0x1f, 0x94, 0x36, 0x49, 0x1f, 0x4d, 0x60, 0xf9, 0xf7, 0x61, 0xa0, 0x16, 0xbc,
	0x79, 0x39, 0x3d, 0x33, 0xc6, 0x78, 0xf6, 0xce, 0x34, 0xf0, 0xeb, 0x89, 0x6d, 0x1a, 0xa7, 0xe7,
	0xc3, 0x73, 0xe3, 0xac, 0x9e, 0x6b, 0xe6, 0x4b, 0x00, 0x21, 0x08, 0x67, 0xc6, 0xc4, 0x9e, 0x5a,
	0xc3, 0xf1, 0xf4, 0x6d, 0x1d, 0x48, 0xad, 0x0a, 0x0f, 0xa7, 0xb3, 0x91, 0x61, 0xd5, 0xf3, 0x69,
	0x39, 0x38, 0xfe, 0xb2, 0x6e, 0x81, 0xaf, 0xeb, 0x16, 0xf8, 0xb6, 0x6e, 0x81, 0x4f, 0xdf, 0x5b,
	0xb9, 0x9f, 0x00, 0xcc, 0x8b, 0x72, 0xa2, 0x27, 0xbf, 0x02, 0x00, 0x00, 0xff, 0xff, 0xc5, 0xaa,
	0xb6, 0x6d, 0xad, 0x03, 0x00, 0x00,
}
