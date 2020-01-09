// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: tensorflow_serving/model_server_config.proto

package tensorflow_serving

import (
	fmt "fmt"
	proto "github.com/gogo/protobuf/proto"
	types "github.com/gogo/protobuf/types"
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

// The type of model.
// TODO(b/31336131): DEPRECATED.
type ModelType int32

const (
	ModelType_MODEL_TYPE_UNSPECIFIED ModelType = 0 // Deprecated: Do not use.
	ModelType_TENSORFLOW             ModelType = 1 // Deprecated: Do not use.
	ModelType_OTHER                  ModelType = 2 // Deprecated: Do not use.
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

func (ModelType) EnumDescriptor() ([]byte, []int) {
	return fileDescriptor_8317b20b2b826bad, []int{0}
}

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
	ModelType ModelType `protobuf:"varint,3,opt,name=model_type,json=modelType,proto3,enum=tensorflow.serving.ModelType" json:"model_type,omitempty"` // Deprecated: Do not use.
	// Type of model (e.g. "tensorflow").
	//
	// (This cannot be changed once a model is in serving.)
	ModelPlatform string `protobuf:"bytes,4,opt,name=model_platform,json=modelPlatform,proto3" json:"model_platform,omitempty"`
	// Version policy for the model indicating which version(s) of the model to
	// load and make available for serving simultaneously.
	// The default option is to serve only the latest version of the model.
	//
	// (This can be changed once a model is in serving.)
	ModelVersionPolicy *FileSystemStoragePathSourceConfig_ServableVersionPolicy `protobuf:"bytes,7,opt,name=model_version_policy,json=modelVersionPolicy,proto3" json:"model_version_policy,omitempty"`
	// String labels to associate with versions of the model, allowing inference
	// queries to refer to versions by label instead of number. Multiple labels
	// can map to the same version, but not vice-versa.
	//
	// An envisioned use-case for these labels is canarying tentative versions.
	// For example, one can assign labels "stable" and "canary" to two specific
	// versions. Perhaps initially "stable" is assigned to version 0 and "canary"
	// to version 1. Once version 1 passes canary, one can shift the "stable"
	// label to refer to version 1 (at that point both labels map to the same
	// version -- version 1 -- which is fine). Later once version 2 is ready to
	// canary one can move the "canary" label to version 2. And so on.
	VersionLabels map[string]int64 `protobuf:"bytes,8,rep,name=version_labels,json=versionLabels,proto3" json:"version_labels,omitempty" protobuf_key:"bytes,1,opt,name=key,proto3" protobuf_val:"varint,2,opt,name=value,proto3"`
	// Configures logging requests and responses, to the model.
	//
	// (This can be changed once a model is in serving.)
	LoggingConfig *LoggingConfig `protobuf:"bytes,6,opt,name=logging_config,json=loggingConfig,proto3" json:"logging_config,omitempty"`
}

func (m *ModelConfig) Reset()         { *m = ModelConfig{} }
func (m *ModelConfig) String() string { return proto.CompactTextString(m) }
func (*ModelConfig) ProtoMessage()    {}
func (*ModelConfig) Descriptor() ([]byte, []int) {
	return fileDescriptor_8317b20b2b826bad, []int{0}
}
func (m *ModelConfig) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *ModelConfig) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_ModelConfig.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalTo(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *ModelConfig) XXX_Merge(src proto.Message) {
	xxx_messageInfo_ModelConfig.Merge(m, src)
}
func (m *ModelConfig) XXX_Size() int {
	return m.Size()
}
func (m *ModelConfig) XXX_DiscardUnknown() {
	xxx_messageInfo_ModelConfig.DiscardUnknown(m)
}

var xxx_messageInfo_ModelConfig proto.InternalMessageInfo

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

// Deprecated: Do not use.
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

func (m *ModelConfig) GetVersionLabels() map[string]int64 {
	if m != nil {
		return m.VersionLabels
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
	Config []*ModelConfig `protobuf:"bytes,1,rep,name=config,proto3" json:"config,omitempty"`
}

func (m *ModelConfigList) Reset()         { *m = ModelConfigList{} }
func (m *ModelConfigList) String() string { return proto.CompactTextString(m) }
func (*ModelConfigList) ProtoMessage()    {}
func (*ModelConfigList) Descriptor() ([]byte, []int) {
	return fileDescriptor_8317b20b2b826bad, []int{1}
}
func (m *ModelConfigList) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *ModelConfigList) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_ModelConfigList.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalTo(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *ModelConfigList) XXX_Merge(src proto.Message) {
	xxx_messageInfo_ModelConfigList.Merge(m, src)
}
func (m *ModelConfigList) XXX_Size() int {
	return m.Size()
}
func (m *ModelConfigList) XXX_DiscardUnknown() {
	xxx_messageInfo_ModelConfigList.DiscardUnknown(m)
}

var xxx_messageInfo_ModelConfigList proto.InternalMessageInfo

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
	return fileDescriptor_8317b20b2b826bad, []int{2}
}
func (m *ModelServerConfig) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *ModelServerConfig) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_ModelServerConfig.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalTo(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *ModelServerConfig) XXX_Merge(src proto.Message) {
	xxx_messageInfo_ModelServerConfig.Merge(m, src)
}
func (m *ModelServerConfig) XXX_Size() int {
	return m.Size()
}
func (m *ModelServerConfig) XXX_DiscardUnknown() {
	xxx_messageInfo_ModelServerConfig.DiscardUnknown(m)
}

var xxx_messageInfo_ModelServerConfig proto.InternalMessageInfo

type isModelServerConfig_Config interface {
	isModelServerConfig_Config()
	MarshalTo([]byte) (int, error)
	Size() int
}

type ModelServerConfig_ModelConfigList struct {
	ModelConfigList *ModelConfigList `protobuf:"bytes,1,opt,name=model_config_list,json=modelConfigList,proto3,oneof"`
}
type ModelServerConfig_CustomModelConfig struct {
	CustomModelConfig *types.Any `protobuf:"bytes,2,opt,name=custom_model_config,json=customModelConfig,proto3,oneof"`
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

func (m *ModelServerConfig) GetCustomModelConfig() *types.Any {
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
		msg := new(types.Any)
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
		n += 1 // tag and wire
		n += proto.SizeVarint(uint64(s))
		n += s
	case *ModelServerConfig_CustomModelConfig:
		s := proto.Size(x.CustomModelConfig)
		n += 1 // tag and wire
		n += proto.SizeVarint(uint64(s))
		n += s
	case nil:
	default:
		panic(fmt.Sprintf("proto: unexpected type %T in oneof", x))
	}
	return n
}

func init() {
	proto.RegisterEnum("tensorflow.serving.ModelType", ModelType_name, ModelType_value)
	proto.RegisterType((*ModelConfig)(nil), "tensorflow.serving.ModelConfig")
	proto.RegisterMapType((map[string]int64)(nil), "tensorflow.serving.ModelConfig.VersionLabelsEntry")
	proto.RegisterType((*ModelConfigList)(nil), "tensorflow.serving.ModelConfigList")
	proto.RegisterType((*ModelServerConfig)(nil), "tensorflow.serving.ModelServerConfig")
}

func init() {
	proto.RegisterFile("tensorflow_serving/model_server_config.proto", fileDescriptor_8317b20b2b826bad)
}

var fileDescriptor_8317b20b2b826bad = []byte{
	// 605 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x84, 0x53, 0x4f, 0x6f, 0xd3, 0x3c,
	0x18, 0x8f, 0xdb, 0xae, 0x6f, 0xeb, 0xaa, 0x5d, 0xe7, 0x77, 0x42, 0xa1, 0x13, 0xa1, 0x0c, 0x21,
	0x2a, 0x84, 0x32, 0x29, 0x1c, 0x98, 0x38, 0x8d, 0x6d, 0xa9, 0xba, 0xd1, 0xb5, 0x25, 0x29, 0xa0,
	0x9d, 0xa2, 0xb4, 0xb8, 0x59, 0x84, 0x13, 0x47, 0xb1, 0x5b, 0x94, 0x03, 0xdf, 0x01, 0x89, 0x2f,
	0xc3, 0x47, 0xe0, 0xb8, 0x23, 0x47, 0xd4, 0x7e, 0x09, 0x8e, 0x28, 0x76, 0x0a, 0x2d, 0x2b, 0xda,
	0x2d, 0xcf, 0x93, 0xdf, 0xf3, 0xfb, 0xe3, 0xc7, 0x86, 0x4f, 0x39, 0x0e, 0x19, 0x8d, 0x27, 0x84,
	0x7e, 0x74, 0x18, 0x8e, 0x67, 0x7e, 0xe8, 0x1d, 0x04, 0xf4, 0x3d, 0x26, 0xa2, 0xc2, 0xb1, 0x33,
	0xa6, 0xe1, 0xc4, 0xf7, 0xf4, 0x28, 0xa6, 0x9c, 0x22, 0xf4, 0x07, 0xad, 0x67, 0xe8, 0xc6, 0x5d,
	0x8f, 0x52, 0x8f, 0xe0, 0x03, 0x81, 0x18, 0x4d, 0x27, 0x07, 0x6e, 0x98, 0x48, 0x78, 0xe3, 0xf1,
	0x06, 0x72, 0x42, 0x3d, 0xcf, 0x0f, 0xbd, 0x35, 0xde, 0xc6, 0xe1, 0x06, 0xe0, 0xc4, 0x27, 0xd8,
	0x61, 0x09, 0xe3, 0x38, 0x70, 0x18, 0xa7, 0xb1, 0xeb, 0x61, 0x27, 0x72, 0xf9, 0x95, 0xc3, 0xe8,
	0x34, 0x1e, 0x63, 0x39, 0xb9, 0xff, 0xa5, 0x00, 0x2b, 0x17, 0xa9, 0xdf, 0x13, 0xc1, 0x87, 0x10,
	0x2c, 0x84, 0x6e, 0x80, 0x55, 0xd0, 0x04, 0xad, 0xb2, 0x25, 0xbe, 0xd1, 0x1e, 0x2c, 0x8f, 0x5c,
	0x26, 0xa7, 0xd5, 0x9c, 0xf8, 0x51, 0x4a, 0x1b, 0x03, 0x97, 0x5f, 0xa1, 0x23, 0x08, 0x65, 0x5e,
	0x9e, 0x44, 0x58, 0xcd, 0x37, 0x41, 0xab, 0x66, 0xdc, 0xd3, 0x6f, 0xe6, 0xd4, 0x85, 0xca, 0x30,
	0x89, 0xf0, 0x71, 0x4e, 0x05, 0x56, 0x39, 0x58, 0x96, 0xe8, 0x11, 0xac, 0x49, 0x86, 0x88, 0xb8,
	0x7c, 0x42, 0xe3, 0x40, 0x2d, 0x08, 0x8d, 0xaa, 0xe8, 0x0e, 0xb2, 0x26, 0xfa, 0x04, 0x77, 0x25,
	0x6c, 0x86, 0x63, 0xe6, 0xd3, 0xd0, 0x89, 0x28, 0xf1, 0xc7, 0x89, 0xfa, 0x5f, 0x13, 0xb4, 0x2a,
	0xc6, 0xab, 0x4d, 0x92, 0x6d, 0x9f, 0x60, 0x5b, 0x9c, 0x80, 0x2d, 0x0f, 0x20, 0x75, 0x6c, 0x8b,
	0xf8, 0x32, 0xae, 0x6e, 0xe3, 0x78, 0xe6, 0x8e, 0x08, 0x7e, 0x2b, 0x39, 0x07, 0x82, 0xd2, 0x42,
	0x42, 0x68, 0xad, 0x87, 0x2e, 0x61, 0x6d, 0x29, 0x4c, 0xdc, 0x11, 0x26, 0x4c, 0x2d, 0x35, 0xf3,
	0xad, 0x8a, 0x61, 0xfc, 0x33, 0x6b, 0x26, 0x91, 0xd1, 0x74, 0xc5, 0x90, 0x19, 0xf2, 0x38, 0xb1,
	0xaa, 0xb3, 0xd5, 0x1e, 0xea, 0xc0, 0xda, 0xfa, 0x56, 0xd5, 0xa2, 0xc8, 0xf4, 0x60, 0x13, 0x75,
	0x57, 0x22, 0x25, 0xb9, 0x55, 0x25, 0xab, 0x65, 0xe3, 0x08, 0xa2, 0x9b, 0x72, 0xa8, 0x0e, 0xf3,
	0x1f, 0x70, 0x92, 0xad, 0x34, 0xfd, 0x44, 0xbb, 0x70, 0x6b, 0xe6, 0x92, 0x29, 0x16, 0xdb, 0xcc,
	0x5b, 0xb2, 0x78, 0x91, 0x3b, 0x04, 0xe7, 0x85, 0xd2, 0x56, 0xbd, 0xb8, 0x7f, 0x0e, 0xb7, 0x57,
	0x22, 0x74, 0x7d, 0xc6, 0xd1, 0x73, 0x58, 0xcc, 0xcc, 0x01, 0x91, 0xfb, 0xfe, 0x2d, 0xb9, 0xad,
	0x0c, 0xbe, 0xff, 0x15, 0xc0, 0x1d, 0xd1, 0xb7, 0xc5, 0x83, 0xc8, 0xee, 0xd9, 0x6b, 0xb8, 0x23,
	0xb7, 0x29, 0x51, 0x0e, 0xf1, 0x19, 0x17, 0x0e, 0x2b, 0xc6, 0xc3, 0x5b, 0x98, 0x53, 0x3b, 0x1d,
	0xc5, 0xda, 0x0e, 0xfe, 0x72, 0xd8, 0x86, 0xff, 0x8f, 0xa7, 0x8c, 0xd3, 0xc0, 0x59, 0x65, 0x16,
	0x11, 0x2b, 0xc6, 0xae, 0x2e, 0x9f, 0x99, 0xbe, 0x7c, 0x66, 0xfa, 0xcb, 0x30, 0xe9, 0x28, 0xd6,
	0x8e, 0x1c, 0x59, 0xa1, 0x3f, 0x2e, 0x2d, 0x93, 0x3e, 0xe9, 0xc1, 0xf2, 0xef, 0x5b, 0x8b, 0x34,
	0x78, 0xe7, 0xa2, 0x7f, 0x6a, 0x76, 0x9d, 0xe1, 0xe5, 0xc0, 0x74, 0xde, 0xf4, 0xec, 0x81, 0x79,
	0x72, 0xd6, 0x3e, 0x33, 0x4f, 0xeb, 0x4a, 0x23, 0x57, 0x02, 0x08, 0x41, 0x38, 0x34, 0x7b, 0x76,
	0xdf, 0x6a, 0x77, 0xfb, 0xef, 0xea, 0x40, 0xf4, 0xaa, 0x70, 0xab, 0x3f, 0xec, 0x98, 0x56, 0x3d,
	0x97, 0x96, 0xc7, 0x7b, 0xdf, 0xe6, 0x1a, 0xb8, 0x9e, 0x6b, 0xe0, 0xc7, 0x5c, 0x03, 0x9f, 0x17,
	0x9a, 0x72, 0xbd, 0xd0, 0x94, 0xef, 0x0b, 0x4d, 0xf9, 0x09, 0xc0, 0xa8, 0x28, 0x9c, 0x3d, 0xfb,
	0x15, 0x00, 0x00, 0xff, 0xff, 0x59, 0xa1, 0xea, 0xee, 0x52, 0x04, 0x00, 0x00,
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
	if len(m.VersionLabels) > 0 {
		for k, _ := range m.VersionLabels {
			dAtA[i] = 0x42
			i++
			v := m.VersionLabels[k]
			mapSize := 1 + len(k) + sovModelServerConfig(uint64(len(k))) + 1 + sovModelServerConfig(uint64(v))
			i = encodeVarintModelServerConfig(dAtA, i, uint64(mapSize))
			dAtA[i] = 0xa
			i++
			i = encodeVarintModelServerConfig(dAtA, i, uint64(len(k)))
			i += copy(dAtA[i:], k)
			dAtA[i] = 0x10
			i++
			i = encodeVarintModelServerConfig(dAtA, i, uint64(v))
		}
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
	if m == nil {
		return 0
	}
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
	if len(m.VersionLabels) > 0 {
		for k, v := range m.VersionLabels {
			_ = k
			_ = v
			mapEntrySize := 1 + len(k) + sovModelServerConfig(uint64(len(k))) + 1 + sovModelServerConfig(uint64(v))
			n += mapEntrySize + 1 + sovModelServerConfig(uint64(mapEntrySize))
		}
	}
	return n
}

func (m *ModelConfigList) Size() (n int) {
	if m == nil {
		return 0
	}
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
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if m.Config != nil {
		n += m.Config.Size()
	}
	return n
}

func (m *ModelServerConfig_ModelConfigList) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if m.ModelConfigList != nil {
		l = m.ModelConfigList.Size()
		n += 1 + l + sovModelServerConfig(uint64(l))
	}
	return n
}
func (m *ModelServerConfig_CustomModelConfig) Size() (n int) {
	if m == nil {
		return 0
	}
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
		case 8:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field VersionLabels", wireType)
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
			if m.VersionLabels == nil {
				m.VersionLabels = make(map[string]int64)
			}
			var mapkey string
			var mapvalue int64
			for iNdEx < postIndex {
				entryPreIndex := iNdEx
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
				if fieldNum == 1 {
					var stringLenmapkey uint64
					for shift := uint(0); ; shift += 7 {
						if shift >= 64 {
							return ErrIntOverflowModelServerConfig
						}
						if iNdEx >= l {
							return io.ErrUnexpectedEOF
						}
						b := dAtA[iNdEx]
						iNdEx++
						stringLenmapkey |= (uint64(b) & 0x7F) << shift
						if b < 0x80 {
							break
						}
					}
					intStringLenmapkey := int(stringLenmapkey)
					if intStringLenmapkey < 0 {
						return ErrInvalidLengthModelServerConfig
					}
					postStringIndexmapkey := iNdEx + intStringLenmapkey
					if postStringIndexmapkey > l {
						return io.ErrUnexpectedEOF
					}
					mapkey = string(dAtA[iNdEx:postStringIndexmapkey])
					iNdEx = postStringIndexmapkey
				} else if fieldNum == 2 {
					for shift := uint(0); ; shift += 7 {
						if shift >= 64 {
							return ErrIntOverflowModelServerConfig
						}
						if iNdEx >= l {
							return io.ErrUnexpectedEOF
						}
						b := dAtA[iNdEx]
						iNdEx++
						mapvalue |= (int64(b) & 0x7F) << shift
						if b < 0x80 {
							break
						}
					}
				} else {
					iNdEx = entryPreIndex
					skippy, err := skipModelServerConfig(dAtA[iNdEx:])
					if err != nil {
						return err
					}
					if skippy < 0 {
						return ErrInvalidLengthModelServerConfig
					}
					if (iNdEx + skippy) > postIndex {
						return io.ErrUnexpectedEOF
					}
					iNdEx += skippy
				}
			}
			m.VersionLabels[mapkey] = mapvalue
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
			v := &types.Any{}
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
