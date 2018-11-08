// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: tensorflow_serving/class_registration_test.proto

/*
	Package tensorflow_serving is a generated protocol buffer package.

	It is generated from these files:
		tensorflow_serving/class_registration_test.proto
		tensorflow_serving/classification.proto
		tensorflow_serving/file_system_storage_path_source.proto
		tensorflow_serving/get_model_metadata.proto
		tensorflow_serving/get_model_status.proto
		tensorflow_serving/inference.proto
		tensorflow_serving/input.proto
		tensorflow_serving/log_collector_config.proto
		tensorflow_serving/logging.proto
		tensorflow_serving/logging_config.proto
		tensorflow_serving/model.proto
		tensorflow_serving/model_management.proto
		tensorflow_serving/model_server_config.proto
		tensorflow_serving/model_service.proto
		tensorflow_serving/monitoring_config.proto
		tensorflow_serving/platform_config.proto
		tensorflow_serving/predict.proto
		tensorflow_serving/prediction_log.proto
		tensorflow_serving/prediction_service.proto
		tensorflow_serving/regression.proto
		tensorflow_serving/session_service.proto
		tensorflow_serving/ssl_config.proto
		tensorflow_serving/static_storage_path_source.proto
		tensorflow_serving/status.proto

	It has these top-level messages:
		Config1
		Config2
		MessageWithAny
		Class
		Classifications
		ClassificationResult
		ClassificationRequest
		ClassificationResponse
		FileSystemStoragePathSourceConfig
		SignatureDefMap
		GetModelMetadataRequest
		GetModelMetadataResponse
		GetModelStatusRequest
		ModelVersionStatus
		GetModelStatusResponse
		InferenceTask
		InferenceResult
		MultiInferenceRequest
		MultiInferenceResponse
		ExampleList
		ExampleListWithContext
		Input
		LogCollectorConfig
		LogMetadata
		SamplingConfig
		LoggingConfig
		ModelSpec
		ReloadConfigRequest
		ReloadConfigResponse
		ModelConfig
		ModelConfigList
		ModelServerConfig
		PrometheusConfig
		MonitoringConfig
		PlatformConfig
		PlatformConfigMap
		PredictRequest
		PredictResponse
		ClassifyLog
		RegressLog
		PredictLog
		MultiInferenceLog
		SessionRunLog
		PredictionLog
		Regression
		RegressionResult
		RegressionRequest
		RegressionResponse
		SessionRunRequest
		SessionRunResponse
		SSLConfig
		StaticStoragePathSourceConfig
		StatusProto
*/
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

// This is a compile-time assertion to ensure that this generated file
// is compatible with the proto package it is being compiled against.
// A compilation error at this line likely means your copy of the
// proto package needs to be updated.
const _ = proto.GoGoProtoPackageIsVersion2 // please upgrade the proto package

type Config1 struct {
	StringField string `protobuf:"bytes,1,opt,name=string_field,json=stringField,proto3" json:"string_field,omitempty"`
}

func (m *Config1) Reset()                    { *m = Config1{} }
func (m *Config1) String() string            { return proto.CompactTextString(m) }
func (*Config1) ProtoMessage()               {}
func (*Config1) Descriptor() ([]byte, []int) { return fileDescriptorClassRegistrationTest, []int{0} }

func (m *Config1) GetStringField() string {
	if m != nil {
		return m.StringField
	}
	return ""
}

type Config2 struct {
	StringField string `protobuf:"bytes,1,opt,name=string_field,json=stringField,proto3" json:"string_field,omitempty"`
}

func (m *Config2) Reset()                    { *m = Config2{} }
func (m *Config2) String() string            { return proto.CompactTextString(m) }
func (*Config2) ProtoMessage()               {}
func (*Config2) Descriptor() ([]byte, []int) { return fileDescriptorClassRegistrationTest, []int{1} }

func (m *Config2) GetStringField() string {
	if m != nil {
		return m.StringField
	}
	return ""
}

type MessageWithAny struct {
	AnyField *google_protobuf.Any `protobuf:"bytes,1,opt,name=any_field,json=anyField" json:"any_field,omitempty"`
}

func (m *MessageWithAny) Reset()         { *m = MessageWithAny{} }
func (m *MessageWithAny) String() string { return proto.CompactTextString(m) }
func (*MessageWithAny) ProtoMessage()    {}
func (*MessageWithAny) Descriptor() ([]byte, []int) {
	return fileDescriptorClassRegistrationTest, []int{2}
}

func (m *MessageWithAny) GetAnyField() *google_protobuf.Any {
	if m != nil {
		return m.AnyField
	}
	return nil
}

func init() {
	proto.RegisterType((*Config1)(nil), "tensorflow.serving.Config1")
	proto.RegisterType((*Config2)(nil), "tensorflow.serving.Config2")
	proto.RegisterType((*MessageWithAny)(nil), "tensorflow.serving.MessageWithAny")
}
func (m *Config1) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalTo(dAtA)
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *Config1) MarshalTo(dAtA []byte) (int, error) {
	var i int
	_ = i
	var l int
	_ = l
	if len(m.StringField) > 0 {
		dAtA[i] = 0xa
		i++
		i = encodeVarintClassRegistrationTest(dAtA, i, uint64(len(m.StringField)))
		i += copy(dAtA[i:], m.StringField)
	}
	return i, nil
}

func (m *Config2) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalTo(dAtA)
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *Config2) MarshalTo(dAtA []byte) (int, error) {
	var i int
	_ = i
	var l int
	_ = l
	if len(m.StringField) > 0 {
		dAtA[i] = 0xa
		i++
		i = encodeVarintClassRegistrationTest(dAtA, i, uint64(len(m.StringField)))
		i += copy(dAtA[i:], m.StringField)
	}
	return i, nil
}

func (m *MessageWithAny) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalTo(dAtA)
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *MessageWithAny) MarshalTo(dAtA []byte) (int, error) {
	var i int
	_ = i
	var l int
	_ = l
	if m.AnyField != nil {
		dAtA[i] = 0xa
		i++
		i = encodeVarintClassRegistrationTest(dAtA, i, uint64(m.AnyField.Size()))
		n1, err := m.AnyField.MarshalTo(dAtA[i:])
		if err != nil {
			return 0, err
		}
		i += n1
	}
	return i, nil
}

func encodeVarintClassRegistrationTest(dAtA []byte, offset int, v uint64) int {
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return offset + 1
}
func (m *Config1) Size() (n int) {
	var l int
	_ = l
	l = len(m.StringField)
	if l > 0 {
		n += 1 + l + sovClassRegistrationTest(uint64(l))
	}
	return n
}

func (m *Config2) Size() (n int) {
	var l int
	_ = l
	l = len(m.StringField)
	if l > 0 {
		n += 1 + l + sovClassRegistrationTest(uint64(l))
	}
	return n
}

func (m *MessageWithAny) Size() (n int) {
	var l int
	_ = l
	if m.AnyField != nil {
		l = m.AnyField.Size()
		n += 1 + l + sovClassRegistrationTest(uint64(l))
	}
	return n
}

func sovClassRegistrationTest(x uint64) (n int) {
	for {
		n++
		x >>= 7
		if x == 0 {
			break
		}
	}
	return n
}
func sozClassRegistrationTest(x uint64) (n int) {
	return sovClassRegistrationTest(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (m *Config1) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowClassRegistrationTest
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
			return fmt.Errorf("proto: Config1: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: Config1: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field StringField", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowClassRegistrationTest
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
				return ErrInvalidLengthClassRegistrationTest
			}
			postIndex := iNdEx + intStringLen
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.StringField = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipClassRegistrationTest(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if skippy < 0 {
				return ErrInvalidLengthClassRegistrationTest
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
func (m *Config2) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowClassRegistrationTest
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
			return fmt.Errorf("proto: Config2: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: Config2: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field StringField", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowClassRegistrationTest
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
				return ErrInvalidLengthClassRegistrationTest
			}
			postIndex := iNdEx + intStringLen
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.StringField = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipClassRegistrationTest(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if skippy < 0 {
				return ErrInvalidLengthClassRegistrationTest
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
func (m *MessageWithAny) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowClassRegistrationTest
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
			return fmt.Errorf("proto: MessageWithAny: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: MessageWithAny: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field AnyField", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowClassRegistrationTest
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
				return ErrInvalidLengthClassRegistrationTest
			}
			postIndex := iNdEx + msglen
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			if m.AnyField == nil {
				m.AnyField = &google_protobuf.Any{}
			}
			if err := m.AnyField.Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipClassRegistrationTest(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if skippy < 0 {
				return ErrInvalidLengthClassRegistrationTest
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
func skipClassRegistrationTest(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowClassRegistrationTest
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
					return 0, ErrIntOverflowClassRegistrationTest
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
					return 0, ErrIntOverflowClassRegistrationTest
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
				return 0, ErrInvalidLengthClassRegistrationTest
			}
			return iNdEx, nil
		case 3:
			for {
				var innerWire uint64
				var start int = iNdEx
				for shift := uint(0); ; shift += 7 {
					if shift >= 64 {
						return 0, ErrIntOverflowClassRegistrationTest
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
				next, err := skipClassRegistrationTest(dAtA[start:])
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
	ErrInvalidLengthClassRegistrationTest = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowClassRegistrationTest   = fmt.Errorf("proto: integer overflow")
)

func init() {
	proto.RegisterFile("tensorflow_serving/class_registration_test.proto", fileDescriptorClassRegistrationTest)
}

var fileDescriptorClassRegistrationTest = []byte{
	// 217 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0xe2, 0x32, 0x28, 0x49, 0xcd, 0x2b,
	0xce, 0x2f, 0x4a, 0xcb, 0xc9, 0x2f, 0x8f, 0x2f, 0x4e, 0x2d, 0x2a, 0xcb, 0xcc, 0x4b, 0xd7, 0x4f,
	0xce, 0x49, 0x2c, 0x2e, 0x8e, 0x2f, 0x4a, 0x4d, 0xcf, 0x2c, 0x2e, 0x29, 0x4a, 0x2c, 0xc9, 0xcc,
	0xcf, 0x8b, 0x2f, 0x49, 0x2d, 0x2e, 0xd1, 0x2b, 0x28, 0xca, 0x2f, 0xc9, 0x17, 0x12, 0x42, 0xe8,
	0xd0, 0x83, 0xea, 0x90, 0x92, 0x4c, 0xcf, 0xcf, 0x4f, 0xcf, 0x49, 0xd5, 0x07, 0xab, 0x48, 0x2a,
	0x4d, 0xd3, 0x4f, 0xcc, 0xab, 0x84, 0x28, 0x57, 0xd2, 0xe1, 0x62, 0x77, 0xce, 0xcf, 0x4b, 0xcb,
	0x4c, 0x37, 0x14, 0x52, 0xe4, 0xe2, 0x29, 0x2e, 0x29, 0xca, 0xcc, 0x4b, 0x8f, 0x4f, 0xcb, 0x4c,
	0xcd, 0x49, 0x91, 0x60, 0x54, 0x60, 0xd4, 0xe0, 0x0c, 0xe2, 0x86, 0x88, 0xb9, 0x81, 0x84, 0x10,
	0xaa, 0x8d, 0x88, 0x51, 0xed, 0xcc, 0xc5, 0xe7, 0x9b, 0x5a, 0x5c, 0x9c, 0x98, 0x9e, 0x1a, 0x9e,
	0x59, 0x92, 0xe1, 0x98, 0x57, 0x29, 0x64, 0xc8, 0xc5, 0x99, 0x98, 0x57, 0x89, 0xa4, 0x83, 0xdb,
	0x48, 0x44, 0x0f, 0xe2, 0x38, 0x3d, 0x98, 0xe3, 0xf4, 0x1c, 0xf3, 0x2a, 0x83, 0x38, 0x12, 0xf3,
	0x2a, 0xc1, 0x86, 0x38, 0x09, 0x9c, 0x78, 0x24, 0xc7, 0x78, 0xe1, 0x91, 0x1c, 0xe3, 0x83, 0x47,
	0x72, 0x8c, 0x13, 0x1e, 0xcb, 0x31, 0x24, 0xb1, 0x81, 0x55, 0x1a, 0x03, 0x02, 0x00, 0x00, 0xff,
	0xff, 0x90, 0x5a, 0x8b, 0x67, 0x1c, 0x01, 0x00, 0x00,
}
