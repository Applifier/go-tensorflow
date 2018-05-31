package predict

import "context"

// Predictor interface for unified model execution with different backend (embedded go model & tensorflow serving)
type Predictor interface {
	// Predict runs prediction with given input map. Output is filtered with given filter. (nil defaults to all outputs)
	Predict(ctx context.Context, inputs map[string]interface{}, outputFilter []string) (map[string]Tensor, ModelInfo, error)
}

// ModelInfo struct contains infomation about the model used for the prediction (name, version, etc.)
type ModelInfo struct {
	Name    string
	Version int
}

// TensorType type of the tensor
type TensorType int

const (
	TensorTypeFloat = TensorType(iota)
	TensorTypeDouble
	TensorTypeInt32
	TensorTypeUInt32
	TensorTypeInt16
	TensorTypeInt8
	TensorTypeUInt8
	TensorTypeString
	TensorTypeComplex64
	TensorTypeComplex128
	TensorTypeInt64
	TensorTypeUInt64
	TensorTypeBool
)

// Tensor unified interface for Tensors
type Tensor interface {
	Value() interface{}
	Shape() []int64
	Type() TensorType
}
