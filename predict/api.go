package predict

import (
	"context"

	"github.com/Applifier/go-tensorflow/types/tensorflow/core/example"
	"github.com/Applifier/go-tensorflow/utils"
)

// An Example is a mostly-normalized data format for storing data for
// training and inference.  It contains a key-value store (features); where
// each key (string) maps to a Feature message (which is oneof packed BytesList,
// FloatList, or Int64List).
type Example = example.Example

// Feature contains Lists which may hold zero or more values.
type Feature = example.Feature

// Examplifier interface for types that can be converted to examples
type Examplifier interface {
	Examples() ([]*Example, error)
}

// MapExample map type that implements Examplifier interface
type MapExample map[string]interface{}

// Examples returns examples (one example) from a given map
func (me MapExample) Examples() ([]*Example, error) {
	example, err := utils.NewExampleFromMap(me)
	if err != nil {
		return nil, err
	}

	return []*Example{example}, nil
}

// Class struct returned by classify calls to a model
type Class struct {
	Label string
	Score float32
}

// Regression struct returned by regress calls to a model
type Regression struct {
	Value float32
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

// Predictor interface for unified model execution with different backend (embedded go model & tensorflow serving)
type Predictor interface {
	// Predict runs prediction with given input map. Output is filtered with given filter. (nil defaults to all outputs)
	Predict(ctx context.Context, inputs map[string]interface{}, outputFilter []string) (map[string]Tensor, ModelInfo, error)
	// Classify runs classify with given features and context
	Classify(ctx context.Context, examples []*Example, context *Example) ([][]Class, ModelInfo, error)
	// Regress runs regression with given features and context
	Regress(ctx context.Context, examples []*Example, context *Example) ([]Regression, ModelInfo, error)
	// GetModelInfo returns the ModelInfo for the Predictor
	GetModelInfo(ctx context.Context) (ModelInfo, error)
}
