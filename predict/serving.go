package predict

import (
	"context"
	"fmt"

	"github.com/Applifier/go-tensorflow/serving"
	"github.com/Applifier/go-tensorflow/types/tensorflow/core/framework"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

// servingPredictor implementation of the Predictor interface for TensorFlow Serving
type servingPredictor struct {
	modelClient serving.ModelPredictionClient
}

// NewServingPredictor returns a new predictor using a given TF Serving client
func NewServingPredictor(modelClient serving.ModelPredictionClient) Predictor {
	return &servingPredictor{
		modelClient: modelClient,
	}
}

func (sp *servingPredictor) convertValueToTensor(val interface{}) (*serving.Tensor, error) {
	switch v := val.(type) {
	case *tf.Tensor:
		return serving.NewTensor(v.Value())
	case map[string]interface{}:
		example, err := serving.NewExampleFromMap(v)
		if err != nil {
			return nil, err
		}
		exampleSerialized, err := example.Marshal()
		if err != nil {
			return nil, err
		}
		return serving.NewTensor([][]byte{exampleSerialized})

	case []map[string]interface{}:
		examples := make([][]byte, len(v))

		for i, m := range v {
			example, err := serving.NewExampleFromMap(m)
			if err != nil {
				return nil, err
			}
			exampleSerialized, err := example.Marshal()
			if err != nil {
				return nil, err
			}
			examples[i] = exampleSerialized
		}

		return serving.NewTensor(examples)
	}

	return serving.NewTensor(val)
}

func (sp *servingPredictor) Predict(ctx context.Context, inputs map[string]interface{}, outputFilter []string) (map[string]Tensor, ModelInfo, error) {
	inputTensorMap := make(serving.TensorMap, len(inputs))
	for key, val := range inputs {
		var err error
		inputTensorMap[key], err = sp.convertValueToTensor(val)
		if err != nil {
			return nil, ModelInfo{}, err
		}
	}

	res, err := sp.modelClient.Predict(ctx, inputTensorMap, outputFilter)
	if err != nil {
		return nil, ModelInfo{}, err
	}

	outputMap := make(map[string]Tensor, len(res.Outputs))

	for key, tensor := range res.Outputs {
		outputMap[key] = &servingPredictorTensor{t: tensor}
	}

	return outputMap, ModelInfo{
		Name:    res.ModelSpec.Name,
		Version: int(res.ModelSpec.Version.Value),
	}, nil
}

type servingPredictorTensor struct {
	t *serving.Tensor
}

func (spt *servingPredictorTensor) Value() interface{} {
	return serving.ValueFromTensor(spt.t)
}
func (spt *servingPredictorTensor) Shape() []int64 {
	return serving.ShapeFromTensor(spt.t)
}
func (spt *servingPredictorTensor) Type() TensorType {
	switch spt.t.Dtype {
	case framework.DataType_DT_FLOAT:
		return TensorTypeFloat
	case framework.DataType_DT_DOUBLE:
		return TensorTypeDouble
	case framework.DataType_DT_INT32:
		return TensorTypeInt32
	case framework.DataType_DT_UINT32:
		return TensorTypeUInt32
	case framework.DataType_DT_STRING:
		return TensorTypeString
	case framework.DataType_DT_INT64:
		return TensorTypeInt64
	case framework.DataType_DT_UINT64:
		return TensorTypeUInt64
	case framework.DataType_DT_BOOL:
		return TensorTypeBool
	case framework.DataType_DT_COMPLEX64:
		return TensorTypeComplex64
	case framework.DataType_DT_COMPLEX128:
		return TensorTypeComplex128
	default:
		panic(fmt.Errorf("unsupported type %v", spt.t.Dtype))
	}
}
