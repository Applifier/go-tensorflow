package serving

import (
	"context"
	"fmt"
	"sync"

	"github.com/Applifier/go-tensorflow/internal/typeconv"
	"github.com/Applifier/go-tensorflow/predict"
	"github.com/Applifier/go-tensorflow/types/tensorflow/core/framework"
	"github.com/Applifier/go-tensorflow/utils"

	st "github.com/Applifier/go-tensorflow/types/tensorflow_serving"
)

// servingPredictor implementation of the Predictor interface for TensorFlow Serving
type servingPredictor struct {
	modelClient ModelPredictionClient

	tensorMapPool sync.Pool
}

// NewPredictor returns a new predictor (implements Predictor interface) using a given TF Serving client
func NewPredictor(modelClient ModelPredictionClient) predict.Predictor {
	return &servingPredictor{
		modelClient: modelClient,
		tensorMapPool: sync.Pool{
			New: func() interface{} {
				return TensorMap{}
			},
		},
	}
}

func (sp *servingPredictor) getTensorMap() TensorMap {
	return sp.tensorMapPool.Get().(TensorMap)
}

func (sp *servingPredictor) putTensorMap(m TensorMap) {
	for key := range m {
		delete(m, key)
	}
	sp.tensorMapPool.Put(m)
}

func (sp *servingPredictor) convertValueToTensor(val interface{}) (*Tensor, error) {
	switch v := val.(type) {
	case *Tensor:
		return v, nil
	case nativeTensor:
		return NewTensor(v.Value())
	case *predict.Example:
		exampleSerialized, err := v.Marshal()
		if err != nil {
			return nil, err
		}

		return NewTensor([][]byte{exampleSerialized})
	case predict.Examplifier:
		examples, err := v.Examples()
		if err != nil {
			return nil, err
		}

		examplesBytes := make([][]byte, len(examples))

		for i, example := range examples {
			exampleSerialized, err := example.Marshal()
			if err != nil {
				return nil, err
			}
			examplesBytes[i] = exampleSerialized
		}

		return NewTensor(examplesBytes)
	case map[string]interface{}:
		example, err := utils.NewExampleFromMap(v)
		if err != nil {
			return nil, err
		}
		exampleSerialized, err := example.Marshal()
		if err != nil {
			return nil, err
		}
		return NewTensor([][]byte{exampleSerialized})

	case []map[string]interface{}:
		examples := make([][]byte, len(v))

		for i, m := range v {
			example, err := utils.NewExampleFromMap(m)
			if err != nil {
				return nil, err
			}
			exampleSerialized, err := example.Marshal()
			if err != nil {
				return nil, err
			}
			examples[i] = exampleSerialized
		}

		return NewTensor(examples)
	case []interface{}:
		typedSlice, err := typeconv.ConvertInterfaceSliceToTypedSlice(v)
		if err != nil {
			return nil, err
		}

		return sp.convertValueToTensor(typedSlice)
	}

	return NewTensor(val)
}

func (sp *servingPredictor) Predict(ctx context.Context, inputs map[string]interface{}, outputFilter []string) (map[string]predict.Tensor, predict.ModelInfo, error) {
	inputTensorMap := sp.getTensorMap()
	defer sp.putTensorMap(inputTensorMap)

	for key, val := range inputs {
		var err error
		inputTensorMap[key], err = sp.convertValueToTensor(val)
		if err != nil {
			return nil, predict.ModelInfo{}, err
		}
	}

	res, err := sp.modelClient.Predict(ctx, inputTensorMap, outputFilter)
	if err != nil {
		return nil, predict.ModelInfo{}, err
	}

	outputMap := make(map[string]predict.Tensor, len(res.Outputs))

	for key, tensor := range res.Outputs {
		outputMap[key] = &servingPredictorTensor{t: tensor}
	}

	return outputMap, predict.ModelInfo{
		Name:    res.ModelSpec.Name,
		Version: int(res.ModelSpec.VersionChoice.(*st.ModelSpec_Version).Version.Value),
	}, nil
}

func (sp *servingPredictor) convertExamplesToInput(examples []*predict.Example, context *predict.Example) (*st.Input, error) {
	input := &st.Input{}

	if context == nil {
		input.Kind = &st.Input_ExampleList{
			ExampleList: &st.ExampleList{
				Examples: examples,
			},
		}
	} else {
		input.Kind = &st.Input_ExampleListWithContext{
			ExampleListWithContext: &st.ExampleListWithContext{
				Context:  context,
				Examples: examples,
			},
		}
	}

	return input, nil
}

func (sp *servingPredictor) Classify(ctx context.Context, examples []*predict.Example, context *predict.Example) ([][]predict.Class, predict.ModelInfo, error) {
	input, err := sp.convertExamplesToInput(examples, context)

	if err != nil {
		return nil, predict.ModelInfo{}, err
	}

	res, err := sp.modelClient.Classify(ctx, input)
	if err != nil {
		return nil, predict.ModelInfo{}, err
	}

	result := make([][]predict.Class, len(res.Result.Classifications))
	for i, classifications := range res.Result.Classifications {
		classes := make([]predict.Class, len(classifications.Classes))
		result[i] = classes
		for i, class := range classifications.Classes {
			classes[i].Label = class.Label
			classes[i].Score = class.Score
		}
	}

	return result, predict.ModelInfo{
		Name:    res.ModelSpec.Name,
		Version: int(res.ModelSpec.VersionChoice.(*st.ModelSpec_Version).Version.Value),
	}, nil
}

func (sp *servingPredictor) Regress(ctx context.Context, examples []*predict.Example, context *predict.Example) ([]predict.Regression, predict.ModelInfo, error) {
	input, err := sp.convertExamplesToInput(examples, context)

	if err != nil {
		return nil, predict.ModelInfo{}, err
	}

	res, err := sp.modelClient.Regress(ctx, input)
	if err != nil {
		return nil, predict.ModelInfo{}, err
	}

	regressions := make([]predict.Regression, len(examples))
	for i, regression := range res.Result.Regressions {
		regressions[i].Value = regression.Value
	}

	return regressions, predict.ModelInfo{
		Name:    res.ModelSpec.Name,
		Version: int(res.ModelSpec.VersionChoice.(*st.ModelSpec_Version).Version.Value),
	}, nil
}

func (sp *servingPredictor) GetModelInfo(ctx context.Context) (predict.ModelInfo, error) {
	res, err := sp.modelClient.GetModelMetadata(ctx)
	if err != nil {
		return predict.ModelInfo{}, err
	}

	return predict.ModelInfo{
		Name:    res.ModelSpec.Name,
		Version: int(res.ModelSpec.VersionChoice.(*st.ModelSpec_Version).Version.Value),
	}, nil
}

type servingPredictorTensor struct {
	t *Tensor
}

func (spt *servingPredictorTensor) Value() interface{} {
	return ValueFromTensor(spt.t)
}
func (spt *servingPredictorTensor) Shape() []int64 {
	return ShapeFromTensor(spt.t)
}
func (spt *servingPredictorTensor) Type() predict.TensorType {
	switch spt.t.Dtype {
	case framework.DataType_DT_FLOAT:
		return predict.TensorTypeFloat
	case framework.DataType_DT_DOUBLE:
		return predict.TensorTypeDouble
	case framework.DataType_DT_INT32:
		return predict.TensorTypeInt32
	case framework.DataType_DT_UINT32:
		return predict.TensorTypeUInt32
	case framework.DataType_DT_STRING:
		return predict.TensorTypeString
	case framework.DataType_DT_INT64:
		return predict.TensorTypeInt64
	case framework.DataType_DT_UINT64:
		return predict.TensorTypeUInt64
	case framework.DataType_DT_BOOL:
		return predict.TensorTypeBool
	case framework.DataType_DT_COMPLEX64:
		return predict.TensorTypeComplex64
	case framework.DataType_DT_COMPLEX128:
		return predict.TensorTypeComplex128
	default:
		panic(fmt.Errorf("unsupported type %v", spt.t.Dtype))
	}
}

type nativeTensor interface {
	Value() interface{}
	Shape() []int64
}
