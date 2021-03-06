package savedmodel

import (
	"context"
	"fmt"
	"os"
	"path"
	"strconv"
	"sync"
	"unsafe"

	"github.com/Applifier/go-tensorflow/internal/typeconv"
	"github.com/Applifier/go-tensorflow/predict"
	"github.com/Applifier/go-tensorflow/serving"
	"github.com/Applifier/go-tensorflow/utils"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

const defaultBufferSize = 2048

type savedModelPredictor struct {
	runner  *Runner
	name    string
	version int
	model 	*tf.SavedModel

	bufferPool sync.Pool
}

// NewPredictor returns a new predictor (predict.Predictor) for a given saved model folder path name and version
func NewPredictor(modelsDir string, name string, version int, signature string) (predict.Predictor, error) {
	tags := []string{"serve"}
	modelPath := path.Join(modelsDir, name, strconv.Itoa(version))
	file, err := os.Open(path.Join(modelPath, "saved_model.pb"))
	if err != nil {
		return nil, err
	}
	defer file.Close()

	signatureDef, err := GetSignatureDefFromReader(tags, signature, file)
	if err != nil {
		return nil, err
	}

	model, err := tf.LoadSavedModel(modelPath, tags, nil)
	if err != nil {
		return nil, err
	}

	runner, err := NewRunnerWithSignature(model, signatureDef)
	if err != nil {
		return nil, err
	}

	return &savedModelPredictor{
		runner:  runner,
		name:    name,
		version: version,
		model: model,

		bufferPool: sync.Pool{
			New: func() interface{} {
				return make([]byte, 0, defaultBufferSize)
			},
		},
	}, nil
}

func (ep *savedModelPredictor) getBuffer(size int) []byte {

	buf := ep.bufferPool.Get().([]byte)

	if cap(buf) >= size {
		return buf[:size]
	}

	return make([]byte, size)
}

func (ep *savedModelPredictor) putBuffer(b []byte) {
	ep.bufferPool.Put(b)
}

func (ep *savedModelPredictor) convertValueToTensor(val interface{}) (*tf.Tensor, error) {
	switch v := val.(type) {
	case *tf.Tensor:
		return v, nil
	case *serving.Tensor:
		return tf.NewTensor(serving.ValueFromTensor(v))
	case *predict.Example:
		exampleSerialized, err := v.Marshal()
		if err != nil {
			return nil, err
		}

		return tf.NewTensor([]string{string(exampleSerialized)})
	case predict.Examplifier:
		examples, err := v.Examples()
		if err != nil {
			return nil, err
		}
		examplesStrings := make([]string, len(examples))

		for i, example := range examples {
			exampleSerialized, err := example.Marshal()
			if err != nil {
				return nil, err
			}
			examplesStrings[i] = string(exampleSerialized)
		}
		return tf.NewTensor(examplesStrings)
	case map[string]interface{}:
		example, err := utils.NewExampleFromMap(v)
		if err != nil {
			return nil, err
		}
		exampleSerialized, err := example.Marshal()
		if err != nil {
			return nil, err
		}
		return tf.NewTensor([]string{string(exampleSerialized)})

	case []map[string]interface{}:
		examples := make([]string, len(v))

		for i, m := range v {
			example, err := utils.NewExampleFromMap(m)
			if err != nil {
				return nil, err
			}
			exampleSerialized, err := example.Marshal()
			if err != nil {
				return nil, err
			}
			examples[i] = string(exampleSerialized)
		}

		return tf.NewTensor(examples)
	case []interface{}:
		typedSlice, err := typeconv.ConvertInterfaceSliceToTypedSlice(v)
		if err != nil {
			return nil, err
		}

		return ep.convertValueToTensor(typedSlice)
	}

	return tf.NewTensor(val)
}

func (ep *savedModelPredictor) Predict(ctx context.Context, inputs map[string]interface{}, outputFilter []string) (map[string]predict.Tensor, predict.ModelInfo, error) {
	inputTensorMap := make(map[string]*tf.Tensor, len(inputs))
	for key, val := range inputs {
		var err error
		inputTensorMap[key], err = ep.convertValueToTensor(val)
		if err != nil {
			return nil, predict.ModelInfo{}, err
		}
	}

	res, err := ep.runner.Run(inputTensorMap, outputFilter)
	if err != nil {
		return nil, predict.ModelInfo{}, err
	}

	outputMap := make(map[string]predict.Tensor, len(res))

	for key, tensor := range res {
		outputMap[key] = &savedModelPredictorTensor{t: tensor}
	}

	return outputMap, predict.ModelInfo{
		Name:    ep.name,
		Version: ep.version,
	}, nil
}

func (ep *savedModelPredictor) marshalExample(e *predict.Example) ([]byte, error) {
	buf := ep.getBuffer(e.Size())
	n, err := e.MarshalTo(buf)
	if err != nil {
		return nil, err
	}
	return buf[:n], nil
}

func (ep *savedModelPredictor) Classify(ctx context.Context, examples []*predict.Example, context *predict.Example) ([][]predict.Class, predict.ModelInfo, error) {
	modelInfo := predict.ModelInfo{
		Name:    ep.name,
		Version: ep.version,
	}

	var contextBuf []byte
	if context != nil {
		var err error
		contextBuf, err = ep.marshalExample(context)
		if contextBuf != nil {
			defer ep.putBuffer(contextBuf)
		}

		if err != nil {
			return nil, modelInfo, err
		}
	}

	serializedExamples := make([]string, len(examples))

	for i, example := range examples {
		buf, err := ep.marshalExample(example)
		if err != nil {
			return nil, modelInfo, err
		}

		if contextBuf != nil {
			buf = append(buf, contextBuf...)
		}

		serializedExamples[i] = byteSlizeToString(buf)
		defer ep.putBuffer(buf)
	}

	inputs, err := tf.NewTensor(serializedExamples)
	if err != nil {
		return nil, modelInfo, err
	}

	res, err := ep.runner.Run(map[string]*tf.Tensor{
		"inputs": inputs,
	}, nil)
	if err != nil {
		return nil, modelInfo, err
	}

	result := make([][]predict.Class, len(examples))

	classesTensor, classesOk := res["classes"]
	scoresTensor, scoresOk := res["scores"]

	var classes [][]string
	var scores [][]float32
	var dims []int64

	if scoresOk {
		scores = scoresTensor.Value().([][]float32)
		dims = scoresTensor.Shape()
	}
	if classesOk {
		classes = classesTensor.Value().([][]string)
		if dims == nil {
			dims = classesTensor.Shape()
		}
	}

	if dims != nil {
		for exampleI := int64(0); exampleI < dims[0]; exampleI++ {
			exampleClasses := make([]predict.Class, dims[1])
			result[exampleI] = exampleClasses
			for classI := int64(0); classI < dims[1]; classI++ {
				if scoresOk {
					exampleClasses[classI].Score = scores[exampleI][classI]
				}
				if classesOk {
					exampleClasses[classI].Label = classes[exampleI][classI]
				}
			}
		}
	}

	return result, modelInfo, err
}

func (ep *savedModelPredictor) Regress(ctx context.Context, examples []*predict.Example, context *predict.Example) ([]predict.Regression, predict.ModelInfo, error) {
	modelInfo := predict.ModelInfo{
		Name:    ep.name,
		Version: ep.version,
	}

	var contextBuf []byte
	if context != nil {
		var err error
		contextBuf, err = ep.marshalExample(context)
		if contextBuf != nil {
			defer ep.putBuffer(contextBuf)
		}

		if err != nil {
			return nil, modelInfo, err
		}
	}

	serializedExamples := make([]string, len(examples))

	for i, example := range examples {
		buf, err := ep.marshalExample(example)
		if err != nil {
			return nil, modelInfo, err
		}

		if contextBuf != nil {
			buf = append(buf, contextBuf...)
		}

		serializedExamples[i] = byteSlizeToString(buf)
		defer ep.putBuffer(buf)
	}

	inputs, err := tf.NewTensor(serializedExamples)
	if err != nil {
		return nil, modelInfo, err
	}

	res, err := ep.runner.Run(map[string]*tf.Tensor{
		"inputs": inputs,
	}, nil)
	if err != nil {
		return nil, modelInfo, err
	}

	regressions := res["outputs"].Value().([][]float32)
	results := make([]predict.Regression, len(regressions))

	for i, reg := range regressions {
		results[i].Value = reg[0]
	}

	return results, predict.ModelInfo{
		Name:    ep.name,
		Version: ep.version,
	}, nil
}

func (ep *savedModelPredictor) GetModelInfo(ctx context.Context) (predict.ModelInfo, error) {
	return predict.ModelInfo{
		Name:    ep.name,
		Version: ep.version,
	}, nil
}

func (ep *savedModelPredictor) Close(ctx context.Context) error {
	return ep.model.Session.Close()
}

type savedModelPredictorTensor struct {
	t *tf.Tensor
}

func (ept *savedModelPredictorTensor) Value() interface{} {
	return ept.t.Value()
}

func (ept *savedModelPredictorTensor) Shape() []int64 {
	return ept.t.Shape()
}

func (ept *savedModelPredictorTensor) Type() predict.TensorType {
	switch ept.t.DataType() {
	case tf.Float:
		return predict.TensorTypeFloat
	case tf.Double:
		return predict.TensorTypeDouble
	case tf.Int32:
		return predict.TensorTypeInt32
	case tf.Uint32:
		return predict.TensorTypeUInt32
	case tf.String:
		return predict.TensorTypeString
	case tf.Int64:
		return predict.TensorTypeInt64
	case tf.Uint64:
		return predict.TensorTypeUInt64
	case tf.Bool:
		return predict.TensorTypeBool
	case tf.Complex64:
		return predict.TensorTypeComplex64
	case tf.Complex128:
		return predict.TensorTypeComplex128
	default:
		panic(fmt.Errorf("unsupported type %v", ept.t.DataType()))
	}
}

func byteSlizeToString(b []byte) string {
	return *(*string)(unsafe.Pointer(&b)) // nolint: gas
}
