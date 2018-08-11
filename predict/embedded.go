package predict

import (
	"context"
	"fmt"
	"os"
	"path"
	"strconv"
	"sync"
	"unsafe"

	"github.com/Applifier/go-tensorflow/savedmodel"
	"github.com/Applifier/go-tensorflow/serving"
	"github.com/Applifier/go-tensorflow/utils"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

const defaultBufferSize = 2048

type embeddedPredictor struct {
	runner  *savedmodel.Runner
	name    string
	version int

	bufferPool sync.Pool
}

// NewEmbeddedPredictor returns a new embedded predictor for a given saved model folder path name and version
func NewEmbeddedPredictor(modelsDir string, name string, version int, signature string) (Predictor, error) {
	tags := []string{"serve"}
	modelPath := path.Join(modelsDir, name, strconv.Itoa(version))
	file, err := os.Open(path.Join(modelPath, "saved_model.pb"))
	if err != nil {
		return nil, err
	}
	defer file.Close()

	signatureDef, err := savedmodel.GetSignatureDefFromReader(tags, signature, file)
	if err != nil {
		return nil, err
	}

	model, err := tf.LoadSavedModel(modelPath, tags, nil)
	if err != nil {
		return nil, err
	}

	runner, err := savedmodel.NewRunnerWithSignature(model, signatureDef)
	if err != nil {
		return nil, err
	}

	return &embeddedPredictor{
		runner:  runner,
		name:    name,
		version: version,

		bufferPool: sync.Pool{
			New: func() interface{} {
				return make([]byte, 0, defaultBufferSize)
			},
		},
	}, nil
}

func (ep *embeddedPredictor) getBuffer(size int) []byte {

	buf := ep.bufferPool.Get().([]byte)

	if cap(buf) >= size {
		return buf[:size]
	}

	return make([]byte, size)
}

func (ep *embeddedPredictor) putBuffer(b []byte) {
	ep.bufferPool.Put(b)
}

func (ep *embeddedPredictor) convertValueToTensor(val interface{}) (*tf.Tensor, error) {
	switch v := val.(type) {
	case *tf.Tensor:
		return v, nil
	case *serving.Tensor:
		return tf.NewTensor(serving.ValueFromTensor(v))
	case *Example:
		exampleSerialized, err := v.Marshal()
		if err != nil {
			return nil, err
		}

		return tf.NewTensor([]string{string(exampleSerialized)})
	case Examplifier:
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
		return tf.NewTensor(examples)
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
	}

	return tf.NewTensor(val)
}

func (ep *embeddedPredictor) Predict(ctx context.Context, inputs map[string]interface{}, outputFilter []string) (map[string]Tensor, ModelInfo, error) {
	inputTensorMap := make(map[string]*tf.Tensor, len(inputs))
	for key, val := range inputs {
		var err error
		inputTensorMap[key], err = ep.convertValueToTensor(val)
		if err != nil {
			return nil, ModelInfo{}, err
		}
	}

	res, err := ep.runner.Run(inputTensorMap, outputFilter)
	if err != nil {
		return nil, ModelInfo{}, err
	}

	outputMap := make(map[string]Tensor, len(res))

	for key, tensor := range res {
		outputMap[key] = &embeddedPredictorTensor{t: tensor}
	}

	return outputMap, ModelInfo{
		Name:    ep.name,
		Version: ep.version,
	}, nil
}

func (ep *embeddedPredictor) marshalExample(e *Example) ([]byte, error) {
	buf := ep.getBuffer(e.Size())
	n, err := e.MarshalTo(buf)
	if err != nil {
		return nil, err
	}
	return buf[:n], nil
}

func (ep *embeddedPredictor) Classify(ctx context.Context, examples []*Example, context *Example) ([][]Class, ModelInfo, error) {
	modelInfo := ModelInfo{
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

	result := make([][]Class, len(examples))

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
			exampleClasses := make([]Class, dims[1])
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

func (ep *embeddedPredictor) Regress(ctx context.Context, examples []*Example, context *Example) ([]Regression, ModelInfo, error) {
	return nil, ModelInfo{
		Name:    ep.name,
		Version: ep.version,
	}, nil
}

type embeddedPredictorTensor struct {
	t *tf.Tensor
}

func (ept *embeddedPredictorTensor) Value() interface{} {
	return ept.t.Value()
}

func (ept *embeddedPredictorTensor) Shape() []int64 {
	return ept.t.Shape()
}

func (ept *embeddedPredictorTensor) Type() TensorType {
	switch ept.t.DataType() {
	case tf.Float:
		return TensorTypeFloat
	case tf.Double:
		return TensorTypeDouble
	case tf.Int32:
		return TensorTypeInt32
	case tf.Uint32:
		return TensorTypeUInt32
	case tf.String:
		return TensorTypeString
	case tf.Int64:
		return TensorTypeInt64
	case tf.Uint64:
		return TensorTypeUInt64
	case tf.Bool:
		return TensorTypeBool
	case tf.Complex64:
		return TensorTypeComplex64
	case tf.Complex128:
		return TensorTypeComplex128
	default:
		panic(fmt.Errorf("unsupported type %v", ept.t.DataType()))
	}
}

func byteSlizeToString(b []byte) string {
	return *(*string)(unsafe.Pointer(&b)) // nolint: gas
}
