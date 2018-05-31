package predict

import (
	"context"
	"fmt"
	"os"
	"path"
	"strconv"

	"github.com/Applifier/go-tensorflow/savedmodel"
	"github.com/Applifier/go-tensorflow/serving"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

type embeddedPredictor struct {
	runner  *savedmodel.Runner
	name    string
	version int
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
	}, nil
}

func (ep *embeddedPredictor) convertValueToTensor(val interface{}) (*tf.Tensor, error) {
	switch v := val.(type) {
	case *serving.Tensor:
		return tf.NewTensor(serving.ValueFromTensor(v))
	case map[string]interface{}:
		example, err := serving.NewExampleFromMap(v)
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
			example, err := serving.NewExampleFromMap(m)
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
