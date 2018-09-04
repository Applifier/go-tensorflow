package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path"
	"strings"

	"github.com/Applifier/go-tensorflow/internal/typeconv"
	"github.com/Applifier/go-tensorflow/savedmodel"
	"github.com/Applifier/go-tensorflow/utils"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

var modelpath string
var tag string
var signature string
var input string

func main() {
	flag.StringVar(&modelpath, "modelpath", "", "Path to saved model")
	flag.StringVar(&tag, "tag", "serve", "Signature tag")
	flag.StringVar(&signature, "signature", "serving_default", "Signature")
	flag.StringVar(&input, "input", "", "Input data for running the prediction")
	flag.Parse()

	var inputData io.Reader
	var err error
	inputData, err = os.Open(input)
	if err != nil {
		// fallback to using input directly as the value
		inputData = strings.NewReader(input)
	}

	inputs := map[string]interface{}{}
	if err := json.NewDecoder(inputData).Decode(&inputs); err != nil {
		panic(err)
	}

	fmt.Printf("Input: %+v\n", inputs)

	file, err := os.Open(path.Join(modelpath, "saved_model.pb"))
	if err != nil {
		panic(err)
	}

	signatureDef, err := savedmodel.GetSignatureDefFromReader([]string{tag}, signature, file)
	if err != nil {
		panic(err)
	}
	file.Close()

	model, err := tf.LoadSavedModel(modelpath, []string{tag}, nil)
	if err != nil {
		panic(err)
	}

	runner, err := savedmodel.NewRunnerWithSignature(model, signatureDef)
	if err != nil {
		panic(err)
	}

	inputTensorMap := make(map[string]*tf.Tensor, len(inputs))
	for key, val := range inputs {
		var err error
		inputTensorMap[key], err = convertValueToTensor(val)
		if err != nil {
			panic(err)
		}
	}

	outputTensors, err := runner.Run(inputTensorMap, nil)
	if err != nil {
		panic(err)
	}

	outputs := map[string]interface{}{}
	for key, tensor := range outputTensors {
		outputs[key] = tensor.Value()
	}

	jsonBytes, err := json.MarshalIndent(outputs, "", "  ")
	if err != nil {
		panic(err)
	}
	fmt.Printf("Output:\n%s\n", string(jsonBytes))
}

func convertValueToTensor(val interface{}) (*tf.Tensor, error) {
	switch v := val.(type) {
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

		return convertValueToTensor(typedSlice)
	}

	return tf.NewTensor(val)
}
