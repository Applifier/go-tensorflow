package savedmodel

import (
	"strconv"
	"strings"

	protobuf "github.com/Applifier/go-tensorflow/types/tensorflow/core/protobuf"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

// Runner transform Session.Run input and output based on given SignatureDef
type Runner struct {
	signatureDef *protobuf.SignatureDef
	savedModel   *tf.SavedModel

	feedMapping   map[string]tf.Output
	outputMapping map[string]tf.Output

	outputs []string
}

// NewRunnerWithSignature returns a new Runner for a given SignatureDef and tf.SavedModel
func NewRunnerWithSignature(savedModel *tf.SavedModel, signatureDef *protobuf.SignatureDef) (*Runner, error) {
	sr := &Runner{
		signatureDef: signatureDef,
		savedModel:   savedModel,

		feedMapping:   make(map[string]tf.Output, len(signatureDef.Inputs)),
		outputMapping: make(map[string]tf.Output, len(signatureDef.Outputs)),
		outputs:       make([]string, 0, len(signatureDef.Outputs)),
	}
	return sr, sr.init()
}

// NewRunner returns a new Runner for a given tag set, signature and tf.SavedModel
func NewRunner(savedModel *tf.SavedModel, tags []string, signature string) (*Runner, error) {
	panic("Not implemented yet")
}

func (sr *Runner) init() error {
	for inputName, def := range sr.signatureDef.Inputs {
		encoding := def.Encoding.(*protobuf.TensorInfo_Name)
		parts := strings.Split(encoding.Name, ":")
		name := parts[0]
		index, err := strconv.Atoi(parts[1])
		if err != nil {
			return err
		}

		sr.feedMapping[inputName] = tf.Output{
			Op:    sr.savedModel.Graph.Operation(name),
			Index: index,
		}
	}

	for outputName, def := range sr.signatureDef.Outputs {
		encoding := def.Encoding.(*protobuf.TensorInfo_Name)
		parts := strings.Split(encoding.Name, ":")
		name := parts[0]
		index, err := strconv.Atoi(parts[1])
		if err != nil {
			return err
		}

		sr.outputMapping[outputName] = tf.Output{
			Op:    sr.savedModel.Graph.Operation(name),
			Index: index,
		}

		sr.outputs = append(sr.outputs, outputName)
	}

	return nil
}

// Run executes contained SaveModel and uses given SignatureDef for mapping feeds and outputs
func (sr *Runner) Run(inputs map[string]*tf.Tensor, outputFilter []string) (map[string]*tf.Tensor, error) {
	feeds := make(map[tf.Output]*tf.Tensor, len(inputs))

	for name, feed := range inputs {
		mapping, ok := sr.feedMapping[name]
		// TODO return error if mapping was not found
		if ok {
			feeds[mapping] = feed
		}
	}

	outputs := sr.outputs
	if outputFilter != nil {
		outputs = outputFilter
	}

	fetches := make([]tf.Output, len(outputs))

	for i, output := range outputs {
		fetches[i] = sr.outputMapping[output]
	}

	res, err := sr.savedModel.Session.Run(feeds, fetches, nil)
	if err != nil {
		return nil, err
	}

	results := make(map[string]*tf.Tensor, len(outputs))

	for i, output := range outputs {
		results[output] = res[i]
	}

	return results, nil
}
