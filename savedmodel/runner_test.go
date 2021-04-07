package savedmodel

import (
	"reflect"
	"testing"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"

	"github.com/Applifier/go-tensorflow/utils"
)

func TestRunner(t *testing.T) {
	model, err := tf.LoadSavedModel(getModelDir("wide_deep", 1527087570), []string{"serve"}, nil)
	if err != nil {
		t.Fatal(err)
	}

	testModelReader := getTestModel("wide_deep", 1527087570)
	defer testModelReader.Close()

	runner, err := NewRunnerWithSignature(model, "serving_default")
	if err != nil {
		t.Fatal(err)
	}

	example, _ := utils.NewExampleFromMap(map[string]interface{}{
		"age":            35.0,
		"capital_gain":   0.0,
		"capital_loss":   0.0,
		"education":      "Masters",
		"education_num":  14.0,
		"gender":         "Female",
		"hours_per_week": 29.0,
		"native_country": "United-States",
		"occupation":     "Prof-specialty",
		"relationship":   "Husband",
		"workclass":      "Private",
	})

	// Convert example to protobuf
	exampleSerialized, _ := example.Marshal()

	// Convert serialized example to tensor
	tensor, err := tf.NewTensor([]string{string(exampleSerialized)})
	if err != nil {
		t.Error(err)
	}

	res, err := runner.Run(map[string]*tf.Tensor{
		"inputs": tensor,
	}, nil)

	if err != nil {
		t.Fatal(err)
	}

	scores := res["scores"].Value().([][]float32)[0]

	if !reflect.DeepEqual(scores, []float32{0.54612064, 0.45387936}) {
		t.Error("invalid result received")
	}
}
