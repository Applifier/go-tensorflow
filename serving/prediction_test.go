package serving

import (
	"context"
	"fmt"
	"os"
	"reflect"
	"testing"
)

func getServingAddr() string {
	addr := os.Getenv("SERVING_ADDR")
	if addr == "" {
		return "127.0.0.1:7000"
	}
	return addr
}

func mustTensor(t *Tensor, err error) *Tensor {
	if err != nil {
		panic(err)
	}

	return t
}

func TestPredictionClient(t *testing.T) {
	cli, err := NewModelPredictionClientFromAddr(
		getServingAddr(),
		"wide_deep",
		"serving_default",
	)
	if err != nil {
		t.Fatal(err)
	}
	defer cli.Close()
	type args struct {
		cli        ModelPredictionClient
		featureMap map[string]interface{}
	}
	tests := []struct {
		name    string
		args    args
		scores  []float32
		classes []string
		wantErr bool
	}{
		{
			args: args{
				featureMap: map[string]interface{}{
					"age":            []float32{25.0},
					"capital_gain":   []float32{0.0},
					"capital_loss":   []float32{0.0},
					"education":      []string{"11th"},
					"education_num":  []float32{7.0},
					"gender":         []string{"Male"},
					"hours_per_week": []float32{40.0},
					"native_country": []string{"United-States"},
					"occupation":     []string{"Machine-op-inspct"},
					"relationship":   []string{"Own-child"},
					"workclass":      []string{"Private"},
				},
			},
			scores:  []float32{0.9860167, 0.013983363},
			classes: []string{"0", "1"},
		},
		{
			args: args{
				featureMap: map[string]interface{}{
					"age":            []float32{35.0},
					"capital_gain":   []float32{0.0},
					"capital_loss":   []float32{0.0},
					"education":      []string{"Masters"},
					"education_num":  []float32{14.0},
					"gender":         []string{"Male"},
					"hours_per_week": []float32{29.0},
					"native_country": []string{"United-States"},
					"occupation":     []string{"Prof-specialty"},
					"relationship":   []string{"Wife"},
					"workclass":      []string{"Private"},
				},
			},
			scores:  []float32{0.48914605, 0.5108539},
			classes: []string{"0", "1"},
		},
		{
			args: args{
				featureMap: map[string]interface{}{
					"age":            []float32{35.0},
					"capital_gain":   []float32{0.0},
					"capital_loss":   []float32{0.0},
					"education":      []string{"Masters"},
					"education_num":  []float32{14.0},
					"gender":         []string{"Female"},
					"hours_per_week": []float32{29.0},
					"native_country": []string{"United-States"},
					"occupation":     []string{"Prof-specialty"},
					"relationship":   []string{"Husband"},
					"workclass":      []string{"Private"},
				},
			},
			scores:  []float32{0.54612064, 0.45387936},
			classes: []string{"0", "1"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, got1, err := testPredict(cli, tt.args.featureMap)
			if (err != nil) != tt.wantErr {
				t.Errorf("testPredict() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.scores) {
				t.Errorf("testPredict() got = %v, want %v", got, tt.scores)
			}
			if !reflect.DeepEqual(got1, tt.classes) {
				t.Errorf("testPredict() got1 = %v, want %v", got1, tt.classes)
			}
		})
	}

}

func TestMultiPrediction(t *testing.T) {
	cli, err := NewModelPredictionClientFromAddr(
		getServingAddr(),
		"wide_deep",
		"serving_default",
	)
	if err != nil {
		t.Fatal(err)
	}
	defer cli.Close()

	scores, _, err := testPredict(
		cli,
		map[string]interface{}{
			"age":            []float32{35.0},
			"capital_gain":   []float32{0.0},
			"capital_loss":   []float32{0.0},
			"education":      []string{"Masters"},
			"education_num":  []float32{14.0},
			"gender":         []string{"Female"},
			"hours_per_week": []float32{29.0},
			"native_country": []string{"United-States"},
			"occupation":     []string{"Prof-specialty"},
			"relationship":   []string{"Husband"},
			"workclass":      []string{"Private"},
		},
		map[string]interface{}{
			"age":            []float32{35.0},
			"capital_gain":   []float32{0.0},
			"capital_loss":   []float32{0.0},
			"education":      []string{"Masters"},
			"education_num":  []float32{14.0},
			"gender":         []string{"Male"},
			"hours_per_week": []float32{29.0},
			"native_country": []string{"United-States"},
			"occupation":     []string{"Prof-specialty"},
			"relationship":   []string{"Wife"},
			"workclass":      []string{"Private"},
		},
	)
	if err != nil {
		t.Fatal(err)
	}

	if !reflect.DeepEqual(scores, []float32{0.5461207, 0.45387933, 0.4891461, 0.5108539}) {
		t.Error("multi prediction result did not match expected")
	}
}

func testPredict(cli ModelPredictionClient, featureMaps ...map[string]interface{}) ([]float32, []string, error) {
	tensorData := [][]byte{}

	for _, featureMap := range featureMaps {
		example, err := NewExampleFromMap(featureMap)
		if err != nil {
			return nil, nil, err
		}
		exampleSerialized, err := example.Marshal()
		if err != nil {
			return nil, nil, err
		}
		tensorData = append(tensorData, exampleSerialized)
	}

	res, err := cli.Predict(
		context.Background(),
		TensorMap{
			"inputs": mustTensor(NewTensor(tensorData)),
		}, nil)
	if err != nil {
		return nil, nil, err
	}

	classesAsStrings := []string{}

	for _, class := range res.Outputs["classes"].StringVal {
		classesAsStrings = append(classesAsStrings, string(class))
	}

	return res.Outputs["scores"].FloatVal,
		classesAsStrings, nil
}

func ExampleModelPredictionClient() {
	// Init client
	cli, _ := NewModelPredictionClientFromAddr(
		getServingAddr(),
		"wide_deep",
		"serving_default",
	)

	// Create Example and Features
	example, _ := NewExampleFromMap(map[string]interface{}{
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
	tensor, _ := NewTensor([][]byte{exampleSerialized})

	res, _ := cli.Predict(context.Background(), TensorMap{
		"inputs": tensor,
	}, nil)

	fmt.Printf("scores %+v\n", res.Outputs["scores"].FloatVal)

	// Output: scores [0.54612064 0.45387936]
}
