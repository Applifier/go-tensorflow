[![GoDoc](https://godoc.org/github.com/Applifier/go-tensorflow?status.svg)](http://godoc.org/github.com/Applifier/go-tensorflow)
[![Build Status](https://travis-ci.org/Applifier/go-tensorflow.svg?branch=master)](https://travis-ci.org/Applifier/go-tensorflow)

# Packages
## predict

Unified interface for TensorFlow prediction for both embedded models and calls to Tensorflow Serving. Implementations automatically convert go types into matching TensorFlow Tensors.

Models should be exported in the SavedModel format.

### Example

Example uses pre-trained model found under testdata/models [wide_deep](https://github.com/tensorflow/models/tree/master/official/wide_deep)

```go
import "github.com/Applifier/go-tensorflow/predict"
```

```go

// Uncomment line below to switch implementation
// predictor := NewServingPredictor(servingModelClient)
predictor, _ := NewSavedModelPredictor("testdata/models", "wide_deep", 1527087570, "serving_default")


m := map[string]interface{}{
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
}

res, modelInfo, _ := predictor.Predict(
    context.Background(),
    map[string]interface{}{
        "inputs": m,
    },
    nil,
)

scores := res["scores"].Value().([][]float32)

fmt.Printf("scores %+v\n", scores[0])

// Output: scores [0.54612064 0.45387936]

fmt.Printf("model name %s, version %s", modelInfo.Name, modelInfo.Version)
// Output: model name wide_deep, version 1527087570

```


## serving

Go client for [Tensorflow Serving](https://github.com/tensorflow/serving)

### Example

Example uses pre-trained model found under testdata/models [wide_deep](https://github.com/tensorflow/models/tree/master/official/wide_deep)

```go
import "github.com/Applifier/go-tensorflow/serving"
```

```go

// Init client
cli, _ := serving.NewModelPredictionClientFromAddr(
    getServingAddr(),
    "wide_deep",
    "serving_default",
    1527087570,
)

// Create Example and Features
example, _ := serving.NewExampleFromMap(map[string]interface{}{
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

res, _ := cli.Predict(context.Background(), serving.TensorMap{
    "inputs": tensor,
}, nil)

fmt.Printf("scores %+v\n", res.Outputs["scores"].FloatVal)

// Output: scores [0.54612064 0.45387936]

```


## License

[MIT](https://github.com/Applifier/go-tensorflow/blob/master/LICENSE)
