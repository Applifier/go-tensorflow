[![GoDoc](https://godoc.org/github.com/Applifier/go-tensorflow?status.svg)](http://godoc.org/github.com/Applifier/go-tensorflow)
[![Build Status](https://travis-ci.com/Applifier/go-tensorflow.svg?branch=master)](https://travis-ci.com/Applifier/go-tensorflow)

# Packages
## serving

Go client for [Tensorflow Serving](https://github.com/tensorflow/serving)

### Example

Example uses pre-trained model found under testdata/models [wide_deep](https://github.com/tensorflow/models/tree/master/official/wide_deep)

```
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