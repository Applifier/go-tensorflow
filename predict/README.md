[![GoDoc](https://godoc.org/github.com/Applifier/go-tensorflow/predict?status.svg)](http://godoc.org/github.com/Applifier/go-tensorflow/predict)
[![Build Status](https://travis-ci.com/Applifier/go-tensorflow.svg?token=jeWt6weUpeDp6aNSSaST&branch=master)](https://travis-ci.com/Applifier/go-tensorflow)

## predict

Unified interface for TensorFlow prediction for both embedded models and calls to Tensorflow Serving. Implementations automatically convert go types into matching TensorFlow Tensors.

### Example

Example uses pre-trained model found under testdata/models [wide_deep](https://github.com/tensorflow/models/tree/master/official/wide_deep)

```go
import "github.com/Applifier/go-tensorflow/predict"
```

```go

// Uncomment line below to switch implementation
// predictor := NewServingPredictor(servingModelClient)
predictor, _ := NewEmbeddedPredictor("testdata/models", "wide_deep", 1527087570, "serving_default")


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
