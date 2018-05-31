package predict

import (
	"context"
	"os"
	"path"
	"reflect"
	"runtime"
	"testing"

	"github.com/Applifier/go-tensorflow/serving"
)

func getServingAddr() string {
	addr := os.Getenv("SERVING_ADDR")
	if addr == "" {
		return "127.0.0.1:7000"
	}
	return addr
}

func getTestPath() string {
	_, filename, _, _ := runtime.Caller(0)
	return filename
}

func getModelsDir() string {
	return path.Join(getTestPath(), "../../testdata/models")
}

func TestPredictorAPI(t *testing.T) {
	servingModelClient, err := serving.NewModelPredictionClientFromAddr(
		getServingAddr(),
		"wide_deep",
		"serving_default",
	)

	if err != nil {
		t.Fatal(err)
	}
	defer servingModelClient.Close()

	servingPredictor := NewServingPredictor(servingModelClient)

	embeddedPredictor, err := NewEmbeddedPredictor(getModelsDir(), "wide_deep", 1527087570, "serving_default")
	if err != nil {
		t.Fatal(err)
	}

	predictors := map[string]Predictor{
		"serving":  servingPredictor,
		"embedded": embeddedPredictor,
	}

	for name, predictor := range predictors {
		t.Run(name, func(t *testing.T) {
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

			res, modelInfo, err := predictor.Predict(
				context.Background(),
				map[string]interface{}{
					"inputs": m,
				},
				nil,
			)

			if err != nil {
				t.Error(err)
			}

			if modelInfo.Name != "wide_deep" {
				t.Error("Wrong model name returned")
			}

			if modelInfo.Version != 1527087570 {
				t.Error("Wrong model version returned")
			}

			scores := res["scores"].Value().([][]float32)[0]
			if !reflect.DeepEqual(scores, []float32{0.54612064, 0.45387936}) {
				t.Error("invalid result received", scores)
			}
		})
	}
}
