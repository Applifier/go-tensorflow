package predict

import (
	"context"
	"os"
	"path"
	"reflect"
	"runtime"
	"testing"

	"github.com/Applifier/go-tensorflow/serving"
	"github.com/Applifier/go-tensorflow/utils"
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

func TestPredictorClassifyApi(t *testing.T) {
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
				"capital_gain":   0.0,
				"capital_loss":   0.0,
				"education":      "Masters",
				"education_num":  14.0,
				"hours_per_week": 29.0,
				"native_country": "United-States",
				"occupation":     "Prof-specialty",
				"relationship":   "Husband",
				"workclass":      "Private",
			}

			contextMap := map[string]interface{}{
				"gender": "Female",
				"age":    35.0,
			}

			example, err := utils.NewExampleFromMap(m)
			if err != nil {
				t.Error(err)
			}

			contextExample, err := utils.NewExampleFromMap(contextMap)
			if err != nil {
				t.Error(err)
			}

			res, modelInfo, err := predictor.Classify(
				context.Background(),
				[]*Example{example},
				contextExample,
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

			if res[0][0].Score != 0.54612064 {
				t.Error("Wrong result received")
			}

		})
	}
}

func TestPredictorRegressAPI(t *testing.T) {
	servingModelClient, err := serving.NewModelPredictionClientFromAddr(
		getServingAddr(),
		"wide_deep",
		"regression",
	)

	if err != nil {
		t.Fatal(err)
	}
	defer servingModelClient.Close()

	servingPredictor := NewServingPredictor(servingModelClient)

	embeddedPredictor, err := NewEmbeddedPredictor(getModelsDir(), "wide_deep", 1527087570, "regression")
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
				"capital_gain":   0.0,
				"capital_loss":   0.0,
				"education":      "Masters",
				"education_num":  14.0,
				"hours_per_week": 29.0,
				"native_country": "United-States",
				"occupation":     "Prof-specialty",
				"relationship":   "Husband",
				"workclass":      "Private",
			}

			contextMap := map[string]interface{}{
				"gender": "Female",
				"age":    35.0,
			}

			example, err := utils.NewExampleFromMap(m)
			if err != nil {
				t.Error(err)
			}

			contextExample, err := utils.NewExampleFromMap(contextMap)
			if err != nil {
				t.Error(err)
			}

			res, modelInfo, err := predictor.Regress(
				context.Background(),
				[]*Example{example},
				contextExample,
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

			if res[0].Value != 0.4538794 {
				t.Error("Wrong value returned", res[0].Value)
			}

		})
	}
}

func TestPredictorPredictAPI(t *testing.T) {
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
