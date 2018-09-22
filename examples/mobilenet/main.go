package main

import (
	"context"
	"fmt"
	"io/ioutil"
	"os"
	"path"

	"github.com/Applifier/go-tensorflow/predict"
	"github.com/Applifier/go-tensorflow/savedmodel"
	"github.com/Applifier/go-tensorflow/serving"
)

func getLocalPredictor() predict.Predictor {
	pred, err := savedmodel.NewPredictor(getModelsDir(), "mobilenet", 1, "serving_default")
	if err != nil {
		panic(err)
	}

	return pred
}

func getServingPredictor() predict.Predictor {
	servingModelClient, err := serving.NewModelPredictionClientFromAddr(
		getServingAddr(),
		"mobilenet",
		"serving_default",
	)

	if err != nil {
		panic(err)
	}

	return serving.NewPredictor(servingModelClient)
}

func readImage(url string) []byte {
	file, err := os.Open(url)
	if err != nil {
		panic(err)
	}

	defer file.Close()

	b, err := ioutil.ReadAll(file)
	if err != nil {
		panic(err)
	}

	return b
}

func main() {
	imagePath := os.Args[1]
	imageBytes := readImage(imagePath)
	labels := loadLabels(getLabelsFile())

	tensor, err := makeTensorFromImage(imageBytes, path.Ext(imagePath))
	if err != nil {
		panic(err)
	}

	// Run prediction both against a local model and tensorflow serving
	for name, predictor := range map[string]predict.Predictor{
		"serving":  getServingPredictor(),
		"embedded": getLocalPredictor(),
	} {

		fmt.Printf("\nRunning prediction on %s\n", name)
		res, _, err := predictor.Predict(
			context.Background(),
			map[string]interface{}{
				"inputs": tensor,
			},
			nil,
		)

		if err != nil {
			panic(err)
		}

		classes := res["detection_classes"].Value().([][]float32)
		scores := res["detection_scores"].Value().([][]float32)

		fmt.Printf("Results from %s\n", name)
		for i, val := range classes[0] {
			label := labels[int(val)]
			fmt.Printf("%s = %f\n", label, scores[0][i])
		}
	}

}
