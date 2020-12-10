package utils

import (
	"bytes"
	"io"
	"os"
	"path"
	"runtime"
	"testing"

	"github.com/Unity-Technologies/go-tensorflow/types/tensorflow_serving"
)

func TestPredictionLog(t *testing.T) {
	filePath := getTestFile("tf_serving_warmup_requests")
	file, err := os.Open(filePath)
	if err != nil {
		t.Fatal(err)
	}

	pr := NewPredictionLogReader(file)

	err = pr.Next()
	if err != nil {
		t.Fatal(err)
	}

	log := tensorflow_serving.PredictionLog{}
	err = pr.Unmarshal(&log)
	if err != nil {
		t.Fatal(err)
	}

	if log.GetPredictLog().Request.ModelSpec.Name != "some_model" {
		t.Error("wrong model name in prediction log", log.GetPredictLog().Request.ModelSpec.Name)
	}
}

func TestPredictionlogEmpty(t *testing.T) {
	pr := NewPredictionLogReader(bytes.NewReader([]byte{}))

	if pr.Next() != io.EOF {
		t.Error("should have returned EOF")
	}
}

func getTestPath() string {
	_, filename, _, _ := runtime.Caller(0)
	return filename
}

func getTestFile(name string) string {
	return path.Join(getTestPath(), "../../testdata/", name)
}
