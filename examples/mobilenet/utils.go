package main

import (
	"os"
	"path"
	"runtime"
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
	return path.Join(getTestPath(), "../../../testdata/models")
}

func getLabelsFile() string {
	return path.Join(getTestPath(), "../../../testdata/labels/coco_mobilenet.txt")
}
