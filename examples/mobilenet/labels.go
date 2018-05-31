package main

import (
	"io/ioutil"
	"os"
	"strings"
)

func loadLabels(labelFilePath string) []string {
	file, err := os.Open(labelFilePath)
	if err != nil {
		panic(err)
	}

	data, err := ioutil.ReadAll(file)
	if err != nil {
		panic(err)
	}

	return strings.Split(string(data), "\n")
}
