package predict

import (
	"context"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"reflect"
	"strings"
	"testing"
)

type testargs struct {
	modelName string
	in        map[string]interface{}
}

type test struct {
	name string
	args testargs
	want map[string]interface{}
}

func getTestSavedModelsDir() string {
	return path.Join(getTestPath(), "../../testdata/test_models")
}

func runPredict(modelName string, in map[string]interface{}) map[string]interface{} {
	embeddedPredictor, err := NewEmbeddedPredictor(getTestSavedModelsDir(), modelName, 1, "serving_default")
	if err != nil {
		panic(err)
	}

	res, _, err := embeddedPredictor.Predict(context.TODO(), in, nil)
	if err != nil {
		panic(err)
	}

	resMap := map[string]interface{}{}

	for name, tensor := range res {
		resMap[name] = tensor.Value()
	}

	return resMap
}

func getTestModelsDir() string {
	return path.Join(getTestPath(), "../../scripts/models")
}

func loadTests() (tests []test) {
	testModels := getTestModelsDir()
	files, err := ioutil.ReadDir(testModels)
	if err != nil {
		panic(err)
	}

	for _, file := range files {
		fileName := file.Name()

		loadFileAsMap := func(testName, dir string) map[string]interface{} {
			f, err := os.Open(path.Join(testModels, fmt.Sprintf("%s_%s.json", testName, dir)))
			if err != nil {
				panic(err)
			}
			defer f.Close()

			m := map[string]interface{}{}
			err = json.NewDecoder(f).Decode(&m)
			if err != nil {
				panic(err)
			}

			return m
		}

		if strings.HasSuffix(fileName, ".py") {
			testName := fileName[0 : len(fileName)-3]
			in := loadFileAsMap(testName, "in")
			out := loadFileAsMap(testName, "out")

			tests = append(tests, test{
				name: testName,
				args: testargs{
					testName, in,
				},
				want: out,
			})
		}
	}

	return tests
}

func jsonify(in map[string]interface{}) map[string]interface{} {
	out := map[string]interface{}{}

	b, err := json.Marshal(in)
	if err != nil {
		panic(err)
	}

	if err := json.Unmarshal(b, &out); err != nil {
		panic(err)
	}

	return out
}

func Test_runPredict(t *testing.T) {
	loadTests()
	tests := loadTests()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := runPredict(tt.args.modelName, tt.args.in); !reflect.DeepEqual(jsonify(got), tt.want) {
				t.Errorf("runPredict() = %v, want %v", got, tt.want)
			}
		})
	}
}
