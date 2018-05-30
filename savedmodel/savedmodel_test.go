package savedmodel

import (
	"io"
	"os"
	"path"
	"reflect"
	"runtime"
	"strconv"
	"testing"

	serving "github.com/Applifier/go-tensorflow/serving"
	protobuf "github.com/Applifier/go-tensorflow/types/tensorflow/core/protobuf"

	"github.com/Applifier/go-tensorflow/types/tensorflow/core/framework"
)

func getTestPath() string {
	_, filename, _, _ := runtime.Caller(0)
	return filename
}

func getModelDir(name string, version int) string {
	return path.Join(getTestPath(), "../../testdata/models", name, strconv.Itoa(version))
}

func getTestModel(name string, version int) io.ReadCloser {
	rc, err := os.Open(path.Join(getModelDir(name, version), "saved_model.pb"))
	if err != nil {
		panic(err)
	}

	return rc
}

func TestGetSignatureDefFromReader(t *testing.T) {
	testModelReader := getTestModel("wide_deep", 1527087570)
	defer testModelReader.Close()

	type args struct {
		tags      []string
		signature string
		r         io.Reader
	}
	tests := []struct {
		name    string
		args    args
		want    *protobuf.SignatureDef
		wantErr bool
	}{
		{
			name: "serving_default",
			args: args{
				tags:      []string{"serve"},
				signature: "serving_default",
				r:         testModelReader,
			},
			want: &protobuf.SignatureDef{
				MethodName: "tensorflow/serving/classify",
				Inputs: map[string]*protobuf.TensorInfo{
					"inputs": &protobuf.TensorInfo{
						Dtype:       framework.DataType_DT_STRING,
						TensorShape: serving.NewShape([]int64{-1}),
						Encoding: &protobuf.TensorInfo_Name{
							Name: "input_example_tensor:0",
						},
					},
				},
				Outputs: map[string]*protobuf.TensorInfo{
					"classes": &protobuf.TensorInfo{
						Dtype:       framework.DataType_DT_STRING,
						TensorShape: serving.NewShape([]int64{-1, 2}),
						Encoding: &protobuf.TensorInfo_Name{
							Name: "head/Tile:0",
						},
					},
					"scores": &protobuf.TensorInfo{
						Dtype:       framework.DataType_DT_FLOAT,
						TensorShape: serving.NewShape([]int64{-1, 2}),
						Encoding: &protobuf.TensorInfo_Name{
							Name: "head/predictions/probabilities:0",
						},
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := GetSignatureDefFromReader(tt.args.tags, tt.args.signature, tt.args.r)
			if (err != nil) != tt.wantErr {
				t.Errorf("GetSignatureDefFromReader() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("GetSignatureDefFromReader() = %v, want %v", got, tt.want)
			}
		})
	}
}
