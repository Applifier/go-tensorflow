package serving

import (
	"reflect"
	"testing"

	"github.com/Applifier/go-tensorflow/types/tensorflow/core/framework"
)

func TestNewTensorWithShape(t *testing.T) {
	raw := []float32{1, 2, 3, 4}
	tensor, err := NewTensorWithShape(raw, []int64{4, 1})
	if err != nil {
		t.Fatal(err)
	}

	v := ValueFromTensor(tensor)

	mat, ok := v.([][]float32)
	if !ok {
		t.Error("wrong value returned")
	}

	for i, val := range raw {
		if mat[i][0] != val {
			t.Error("wrong value received")
		}
	}

	if reflect.ValueOf(tensor.FloatVal).Pointer() != reflect.ValueOf(raw).Pointer() {
		t.Error("should be backed with original slice")
	}
}

func TestNewTensor(t *testing.T) {
	type args struct {
		value interface{}
	}
	tests := []struct {
		name    string
		args    args
		want    *Tensor
		wantErr bool
	}{
		{
			name: "array of int32",
			args: args{
				value: []int32{1, 2, 3},
			},
			want: &Tensor{
				Dtype:       framework.DataType_DT_INT32,
				TensorShape: NewShape([]int64{3}),
				IntVal:      []int32{1, 2, 3},
			},
		},
		{
			name: "array of uint32",
			args: args{
				value: []uint32{1, 2, 3},
			},
			want: &Tensor{
				Dtype:       framework.DataType_DT_UINT32,
				TensorShape: NewShape([]int64{3}),
				Uint32Val:   []uint32{1, 2, 3},
			},
		},
		{
			name: "2d array of uint32",
			args: args{
				value: [][]uint32{[]uint32{1, 2, 3}, []uint32{1, 2, 3}, []uint32{1, 2, 3}},
			},
			want: &Tensor{
				Dtype:       framework.DataType_DT_UINT32,
				TensorShape: NewShape([]int64{3, 3}),
				Uint32Val:   []uint32{1, 2, 3, 1, 2, 3, 1, 2, 3},
			},
		},
		{
			name: "3d array of uint32",
			args: args{
				value: [][][]uint32{[][]uint32{[]uint32{1, 2, 3}, []uint32{1, 2, 3}, []uint32{1, 2, 3}}},
			},
			want: &Tensor{
				Dtype:       framework.DataType_DT_UINT32,
				TensorShape: NewShape([]int64{1, 3, 3}),
				Uint32Val:   []uint32{1, 2, 3, 1, 2, 3, 1, 2, 3},
			},
		},
		{
			name: "2d array of int32",
			args: args{
				value: [][]int32{[]int32{1, 2, 3}, []int32{1, 2, 3}, []int32{1, 2, 3}},
			},
			want: &Tensor{
				Dtype:       framework.DataType_DT_INT32,
				TensorShape: NewShape([]int64{3, 3}),
				IntVal:      []int32{1, 2, 3, 1, 2, 3, 1, 2, 3},
			},
		},
		{
			name: "3d array of int32",
			args: args{
				value: [][][]int32{[][]int32{[]int32{1, 2, 3}, []int32{1, 2, 3}, []int32{1, 2, 3}}},
			},
			want: &Tensor{
				Dtype:       framework.DataType_DT_INT32,
				TensorShape: NewShape([]int64{1, 3, 3}),
				IntVal:      []int32{1, 2, 3, 1, 2, 3, 1, 2, 3},
			},
		},
		{
			name: "array of int64",
			args: args{
				value: []int64{1, 2, 3},
			},
			want: &Tensor{
				Dtype:       framework.DataType_DT_INT64,
				TensorShape: NewShape([]int64{3}),
				Int64Val:    []int64{1, 2, 3},
			},
		},
		{
			name: "array of uint64",
			args: args{
				value: []uint64{1, 2, 3},
			},
			want: &Tensor{
				Dtype:       framework.DataType_DT_UINT64,
				TensorShape: NewShape([]int64{3}),
				Uint64Val:   []uint64{1, 2, 3},
			},
		},
		{
			name: "2d array of uint64",
			args: args{
				value: [][]uint64{[]uint64{1, 2, 3}, []uint64{1, 2, 3}, []uint64{1, 2, 3}},
			},
			want: &Tensor{
				Dtype:       framework.DataType_DT_UINT64,
				TensorShape: NewShape([]int64{3, 3}),
				Uint64Val:   []uint64{1, 2, 3, 1, 2, 3, 1, 2, 3},
			},
		},
		{
			name: "3d array of uint64",
			args: args{
				value: [][][]uint64{[][]uint64{[]uint64{1, 2, 3}, []uint64{1, 2, 3}, []uint64{1, 2, 3}}},
			},
			want: &Tensor{
				Dtype:       framework.DataType_DT_UINT64,
				TensorShape: NewShape([]int64{1, 3, 3}),
				Uint64Val:   []uint64{1, 2, 3, 1, 2, 3, 1, 2, 3},
			},
		},
		{
			name: "2d array of int64",
			args: args{
				value: [][]int64{[]int64{1, 2, 3}, []int64{1, 2, 3}, []int64{1, 2, 3}},
			},
			want: &Tensor{
				Dtype:       framework.DataType_DT_INT64,
				TensorShape: NewShape([]int64{3, 3}),
				Int64Val:    []int64{1, 2, 3, 1, 2, 3, 1, 2, 3},
			},
		},
		{
			name: "3d array of int64",
			args: args{
				value: [][][]int64{[][]int64{[]int64{1, 2, 3}, []int64{1, 2, 3}, []int64{1, 2, 3}}},
			},
			want: &Tensor{
				Dtype:       framework.DataType_DT_INT64,
				TensorShape: NewShape([]int64{1, 3, 3}),
				Int64Val:    []int64{1, 2, 3, 1, 2, 3, 1, 2, 3},
			},
		},
		{
			name: "array of strings",
			args: args{
				value: []string{"foo"},
			},
			want: &Tensor{
				Dtype:       framework.DataType_DT_STRING,
				TensorShape: NewShape([]int64{1}),
				StringVal:   [][]byte{[]byte("foo")},
			},
		},
		{
			name: "single of string (scalar)",
			args: args{
				value: "foo",
			},
			want: &Tensor{
				Dtype:       framework.DataType_DT_STRING,
				TensorShape: NewShape([]int64{}),
				StringVal:   [][]byte{[]byte("foo")},
			},
		},
		{
			name: "2d array of strings",
			args: args{
				value: [][]string{[]string{"foo"}},
			},
			want: &Tensor{
				Dtype:       framework.DataType_DT_STRING,
				TensorShape: NewShape([]int64{1, 1}),
				StringVal:   [][]byte{[]byte("foo")},
			},
		},
		{
			name: "3d array of strings",
			args: args{
				value: [][][]string{[][]string{[]string{"foo"}}},
			},
			want: &Tensor{
				Dtype:       framework.DataType_DT_STRING,
				TensorShape: NewShape([]int64{1, 1, 1}),
				StringVal:   [][]byte{[]byte("foo")},
			},
		},
		{
			name: "array of byte slices",
			args: args{
				value: [][]byte{[]byte("foo")},
			},
			want: &Tensor{
				Dtype:       framework.DataType_DT_STRING,
				TensorShape: NewShape([]int64{1}),
				StringVal:   [][]byte{[]byte("foo")},
			},
		},
		{
			name: "array of float64",
			args: args{
				value: []float64{1.5, 2.5},
			},
			want: &Tensor{
				Dtype:       framework.DataType_DT_DOUBLE,
				TensorShape: NewShape([]int64{2}),
				DoubleVal:   []float64{1.5, 2.5},
			},
		},
		{
			name: "2d array of float64",
			args: args{
				value: [][]float64{[]float64{1, 2, 3}, []float64{1, 2, 3}, []float64{1, 2, 3}},
			},
			want: &Tensor{
				Dtype:       framework.DataType_DT_DOUBLE,
				TensorShape: NewShape([]int64{3, 3}),
				DoubleVal:   []float64{1, 2, 3, 1, 2, 3, 1, 2, 3},
			},
		},
		{
			name: "3d array of float64",
			args: args{
				value: [][][]float64{[][]float64{[]float64{1, 2, 3}, []float64{1, 2, 3}, []float64{1, 2, 3}}},
			},
			want: &Tensor{
				Dtype:       framework.DataType_DT_DOUBLE,
				TensorShape: NewShape([]int64{1, 3, 3}),
				DoubleVal:   []float64{1, 2, 3, 1, 2, 3, 1, 2, 3},
			},
		},
		{
			name: "array of float32",
			args: args{
				value: []float32{1.5, 2.5},
			},
			want: &Tensor{
				Dtype:       framework.DataType_DT_FLOAT,
				TensorShape: NewShape([]int64{2}),
				FloatVal:    []float32{1.5, 2.5},
			},
		},
		{
			name: "2d array of float32",
			args: args{
				value: [][]float32{[]float32{1, 2, 3}, []float32{1, 2, 3}, []float32{1, 2, 3}},
			},
			want: &Tensor{
				Dtype:       framework.DataType_DT_FLOAT,
				TensorShape: NewShape([]int64{3, 3}),
				FloatVal:    []float32{1, 2, 3, 1, 2, 3, 1, 2, 3},
			},
		},
		{
			name: "3d array of float32",
			args: args{
				value: [][][]float32{[][]float32{[]float32{1, 2, 3}, []float32{1, 2, 3}, []float32{1, 2, 3}}},
			},
			want: &Tensor{
				Dtype:       framework.DataType_DT_FLOAT,
				TensorShape: NewShape([]int64{1, 3, 3}),
				FloatVal:    []float32{1, 2, 3, 1, 2, 3, 1, 2, 3},
			},
		},
		{
			name: "4d array of float32",
			args: args{
				value: [][][][]float32{[][][]float32{[][]float32{[]float32{1, 2, 3}}}},
			},
			want: &Tensor{
				Dtype:       framework.DataType_DT_FLOAT,
				TensorShape: NewShape([]int64{1, 1, 1, 3}),
				FloatVal:    []float32{1, 2, 3},
			},
		},
		{
			name: "array of bool",
			args: args{
				value: []bool{true, false},
			},
			want: &Tensor{
				Dtype:       framework.DataType_DT_BOOL,
				TensorShape: NewShape([]int64{2}),
				BoolVal:     []bool{true, false},
			},
		},
		{
			name: "2d array of bool",
			args: args{
				value: [][]bool{[]bool{true, false, true}, []bool{true, false, true}, []bool{true, false, true}},
			},
			want: &Tensor{
				Dtype:       framework.DataType_DT_BOOL,
				TensorShape: NewShape([]int64{3, 3}),
				BoolVal:     []bool{true, false, true, true, false, true, true, false, true},
			},
		},
		{
			name: "3d array of bool",
			args: args{
				value: [][][]bool{[][]bool{[]bool{true, false, true}, []bool{true, false, true}, []bool{true, false, true}}},
			},
			want: &Tensor{
				Dtype:       framework.DataType_DT_BOOL,
				TensorShape: NewShape([]int64{1, 3, 3}),
				BoolVal:     []bool{true, false, true, true, false, true, true, false, true},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := NewTensor(tt.args.value)
			if (err != nil) != tt.wantErr {
				t.Errorf("NewTensor() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("NewTensor() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestNewShape(t *testing.T) {
	type args struct {
		dims []int64
	}
	tests := []struct {
		name string
		args args
		want *Shape
	}{
		{
			name: "shape",
			args: args{
				dims: []int64{1, 2, 3},
			},
			want: &framework.TensorShapeProto{
				Dim: []*framework.TensorShapeProto_Dim{
					&framework.TensorShapeProto_Dim{
						Size_: int64(1),
					},
					&framework.TensorShapeProto_Dim{
						Size_: int64(2),
					},
					&framework.TensorShapeProto_Dim{
						Size_: int64(3),
					},
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := NewShape(tt.args.dims); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("NewShape() = %v, want %v", got, tt.want)
			}
		})
	}
}

func testToTensorAndBack(val interface{}) bool {
	t := mustTensor(NewTensor(val))
	return reflect.DeepEqual(val, ValueFromTensor(t))
}

func Test_testToTensorAndBack(t *testing.T) {
	type args struct {
		val interface{}
	}
	tests := []struct {
		name string
		args args
		want bool
	}{
		{
			name: "int",
			args: args{
				val: int64(1),
			},
			want: true,
		},
		{
			name: "1d array",
			args: args{
				val: []int32{1, 2},
			},
			want: true,
		},
		{
			name: "1d float array",
			args: args{
				val: []float32{1, 2},
			},
			want: true,
		},
		{
			name: "1d double array",
			args: args{
				val: []float64{1, 2},
			},
			want: true,
		},
		{
			name: "1d int64 array",
			args: args{
				val: []int64{1, 2},
			},
			want: true,
		},
		{
			name: "1d uint64 array",
			args: args{
				val: []uint64{1, 2},
			},
			want: true,
		},
		{
			name: "1d uint32 array",
			args: args{
				val: []uint32{1, 2},
			},
			want: true,
		},
		{
			name: "1d string array",
			args: args{
				val: []string{"foo"},
			},
			want: true,
		},
		{
			name: "2d array",
			args: args{
				val: [][]int32{[]int32{1, 2}, []int32{3, 4}},
			},
			want: true,
		},
		{
			name: "3d array",
			args: args{
				val: [][][]bool{[][]bool{[]bool{true, false, true}, []bool{true, false, true}, []bool{true, false, true}}},
			},
			want: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := testToTensorAndBack(tt.args.val); got != tt.want {
				t.Errorf("testToTensorAndBack() = %v, want %v", got, tt.want)
			}
		})
	}
}
