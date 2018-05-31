package serving

import (
	"fmt"
	"reflect"

	"github.com/Applifier/go-tensorflow/types/tensorflow/core/framework"
)

// Tensor a tensorflow tensor
type Tensor = framework.TensorProto

// TensorMap map of tensors
type TensorMap = map[string]*framework.TensorProto

// Shape tensor shape
type Shape = framework.TensorShapeProto

// ShapeDim shape dimension
type ShapeDim = framework.TensorShapeProto_Dim

// NewShape creates a new Shape
func NewShape(dims []int64) *Shape {
	dim := make([]*framework.TensorShapeProto_Dim, len(dims))
	for i, size := range dims {
		dim[i] = &framework.TensorShapeProto_Dim{
			Size_: size,
		}
	}
	return &Shape{
		Dim: dim,
	}
}

// NewTensor returns a Tensor for a given go value
func NewTensor(value interface{}) (*Tensor, error) {
	if tensor, ok := value.(*Tensor); ok {
		return tensor, nil
	}

	// TODO figure out if this actually causes more issues that it solves
	if byteSliceSlize, ok := value.([][]byte); ok {
		newValue := make([]string, len(byteSliceSlize))
		for i, b := range byteSliceSlize {
			// TODO optimize this for alloc
			newValue[i] = string(b)
		}

		value = newValue
	}

	val := reflect.ValueOf(value)
	shape, dataType, err := shapeAndDataTypeOf(val)
	if err != nil {
		return nil, err
	}
	flattened := numElements(shape)

	// TODO optimize by memory pooling
	tensor := &Tensor{
		Dtype:       dataType,
		TensorShape: NewShape(shape),
	}

	switch dataType {
	case framework.DataType_DT_FLOAT:
		tensor.FloatVal = singleDimFloat32Slice(value, shape, flattened)
	case framework.DataType_DT_DOUBLE:
		tensor.DoubleVal = singleDimFloat64Slice(value, shape, flattened)
	case framework.DataType_DT_INT32:
		tensor.IntVal = singleDimInt32Slice(value, shape, flattened)
	case framework.DataType_DT_UINT32:
		tensor.Uint32Val = singleDimUInt32Slice(value, shape, flattened)
	case framework.DataType_DT_UINT8:
		tensor.IntVal = singleDimUInt8Slice(value, shape, flattened)
	case framework.DataType_DT_STRING:
		flattenedArr := singleDimStringSlice(value, shape, flattened)
		if flattenedArr != nil {
			value = flattenedArr
		}
		switch v := value.(type) {
		case []string:
			byteArrArr := make([][]byte, len(v))
			for i, s := range v {
				// TODO optimize for memory consumtion
				byteArrArr[i] = []byte(s)
			}
			tensor.StringVal = byteArrArr
		}

	case framework.DataType_DT_INT64:
		tensor.Int64Val = singleDimInt64Slice(value, shape, flattened)
	case framework.DataType_DT_UINT64:
		tensor.Uint64Val = singleDimUInt64Slice(value, shape, flattened)
	case framework.DataType_DT_BOOL:
		tensor.BoolVal = singleDimBoolSlice(value, shape, flattened)
	case framework.DataType_DT_COMPLEX64:
		vals := value.([]complex64)
		scomplex := make([]float32, len(vals)*2)

		for i := 0; i < len(vals); i += 2 {
			scomplex[i] = real(vals[i])
			scomplex[i+1] = real(vals[i])
		}

		tensor.ScomplexVal = scomplex
	case framework.DataType_DT_COMPLEX128:
		vals := value.([]complex128)
		dcomplex := make([]float64, len(vals)*2)

		for i := 0; i < len(vals); i += 2 {
			dcomplex[i] = real(vals[i])
			dcomplex[i+1] = real(vals[i])
		}

		tensor.DcomplexVal = dcomplex

	default:
		return nil, fmt.Errorf("unsupported type %v", val.Type())
	}

	return tensor, nil
}

// ShapeFromTensor returns shape from a tensor
func ShapeFromTensor(t *Tensor) []int64 {
	dims := make([]int64, 0, len(t.TensorShape.Dim))
	for _, d := range t.TensorShape.Dim {
		dims = append(dims, d.Size_)
	}
	return dims
}

// ValueFromTensor returns value from a given tensor
func ValueFromTensor(t *Tensor) interface{} {
	typ := typeOf(t.Dtype, t.TensorShape.Dim)

	dims := ShapeFromTensor(t)

	val := reflect.New(typ)

	switch t.Dtype {
	case framework.DataType_DT_FLOAT:
		arr := t.FloatVal
		populateMultiDimensionalSlice(val, dims, func(i int) interface{} {
			return arr[i]
		})
	case framework.DataType_DT_DOUBLE:
		arr := t.DoubleVal
		populateMultiDimensionalSlice(val, dims, func(i int) interface{} {
			return arr[i]
		})
	case framework.DataType_DT_INT32:
		arr := t.IntVal
		populateMultiDimensionalSlice(val, dims, func(i int) interface{} {
			return arr[i]
		})
	case framework.DataType_DT_UINT32:
		arr := t.Uint32Val
		populateMultiDimensionalSlice(val, dims, func(i int) interface{} {
			return arr[i]
		})
	case framework.DataType_DT_STRING:
		arr := t.StringVal
		populateMultiDimensionalSlice(val, dims, func(i int) interface{} {
			return string(arr[i])
		})
	case framework.DataType_DT_INT64:
		arr := t.Int64Val
		populateMultiDimensionalSlice(val, dims, func(i int) interface{} {
			return arr[i]
		})
	case framework.DataType_DT_UINT64:
		arr := t.Uint64Val
		populateMultiDimensionalSlice(val, dims, func(i int) interface{} {
			return arr[i]
		})
	case framework.DataType_DT_BOOL:
		arr := t.BoolVal
		populateMultiDimensionalSlice(val, dims, func(i int) interface{} {
			return arr[i]
		})
	case framework.DataType_DT_COMPLEX64:
		fallthrough
	case framework.DataType_DT_COMPLEX128:
		fallthrough
	default:
		panic(fmt.Errorf("unsupported type %v", t.Dtype))
	}

	return reflect.Indirect(val).Interface()
}

func numElements(shape []int64) int64 {
	n := int64(1)
	for _, d := range shape {
		n *= d
	}
	return n
}

// shapeAndDataTypeOf returns the data type and shape of the Tensor
// corresponding to a Go type.
func shapeAndDataTypeOf(val reflect.Value) (shape []int64, dt framework.DataType, err error) {
	typ := val.Type()
	for typ.Kind() == reflect.Array || typ.Kind() == reflect.Slice {
		shape = append(shape, int64(val.Len()))
		if val.Len() > 0 {
			// In order to check tensor structure properly in general case we need to iterate over all slices of the tensor to check sizes match
			// Since we already going to iterate over all elements in encodeTensor() let's
			// 1) do the actual check in encodeTensor() to save some cpu cycles here
			// 2) assume the shape is represented by lengths of elements with zero index in each dimension
			val = val.Index(0)
		}
		typ = typ.Elem()
	}
	for _, t := range types {
		if typ.Kind() == t.typ.Kind() {
			return shape, t.dataType, nil
		}
	}
	return shape, dt, fmt.Errorf("unsupported type %v", typ)
}

func populateMultiDimensionalSlice(val reflect.Value, dims []int64, getValue func(i int) interface{}) {
	typ := val.Type()
	if typ.Elem().Kind() == reflect.Slice {
		itemOffset := 0
		var traverse func(reflect.Type, []int64) reflect.Value
		traverse = func(typ reflect.Type, dims []int64) reflect.Value {
			sli := reflect.MakeSlice(typ, int(dims[0]), int(dims[0]))
			for i := 0; i < int(dims[0]); i++ {
				elemType := typ.Elem()
				if elemType.Kind() == reflect.Slice {
					sli.Index(i).Set(traverse(elemType, dims[1:]))
				} else {
					sli.Index(i).Set(reflect.ValueOf(getValue(itemOffset)))
					itemOffset++
				}
			}

			return sli
		}
		val.Elem().Set(traverse(typ.Elem(), dims))
	} else {
		val.Elem().Set(reflect.ValueOf(getValue(0)))
	}
}

// typeOf converts from a DataType and Shape to the equivalent Go type.
func typeOf(dt framework.DataType, shape []*framework.TensorShapeProto_Dim) reflect.Type {
	var ret reflect.Type
	for _, t := range types {
		if dt == t.dataType {
			ret = t.typ
			break
		}
	}
	if ret == nil {
		panic(fmt.Sprintf("DataType %v is not supported", dt))
	}
	for range shape {
		ret = reflect.SliceOf(ret)
	}
	return ret
}

var types = []struct {
	typ      reflect.Type
	dataType framework.DataType
}{
	{reflect.TypeOf(float32(0)), framework.DataType_DT_FLOAT},
	{reflect.TypeOf(float64(0)), framework.DataType_DT_DOUBLE},
	{reflect.TypeOf(int32(0)), framework.DataType_DT_INT32},
	{reflect.TypeOf(uint32(0)), framework.DataType_DT_UINT32},
	{reflect.TypeOf(int16(0)), framework.DataType_DT_INT16},
	{reflect.TypeOf(int8(0)), framework.DataType_DT_INT8},
	{reflect.TypeOf(uint8(0)), framework.DataType_DT_UINT8},
	{reflect.TypeOf(""), framework.DataType_DT_STRING},
	{reflect.TypeOf(complex(float32(0), float32(0))), framework.DataType_DT_COMPLEX64},
	{reflect.TypeOf(int64(0)), framework.DataType_DT_INT64},
	{reflect.TypeOf(uint64(0)), framework.DataType_DT_UINT64},
	{reflect.TypeOf(false), framework.DataType_DT_BOOL},
	{reflect.TypeOf(complex(float64(0), float64(0))), framework.DataType_DT_COMPLEX128},
	// TODO: support DT_RESOURCE representation in go.
	// TODO: support DT_VARIANT representation in go.
}

func flattenNDArray(val interface{}, resSlice interface{}) {

	resI := 0
	res := reflect.ValueOf(resSlice)
	var traverse func(val reflect.Value)
	traverse = func(val reflect.Value) {
		typ := val.Type()
		kind := typ.Elem().Kind()
		len := val.Len()

		if kind == reflect.Slice || kind == reflect.Array {
			for i := 0; i < len; i++ {
				traverse(val.Index(i))
			}

		} else {
			for i := 0; i < len; i++ {
				res.Index(resI).Set(val.Index(i))
				resI++
			}

		}
	}
	traverse(reflect.ValueOf(val))
}

func flatten2DInt32Slice(in [][]int32, out []int32) []int32 {
	for _, vals := range in {
		out = append(out, vals...)
	}
	return out
}

func flatten2DUInt8Slice(in [][]uint8, out []int32) []int32 {
	for _, vals := range in {
		arr := make([]int32, len(vals))
		for i, v := range vals {
			arr[i] = int32(v)
		}
		out = append(out, arr...)
	}
	return out
}

func flatten2DInt64Slice(in [][]int64, out []int64) []int64 {
	for _, vals := range in {
		out = append(out, vals...)
	}
	return out
}

func flatten2DUInt32Slice(in [][]uint32, out []uint32) []uint32 {
	for _, vals := range in {
		out = append(out, vals...)
	}
	return out
}

func flatten2DUInt64Slice(in [][]uint64, out []uint64) []uint64 {
	for _, vals := range in {
		out = append(out, vals...)
	}
	return out
}

func flatten2DFloat32Slice(in [][]float32, out []float32) []float32 {
	for _, vals := range in {
		out = append(out, vals...)
	}
	return out
}

func flatten2DFloat64Slice(in [][]float64, out []float64) []float64 {
	for _, vals := range in {
		out = append(out, vals...)
	}
	return out
}

func flatten2DStringSlice(in [][]string, out []string) []string {
	for _, vals := range in {
		out = append(out, vals...)
	}
	return out
}

func flatten2DBoolSlice(in [][]bool, out []bool) []bool {
	for _, vals := range in {
		out = append(out, vals...)
	}
	return out
}

func singleDimBoolSlice(val interface{}, dims []int64, flatN int64) []bool {
	switch v := val.(type) {
	case bool:
		return []bool{v}
	case []bool:
		return v
	case [][]bool:
		flat := make([]bool, 0, flatN)
		return flatten2DBoolSlice(v, flat)
	case [][][]bool:
		flat := make([]bool, 0, flatN)
		for _, dSlice := range v {
			flat = flatten2DBoolSlice(dSlice, flat)
		}

		return flat
	}

	flat := make([]bool, flatN, flatN)
	flattenNDArray(val, flat)

	return flat
}
func singleDimStringSlice(val interface{}, dims []int64, flatN int64) []string {
	switch v := val.(type) {
	case string:
		return []string{v}
	case []string:
		return v
	case [][]string:
		flat := make([]string, 0, flatN)
		return flatten2DStringSlice(v, flat)
	case [][][]string:
		flat := make([]string, 0, flatN)
		for _, dSlice := range v {
			flat = flatten2DStringSlice(dSlice, flat)
		}

		return flat
	}

	flat := make([]string, flatN, flatN)
	flattenNDArray(val, flat)

	return flat
}

func singleDimInt32Slice(val interface{}, dims []int64, flatN int64) []int32 {
	switch v := val.(type) {
	case int32:
		return []int32{v}
	case []int32:
		return v
	case [][]int32:
		flat := make([]int32, 0, flatN)
		return flatten2DInt32Slice(v, flat)
	case [][][]int32:
		flat := make([]int32, 0, flatN)
		for _, dSlice := range v {
			flat = flatten2DInt32Slice(dSlice, flat)
		}

		return flat
	}

	flat := make([]int32, flatN, flatN)
	flattenNDArray(val, flat)

	return flat
}

func singleDimUInt8Slice(val interface{}, dims []int64, flatN int64) []int32 {
	switch v := val.(type) {
	case uint8:
		return []int32{int32(v)}
	case []uint8:
		flat := make([]int32, flatN)
		for i, v := range v {
			flat[i] = int32(v)
		}
		return flat
	case [][]uint8:
		flat := make([]int32, 0, flatN)
		return flatten2DUInt8Slice(v, flat)
	case [][][]uint8:
		flat := make([]int32, 0, flatN)
		for _, dSlice := range v {
			flat = flatten2DUInt8Slice(dSlice, flat)
		}

		return flat
	}

	flat := make([]uint8, flatN, flatN)
	flattenNDArray(val, flat)

	// TODO optimize this
	returnArr := make([]int32, flatN)
	for i, v := range flat {
		returnArr[i] = int32(v)
	}

	return returnArr
}

func singleDimInt64Slice(val interface{}, dims []int64, flatN int64) []int64 {
	switch v := val.(type) {
	case int64:
		return []int64{v}
	case []int64:
		return v
	case [][]int64:
		flat := make([]int64, 0, flatN)
		return flatten2DInt64Slice(v, flat)
	case [][][]int64:
		flat := make([]int64, 0, flatN)
		for _, dSlice := range v {
			flat = flatten2DInt64Slice(dSlice, flat)
		}

		return flat
	}
	flat := make([]int64, flatN, flatN)
	flattenNDArray(val, flat)

	return flat
}

func singleDimUInt32Slice(val interface{}, dims []int64, flatN int64) []uint32 {
	switch v := val.(type) {
	case uint32:
		return []uint32{v}
	case []uint32:
		return v
	case [][]uint32:
		flat := make([]uint32, 0, flatN)
		return flatten2DUInt32Slice(v, flat)
	case [][][]uint32:
		flat := make([]uint32, 0, flatN)
		for _, dSlice := range v {
			flat = flatten2DUInt32Slice(dSlice, flat)
		}

		return flat
	}

	flat := make([]uint32, flatN, flatN)
	flattenNDArray(val, flat)

	return flat
}

func singleDimUInt64Slice(val interface{}, dims []int64, flatN int64) []uint64 {
	switch v := val.(type) {
	case uint64:
		return []uint64{v}
	case []uint64:
		return v
	case [][]uint64:
		flat := make([]uint64, 0, flatN)
		return flatten2DUInt64Slice(v, flat)
	case [][][]uint64:
		flat := make([]uint64, 0, flatN)
		for _, dSlice := range v {
			flat = flatten2DUInt64Slice(dSlice, flat)
		}

		return flat
	}

	flat := make([]uint64, flatN, flatN)
	flattenNDArray(val, flat)
	return flat
}

func singleDimFloat32Slice(val interface{}, dims []int64, flatN int64) []float32 {
	switch v := val.(type) {
	case float32:
		return []float32{v}
	case []float32:
		return v
	case [][]float32:
		flat := make([]float32, 0, flatN)
		return flatten2DFloat32Slice(v, flat)
	case [][][]float32:
		flat := make([]float32, 0, flatN)
		for _, dSlice := range v {
			flat = flatten2DFloat32Slice(dSlice, flat)
		}

		return flat
	}

	flat := make([]float32, flatN, flatN)
	flattenNDArray(val, flat)

	return flat
}

func singleDimFloat64Slice(val interface{}, dims []int64, flatN int64) []float64 {
	switch v := val.(type) {
	case float64:
		return []float64{v}
	case []float64:
		return v
	case [][]float64:
		flat := make([]float64, 0, flatN)
		return flatten2DFloat64Slice(v, flat)
	case [][][]float64:
		flat := make([]float64, 0, flatN)
		for _, dSlice := range v {
			flat = flatten2DFloat64Slice(dSlice, flat)
		}

		return flat
	}

	flat := make([]float64, flatN, flatN)
	flattenNDArray(val, flat)
	return flat
}
