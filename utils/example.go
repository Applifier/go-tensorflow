package utils

import (
	"fmt"
	"reflect"

	"github.com/Applifier/go-tensorflow/internal/typeconv"
	"github.com/Applifier/go-tensorflow/types/tensorflow/core/example"
)

// An Example is a mostly-normalized data format for storing data for
// training and inference.  It contains a key-value store (features); where
// each key (string) maps to a Feature message (which is oneof packed BytesList,
// FloatList, or Int64List).
type Example = example.Example

// Feature contains Lists which may hold zero or more values.
type Feature = example.Feature

// NewExampleFromMap returns a new Example based on a given map of go values
func NewExampleFromMap(m map[string]interface{}) (*Example, error) {
	featureMap := make(map[string]*example.Feature, len(m))

	example := &Example{
		Features: &example.Features{
			Feature: featureMap,
		},
	}

	for name, val := range m {
		f, err := NewFeature(val)
		if err != nil {
			return nil, err
		}
		featureMap[name] = f
	}

	return example, nil
}

// NewFeature returns a Feature from a given go value
func NewFeature(val interface{}) (*Feature, error) {
	switch v := val.(type) {
	case []byte:
		return &Feature{
			Kind: &example.Feature_BytesList{
				BytesList: &example.BytesList{
					Value: [][]byte{v},
				},
			},
		}, nil
	case [][]byte:
		return &Feature{
			Kind: &example.Feature_BytesList{
				BytesList: &example.BytesList{
					Value: v,
				},
			},
		}, nil
	case string:
		return &Feature{
			Kind: &example.Feature_BytesList{
				BytesList: &example.BytesList{
					Value: [][]byte{[]byte(v)},
				},
			},
		}, nil
	case []string:
		byteSliceSlice := make([][]byte, 0, len(v))
		for _, str := range v {
			byteSliceSlice = append(byteSliceSlice, []byte(str))
		}
		return NewFeature(byteSliceSlice)
	case float64:
		return &Feature{
			Kind: &example.Feature_FloatList{
				FloatList: &example.FloatList{
					Value: []float32{float32(v)},
				},
			},
		}, nil
	case float32:
		return &Feature{
			Kind: &example.Feature_FloatList{
				FloatList: &example.FloatList{
					Value: []float32{v},
				},
			},
		}, nil
	case []float32:
		return &Feature{
			Kind: &example.Feature_FloatList{
				FloatList: &example.FloatList{
					Value: v,
				},
			},
		}, nil
	case int64:
		return &Feature{
			Kind: &example.Feature_Int64List{
				Int64List: &example.Int64List{
					Value: []int64{v},
				},
			},
		}, nil
	case []int64:
		return &Feature{
			Kind: &example.Feature_Int64List{
				Int64List: &example.Int64List{
					Value: v,
				},
			},
		}, nil
	case map[string]interface{}:
		ex, err := NewExampleFromMap(v)
		if err != nil {
			return nil, err
		}

		b, err := ex.Marshal()
		if err != nil {
			return nil, err
		}

		return &Feature{
			Kind: &example.Feature_BytesList{
				BytesList: &example.BytesList{
					Value: [][]byte{b},
				},
			},
		}, nil
	case []map[string]interface{}:
		values := make([][]byte, len(v))

		for i, m := range v {
			ex, err := NewExampleFromMap(m)
			if err != nil {
				return nil, err
			}

			b, err := ex.Marshal()
			if err != nil {
				return nil, err
			}
			values[i] = b
		}

		return &Feature{
			Kind: &example.Feature_BytesList{
				BytesList: &example.BytesList{
					Value: values,
				},
			},
		}, nil
	case []interface{}:
		arr, err := typeconv.ConvertInterfaceSliceToTypedSlice(v)
		if err != nil {
			return nil, err
		}
		return NewFeature(arr)
	default:
		return nil, fmt.Errorf("unsupported type %v", reflect.TypeOf(val))
	}
}
