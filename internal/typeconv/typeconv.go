package typeconv

import (
	"errors"
	"reflect"
)

// ErrInconsistentTypes returned when a given slice of has inconsistent types of items
var ErrInconsistentTypes = errors.New("slice items have inconsistent types")

// ConvertInterfaceSliceToTypedSlice converts interface slice into a types go slice
func ConvertInterfaceSliceToTypedSlice(interfaceSlice []interface{}) (interface{}, error) {
	if len(interfaceSlice) == 0 {
		return nil, nil
	}

	item := interfaceSlice[0]
	firstItemType := reflect.TypeOf(item)

	newSliceType := reflect.SliceOf(firstItemType)
	newSlice := reflect.MakeSlice(newSliceType, len(interfaceSlice), len(interfaceSlice))

	for i, item := range interfaceSlice {
		itemType := reflect.TypeOf(item)
		itemValue := reflect.ValueOf(item)

		if itemType != firstItemType {
			return nil, ErrInconsistentTypes
		}

		indexVal := newSlice.Index(i)
		indexVal.Set(itemValue)
	}

	return newSlice.Interface(), nil
}
