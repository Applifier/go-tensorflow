package typeconv

import (
	"reflect"
	"testing"
)

func Test_convertInterfaceSliceToTypedSlice(t *testing.T) {
	type args struct {
		interfaceSlice []interface{}
	}
	tests := []struct {
		name    string
		args    args
		want    interface{}
		wantErr bool
	}{
		{
			name: "int slice",
			args: args{
				interfaceSlice: []interface{}{
					1,
					2,
					3,
				},
			},
			want: []int{1, 2, 3},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ConvertInterfaceSliceToTypedSlice(tt.args.interfaceSlice)
			if (err != nil) != tt.wantErr {
				t.Errorf("convertInterfaceSliceToTypedSlice() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("convertInterfaceSliceToTypedSlice() = %v, want %v", got, tt.want)
			}
		})
	}
}
