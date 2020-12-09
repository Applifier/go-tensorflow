package utils

import (
	"io"

	"github.com/Unity-Technologies/go-tensorflow/types/tensorflow_serving"
)

// PredictionLogReader reads tensorflow_serving.PredictionLog from a given io.Reader
type PredictionLogReader struct {
	tfRecordReader *TFRecordReader
}

// NewPredictionLogReader returns PredictionLogReader for a given io.Reader
func NewPredictionLogReader(r io.Reader) *PredictionLogReader {
	return &PredictionLogReader{
		tfRecordReader: NewTFRecordReader(r),
	}
}

// Next advances PredictionLogReader or return io.EOF
func (plr *PredictionLogReader) Next() error {
	return plr.tfRecordReader.Next()
}

// Unmarshal unmarshals current PredictionLog from the reader
func (plr *PredictionLogReader) Unmarshal(out *tensorflow_serving.PredictionLog) error {
	b := plr.tfRecordReader.Bytes()
	return out.Unmarshal(b)
}
