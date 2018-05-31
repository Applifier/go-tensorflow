package main

import (
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

func makeTensorFromImage(imageBytes []byte, imageFormat string) (*tf.Tensor, error) {
	tensor, err := tf.NewTensor(string(imageBytes))
	if err != nil {
		return nil, err
	}
	graph, input, output, err := makeTransformImageGraph(imageFormat)
	if err != nil {
		return nil, err
	}
	session, err := tf.NewSession(graph, nil)
	if err != nil {
		return nil, err
	}
	defer session.Close()
	normalized, err := session.Run(
		map[tf.Output]*tf.Tensor{input: tensor},
		[]tf.Output{output},
		nil)
	if err != nil {
		return nil, err
	}
	return normalized[0], nil
}

// Creates a graph to decode, rezise and normalize an image
func makeTransformImageGraph(imageFormat string) (graph *tf.Graph, input, output tf.Output, err error) {
	s := op.NewScope()
	input = op.Placeholder(s, tf.String)

	// Decode PNG or JPEG
	if imageFormat == "png" {
		output = op.DecodePng(s, input, op.DecodePngChannels(3))
	} else {
		output = op.DecodeJpeg(s, input, op.DecodeJpegChannels(3))
	}

	output = op.ExpandDims(s,
		op.Cast(s, output, tf.Uint8),
		op.Const(s.SubScope("make_batch"), int32(0)))

	graph, err = s.Finalize()

	return graph, input, output, err
}
