// Package serving is a client for Tensorflow Serving written in Go
package serving

import (
	"context"
	"io"

	serving "github.com/Applifier/go-tensorflow/types/tensorflow_serving"
	protobufTypes "github.com/gogo/protobuf/types"
	"google.golang.org/grpc"
)

// ModelPredictionClient represents a prediction client for a single model
type ModelPredictionClient interface {
	// Classify.
	Classify(ctx context.Context, input *serving.Input, opts ...grpc.CallOption) (*serving.ClassificationResponse, error)
	// Regress.
	Regress(ctx context.Context, input *serving.Input, opts ...grpc.CallOption) (*serving.RegressionResponse, error)
	// Predict -- provides access to loaded TensorFlow model.
	Predict(ctx context.Context, inputs TensorMap, outputFilter []string, opts ...grpc.CallOption) (*serving.PredictResponse, error)
	// GetModelMetadata - provides access to metadata for loaded models.
	GetModelMetadata(ctx context.Context, input *serving.Input, opts ...grpc.CallOption) (*serving.GetModelMetadataResponse, error)

	io.Closer
}

type modelPredictionClient struct {
	spec *serving.ModelSpec

	cli serving.PredictionServiceClient
}

// NewModelPredictionClientFromAddr returns a new client for a given TensorFlow Serving address
func NewModelPredictionClientFromAddr(serverAddr string, name string, signatureName string) (ModelPredictionClient, error) {
	conn, err := grpc.Dial(serverAddr, grpc.WithInsecure())
	if err != nil {
		return nil, err
	}

	return NewModelPredictionClientFromConn(conn, name, signatureName), nil
}

// NewModelPredictionClientFromConn returns a new client from a GRPC client connection
func NewModelPredictionClientFromConn(conn *grpc.ClientConn, name string, signatureName string) ModelPredictionClient {
	return NewModelPredictionClientFromPredictionServiceClient(serving.NewPredictionServiceClient(conn), name, signatureName)
}

// NewModelPredictionClientFromPredictionServiceClient returns a new client from a serving PredictionServiceClient
func NewModelPredictionClientFromPredictionServiceClient(cli serving.PredictionServiceClient, name string, signatureName string) ModelPredictionClient {
	return &modelPredictionClient{
		spec: &serving.ModelSpec{
			Name:          name,
			SignatureName: signatureName,
		},
		cli: cli,
	}
}

// SetVersion sets the used model version (defaults to latest)
func (client *modelPredictionClient) SetVersion(version int) {
	client.spec.Version = &protobufTypes.Int64Value{
		Value: int64(version),
	}
}

func (client *modelPredictionClient) Classify(ctx context.Context, input *serving.Input, opts ...grpc.CallOption) (*serving.ClassificationResponse, error) {
	// TODO optimize by memory pooling
	return client.cli.Classify(ctx, &serving.ClassificationRequest{
		ModelSpec: client.spec,
		Input:     input,
	}, opts...)
}

func (client *modelPredictionClient) Regress(ctx context.Context, input *serving.Input, opts ...grpc.CallOption) (*serving.RegressionResponse, error) {
	// TODO optimize by memory pooling
	return client.cli.Regress(ctx, &serving.RegressionRequest{
		ModelSpec: client.spec,
		Input:     input,
	}, opts...)
}

func (client *modelPredictionClient) Predict(ctx context.Context, inputs TensorMap, outputFilter []string, opts ...grpc.CallOption) (*serving.PredictResponse, error) {
	// TODO optimize by memory pooling
	return client.cli.Predict(ctx, &serving.PredictRequest{
		ModelSpec:    client.spec,
		Inputs:       inputs,
		OutputFilter: outputFilter,
	}, opts...)
}

func (client *modelPredictionClient) GetModelMetadata(ctx context.Context, input *serving.Input, opts ...grpc.CallOption) (*serving.GetModelMetadataResponse, error) {
	// TODO optimize by memory pooling
	return client.cli.GetModelMetadata(ctx, &serving.GetModelMetadataRequest{
		ModelSpec: client.spec,
	}, opts...)
}

func (client *modelPredictionClient) Close() error {
	if closer, ok := client.cli.(io.Closer); ok {
		return closer.Close()
	}

	return nil
}
