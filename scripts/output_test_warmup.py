import tensorflow as tf

from tensorflow_serving.apis import classification_pb2
from tensorflow_serving.apis import inference_pb2
from tensorflow_serving.apis import model_pb2
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_log_pb2
from tensorflow_serving.apis import regression_pb2


with tf.io.TFRecordWriter("../testdata/tf_serving_warmup_requests") as writer:
    predict_request = predict_pb2.PredictRequest()
    predict_request.model_spec.name = 'some_model'
    predict_request.model_spec.signature_name = 'some_signature'

    example_proto = tf.train.Example(
        features=tf.train.Features(
            feature={}
        )
    )

    predict_request.inputs['inputs'].CopyFrom(
        tf.make_tensor_proto([example_proto.SerializeToString()]))

    log = prediction_log_pb2.PredictionLog(
        predict_log=prediction_log_pb2.PredictLog(request=predict_request))
    writer.write(log.SerializeToString())
